#!/usr/bin/env python3
"""
MVTec AD evaluation for the TensorRT + fused-GEMM PatchCore pipeline.

This is the number the latency benchmark cannot give you: it proves the
speedup did not cost detection quality. It runs the SAME pipeline the
benchmark times, so the latency figures and the AUROC describe one system.

    python benchmark/eval_auroc.py \
        --onnx patchcore_feature_extractor.onnx \
        --ckpt results/Patchcore/MVTecAD/bottle/v0/weights/lightning/model.ckpt \
        --data-root ~/projects/data --category bottle --pixel

Expected for `bottle`: image-AUROC 1.000, pixel-AUROC ~0.98.
"""

import argparse
import os
import sys
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_trt_patchcore import (          # noqa: E402
    BoundSession, TorchSearcher, assert_input_sensitive, build_session,
    load_memory_bank, make_embedding,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default="datasets/MVTecAD")
    p.add_argument("--category", default="bottle")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--pixel", action="store_true",
                   help="also compute pixel-level AUROC (needs ground_truth/)")
    p.add_argument("--sigma", type=float, default=4.0,
                   help="gaussian smoothing of the anomaly map (anomalib: 4)")
    p.add_argument("--limit", type=int, default=0, help="debug: first N images")
    p.add_argument("--trt-cache", default="./trt_cache")
    p.add_argument("--no-trt", action="store_true")
    p.add_argument("--cuda-graph", action="store_true")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--pool", action="store_true", default=True)
    p.add_argument("--no-pool", dest="pool", action="store_false")
    return p.parse_args()


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def list_test_images(root, category):
    root = os.path.expanduser(root)
    test_dir = os.path.join(root, category, "test")
    if not os.path.isdir(test_dir):
        raise SystemExit(f"test dir not found: {test_dir}\n"
                         f"pass the folder that contains '{category}/'")
    items = []
    for defect in sorted(os.listdir(test_dir)):
        d = os.path.join(test_dir, defect)
        if not os.path.isdir(d):
            continue
        for f in sorted(glob(os.path.join(d, "*.png")) +
                        glob(os.path.join(d, "*.jpg"))):
            items.append((f, defect, 0 if defect == "good" else 1))
    if not items:
        raise SystemExit(f"no images under {test_dir}")
    return items


def load_image(path, size, normalize=True):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    x = x.permute(2, 0, 1)                                   # [3,H,W]
    if normalize:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = (x - mean) / std
    return x.unsqueeze(0)


def load_mask(img_path, root, category, size):
    """MVTec stores masks as ground_truth/<defect>/<stem>_mask.png."""
    root = os.path.expanduser(root)
    parts = img_path.split(os.sep)
    defect, stem = parts[-2], os.path.splitext(parts[-1])[0]
    if defect == "good":
        return torch.zeros(1, 1, size, size)
    base = os.path.join(root, category, "ground_truth", defect)
    for cand in (f"{stem}_mask.png", f"{stem}.png"):
        p = os.path.join(base, cand)
        if os.path.exists(p):
            m = Image.open(p).convert("L").resize((size, size), Image.NEAREST)
            m = torch.from_numpy((np.asarray(m) > 127).astype(np.float32))
            return m.view(1, 1, size, size)
    return None


# ----------------------------------------------------------------------
# anomaly map
# ----------------------------------------------------------------------
def gaussian_blur(x, sigma):
    k = 2 * int(4.0 * sigma + 0.5) + 1
    c = torch.arange(k, device=x.device, dtype=torch.float32) - k // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(x.dtype)
    x = F.conv2d(x, g.view(1, 1, 1, k), padding=(0, k // 2))
    x = F.conv2d(x, g.view(1, 1, k, 1), padding=(k // 2, 0))
    return x


def to_anomaly_map(dists, hw, size, sigma):
    """[N] patch distances -> [1,1,size,size] smoothed map."""
    m = dists.view(1, 1, hw, hw)
    m = F.interpolate(m, size=(size, size), mode="bilinear",
                      align_corners=False)
    return gaussian_blur(m, sigma)


# ----------------------------------------------------------------------
def main():
    args = get_args()
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = True
    print("[gpu]", torch.cuda.get_device_name(0))

    bank = load_memory_bank(args.ckpt)
    dim = bank.shape[1]

    sess = build_session(args)
    bs = BoundSession(sess, 1, args.size)
    assert_input_sensitive(bs)
    searcher = TorchSearcher(bank)
    print("[search]", searcher.name())

    sd = torch.load(args.ckpt, map_location="cpu",
                    weights_only=False)["state_dict"]
    thr = next((float(sd[k].reshape(-1)[0]) for k in sd
                if "image_threshold" in k and sd[k].numel() >= 1), None)
    print("[ckpt] image_threshold:", f"{thr:.4f}" if thr else "not found")

    items = list_test_images(args.data_root, args.category)
    if args.limit:
        items = items[:args.limit]
    print(f"[data] {len(items)} test images "
          f"({sum(1 for _, _, y in items if y == 0)} good / "
          f"{sum(1 for _, _, y in items if y == 1)} defect)")

    scores, labels, px_scores, px_labels = [], [], [], []
    hw, missing_masks = None, 0

    with torch.no_grad():
        for i, (path, _defect, label) in enumerate(items):
            bs.x.copy_(load_image(path, args.size,
                                  not args.no_normalize).cuda())
            f_hi, f_lo = bs.run()
            emb = make_embedding(f_hi, f_lo, dim, do_pool=args.pool)
            d = searcher.patch_dists(emb)                     # [N]

            scores.append(float(d.max()))
            labels.append(label)

            if args.pixel:
                hw = hw or int(round(d.numel() ** 0.5))
                amap = to_anomaly_map(d, hw, args.size, args.sigma)
                mask = load_mask(path, args.data_root, args.category, args.size)
                if mask is None:
                    missing_masks += 1
                else:
                    px_scores.append(amap.flatten().cpu().numpy())
                    px_labels.append(mask.flatten().numpy())

            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(items)}")

    labels, scores = np.array(labels), np.array(scores)

    # a degenerate score distribution means the pipeline ignored its input
    if float(scores.std()) < 1e-6:
        raise SystemExit(
            "\n[FATAL] every image produced the same score -- the pipeline "
            "is not\n        reading its input. This is a plumbing bug, not "
            "a model result.\n")

    img_auroc = roc_auc_score(labels, scores)

    print("=" * 52)
    print(f"category            {args.category}")
    print(f"images              {len(items)}")
    print(f"image-AUROC         {img_auroc:.4f}")
    if thr is not None:
        pred = (scores > thr).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        print(f"accuracy @thr       {(pred == labels).mean():.4f}")
        print(f"F1 @thr             {f1:.4f}   (TP {tp}  FP {fp}  FN {fn})")
    print(f"score range         good  {scores[labels == 0].min():.2f} .. "
          f"{scores[labels == 0].max():.2f}")
    print(f"                    bad   {scores[labels == 1].min():.2f} .. "
          f"{scores[labels == 1].max():.2f}")

    if args.pixel and px_scores:
        px_auroc = roc_auc_score(np.concatenate(px_labels),
                                 np.concatenate(px_scores))
        print(f"pixel-AUROC         {px_auroc:.4f}")
        if missing_masks:
            print(f"  ({missing_masks} images had no ground-truth mask)")
    print("=" * 52)

    if img_auroc < 0.90:
        print("\n[hint] Scores do vary, so the input plumbing is fine and the\n"
              "       search kernel is verified exact. Suspect preprocessing\n"
              "       drift instead: the resize (256x256 bilinear) and\n"
              "       ImageNet normalisation here must match what anomalib\n"
              "       used when it built the memory bank, and the ONNX export\n"
              "       must take the same layers. Try --no-normalize to see if\n"
              "       the score ordering flips.")


if __name__ == "__main__":
    main()

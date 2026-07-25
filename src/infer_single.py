import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from anomalib.models import Patchcore
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



DEVICE = "cuda"

parser = argparse.ArgumentParser(
    description="Score a single image with a trained PatchCore model.")
parser.add_argument("--image", required=True,
                    help="path to the image to score")
parser.add_argument("--ckpt",
                    default="results/Patchcore/MVTecAD/bottle/v0/"
                            "weights/lightning/model.ckpt",
                    help="trained PatchCore checkpoint")
parser.add_argument("--overlay", default="anomaly_overlay.png",
                    help="where to write the heat-map overlay")
args = parser.parse_args()

IMAGE_PATH = args.image
CHECKPOINT = args.ckpt


def main():
    model = Patchcore()
    ckpt = torch.load(CHECKPOINT, map_location="cuda", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"],
                                                strict=False)
    if missing or unexpected:
        print(f"[warn] missing={missing}  unexpected={unexpected}")

    model.to(DEVICE).eval()

    image = Image.open(IMAGE_PATH).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)

    score = float(out.pred_score[0])
    label = bool(out.pred_label[0])

    print(f"image  : {IMAGE_PATH}")
    print(f"score  : {score:.4f}   (anomalib normalised, 0-1)")
    print(f"verdict: {'ANOMALOUS' if label else 'normal'}")

    amap = out.anomaly_map[0, 0].detach().cpu().numpy()
    base = image.resize((256, 256))

    fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
    ax[0].imshow(base)
    ax[0].set_title("input")

    hm = ax[1].imshow(amap, cmap="turbo")
    ax[1].set_title("anomaly map")
    fig.colorbar(hm, ax=ax[1], fraction=0.046)

    ax[2].imshow(base)
    ax[2].imshow(amap, cmap="turbo", alpha=0.5)
    ax[2].set_title(f"overlay — {score:.3f} "
                    f"({'ANOMALOUS' if label else 'normal'})")

    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(args.overlay, dpi=130)
    print(f"saved  : {args.overlay}")


if __name__ == "__main__":
    main()

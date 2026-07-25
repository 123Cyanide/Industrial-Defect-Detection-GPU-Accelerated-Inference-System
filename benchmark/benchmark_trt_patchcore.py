#!/usr/bin/env python3
"""
GPU-resident PatchCore + TensorRT end-to-end latency benchmark.

Design
------
1. ONNX Runtime IO-Binding -> model input & outputs live in CUDA memory.
   A naive integration round-trips through numpy 4x per frame (~19 MB
   over PCIe), which costs more than the entire backbone.
2. Post-processing on GPU with correct spatial alignment
   (3x3 avg-pool + 2D nearest upsample BEFORE flattening).
3. NN search consumes a CUDA tensor directly:
      --search torch  -> fp16 tensor-core GEMM with ||b||^2 folded in
      --search faiss  -> faiss.contrib.torch_utils (device pointers)
4. Only one float scalar crosses PCIe per frame.

Correctness
-----------
ORT executes on its own CUDA stream, so a torch write into the bound
input buffer is NOT ordered against the engine reading it. We sync
explicitly. CUDA-graph capture is opt-in (--cuda-graph) because replay
can pin stale input state; assert_input_sensitive() catches that class
of bug regardless of which knobs are set.
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort


# ----------------------------------------------------------------------
# args
# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--search", choices=["faiss", "torch"], default="torch")
    p.add_argument("--trt-cache", default="./trt_cache")
    p.add_argument("--no-trt", action="store_true",
                   help="CUDA EP only, for an apples-to-apples baseline")
    p.add_argument("--cuda-graph", action="store_true",
                   help="opt-in. Replay can pin stale input state -- always "
                        "confirm the input-sensitivity check still passes")
    p.add_argument("--pool", action="store_true", default=True,
                   help="PatchCore 3x3 neighbourhood avg-pool (anomalib "
                        "applies this when building the memory bank)")
    p.add_argument("--no-pool", dest="pool", action="store_false")
    return p.parse_args()


# ----------------------------------------------------------------------
# memory bank
# ----------------------------------------------------------------------
def load_memory_bank(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    key = next(k for k in sd if k.endswith("memory_bank"))
    mb = sd[key].float().contiguous()
    print(f"[bank] {key}  shape={tuple(mb.shape)}  "
          f"{mb.numel() * 4 / 1e6:.1f} MB fp32")
    return mb


# ----------------------------------------------------------------------
# ORT session
# ----------------------------------------------------------------------
def build_session(args):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    use_graph = bool(getattr(args, "cuda_graph", False))
    cuda_opts = {"device_id": 0}
    trt_opts = {
        "device_id": 0,
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": args.trt_cache,
        "trt_builder_optimization_level": 5,
        "trt_cuda_graph_enable": use_graph,
    }

    providers = [] if args.no_trt else [("TensorrtExecutionProvider", trt_opts)]
    providers.append(("CUDAExecutionProvider", cuda_opts))

    sess = ort.InferenceSession(args.onnx, sess_options=so, providers=providers)
    print("[ort] providers:", sess.get_providers())
    print(f"[ort] cuda_graph={use_graph}")
    return sess


# ----------------------------------------------------------------------
# IO binding
# ----------------------------------------------------------------------
class BoundSession:
    """Wraps an ORT session so run() touches no host memory."""

    def __init__(self, sess, batch, size):
        self.sess = sess
        self.in_name = sess.get_inputs()[0].name
        self.out_names = [o.name for o in sess.get_outputs()]

        self.x = torch.zeros(batch, 3, size, size,
                             device="cuda", dtype=torch.float32).contiguous()

        # probe concrete output shapes once (host path, one time only)
        probe = sess.run(None, {self.in_name: self.x.cpu().numpy()})
        self.outs = []
        for arr in probe:
            self.outs.append(torch.empty(tuple(arr.shape), device="cuda",
                                         dtype=torch.float32).contiguous())
            print(f"[ort] output shape {tuple(arr.shape)}")

        self.io = sess.io_binding()
        self.io.bind_input(
            name=self.in_name, device_type="cuda", device_id=0,
            element_type=np.float32, shape=tuple(self.x.shape),
            buffer_ptr=self.x.data_ptr(),
        )
        for name, buf in zip(self.out_names, self.outs):
            self.io.bind_output(
                name=name, device_type="cuda", device_id=0,
                element_type=np.float32, shape=tuple(buf.shape),
                buffer_ptr=buf.data_ptr(),
            )

        # layer2 = the feature map with the larger spatial size
        order = sorted(range(len(self.outs)),
                       key=lambda i: -self.outs[i].shape[-1])
        self.hi, self.lo = order[0], order[1]

    def run(self):
        # torch wrote self.x on torch's stream; ORT reads it on its own.
        # Nothing orders those two without this.
        torch.cuda.synchronize()
        self.sess.run_with_iobinding(self.io)
        self.io.synchronize_outputs()
        return self.outs[self.hi], self.outs[self.lo]


def assert_input_sensitive(bs, tol=1e-6):
    """Two different inputs must produce different features.

    The check that was missing. The benchmark fed one fixed tensor 200
    times, so a pipeline silently ignoring its input would still have
    produced plausible latencies and a plausible anomaly score -- and did,
    until real images all came back with an identical score.
    """
    bs.x.normal_()
    a = bs.run()[0].clone()
    bs.x.normal_()
    b = bs.run()[0].clone()

    delta = (a - b).abs().max().item()
    if delta < tol:
        raise SystemExit(
            "\n[FATAL] the engine returned identical features for two "
            "different inputs.\n"
            "        The bound input buffer is not reaching it. If you "
            "passed --cuda-graph,\n"
            "        drop it: graph replay can pin the captured input "
            "state.\n")
    print(f"[check] input sensitivity OK (max feature delta {delta:.3f})")


# ----------------------------------------------------------------------
# post-processing
# ----------------------------------------------------------------------
def make_embedding(f_hi, f_lo, dim, do_pool=True):
    """
    f_hi: [B, C1, H, W] (layer2)   f_lo: [B, C2, h, w] (layer3)
    -> [B*H*W, C1+C2]

    Two things that are easy to get wrong here:
      * the 3x3 avg-pool (PatchCore local neighbourhood aggregation) --
        anomalib applies it before building the memory bank, so skipping
        it at inference biases every distance;
      * F.interpolate must run on the 4D tensor. Upsampling AFTER
        flatten(2) is a 1-D resample with mapping k -> k//4, whereas
        correct 2-D upsampling maps (i,j) -> (i//2, j//2). These differ,
        and the result is a spatially scrambled patch grid.
    """
    if do_pool:
        f_hi = F.avg_pool2d(f_hi, 3, stride=1, padding=1)
        f_lo = F.avg_pool2d(f_lo, 3, stride=1, padding=1)

    f_lo = F.interpolate(f_lo, size=f_hi.shape[-2:], mode="nearest")
    emb = torch.cat([f_hi, f_lo], dim=1)              # [B, C, H, W]
    emb = emb.permute(0, 2, 3, 1).reshape(-1, dim)    # [B*H*W, C]
    return emb.contiguous()


# ----------------------------------------------------------------------
# searchers
# ----------------------------------------------------------------------
class TorchSearcher:
    """
    Exact L2 nearest neighbour, ||b||^2 folded into the GEMM.

        q' = [q, 1]          b' = [-2b, ||b||^2]
        q' . b' = ||b||^2 - 2 q.b

    ||q||^2 is constant per row so it drops out of the argmin and is added
    back afterwards. One fp16 GEMM + one min reduction, no intermediate
    [N, M] elementwise passes.
    """

    def __init__(self, bank):
        b = bank.cuda()                                  # [M, D] fp32
        M, D = b.shape
        aug = torch.empty(M, D + 1, device="cuda", dtype=torch.half)
        aug[:, :D] = (-2.0 * b).half()
        aug[:, D] = (b * b).sum(1).half()                # ||b||^2
        self.aug = aug.t().contiguous()                  # [D+1, M]
        self.D, self.M = D, M

    @torch.no_grad()
    def patch_dists(self, q):
        """Per-patch nearest-neighbour L2 distance -> [N] fp32."""
        n = q.shape[0]
        qa = torch.empty(n, self.D + 1, device=q.device, dtype=torch.half)
        qa[:, :self.D] = q.half()
        qa[:, self.D] = 1.0
        part = torch.mm(qa, self.aug).min(dim=1).values.float()
        return (part + (q * q).sum(1)).clamp_min_(0).sqrt_()

    @torch.no_grad()
    def max_score(self, q):
        return self.patch_dists(q).max()

    def name(self):
        return f"torch-fp16-gemm-fused (M={self.M})"


class FaissSearcher:
    """FAISS GPU flat index fed with CUDA tensors (no host copies)."""

    def __init__(self, bank):
        import faiss
        import faiss.contrib.torch_utils  # noqa: F401  patches search()

        d = bank.shape[1]
        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(bank.numpy())

        self.res = faiss.StandardGpuResources()
        self.res.setTempMemory(256 * 1024 * 1024)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index, co)
        self.M = bank.shape[0]

    @torch.no_grad()
    def patch_dists(self, q):
        D, _ = self.index.search(q, 1)
        return D[:, 0].clamp_min(0).sqrt()

    @torch.no_grad()
    def max_score(self, q):
        return self.patch_dists(q).max()

    def name(self):
        return f"faiss-gpu-fp16-flat (M={self.M})"


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    args = get_args()
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("[gpu]", torch.cuda.get_device_name(0))

    bank = load_memory_bank(args.ckpt)
    dim = bank.shape[1]

    sess = build_session(args)
    bs = BoundSession(sess, args.batch, args.size)
    assert_input_sensitive(bs)

    searcher = (FaissSearcher(bank) if args.search == "faiss"
                else TorchSearcher(bank))
    print("[search]", searcher.name())

    bs.x.normal_()

    @torch.no_grad()
    def one_frame():
        f_hi, f_lo = bs.run()
        emb = make_embedding(f_hi, f_lo, dim, do_pool=args.pool)
        return searcher.max_score(emb)

    # ---- sanity: fp16 GPU path vs exact CPU FAISS ---------------------
    ref_emb = make_embedding(*bs.run(), dim, do_pool=args.pool)
    try:
        import faiss
        idx = faiss.IndexFlatL2(dim)
        idx.add(bank.numpy())
        D, _ = idx.search(ref_emb.cpu().numpy(), 1)
        exact = float(np.sqrt(np.maximum(D[:, 0], 0)).max())
        got = float(searcher.max_score(ref_emb))
        print(f"[check] exact={exact:.6f}  gpu={got:.6f}  "
              f"rel_err={abs(got - exact) / max(exact, 1e-9):.2e}")
    except Exception as e:                                   # noqa: BLE001
        print("[check] exactness check skipped:", e)

    # ---- warmup -------------------------------------------------------
    print(f"[warmup] {args.warmup} iters ...")
    for _ in range(args.warmup):
        one_frame()
    torch.cuda.synchronize()

    # ---- stage breakdown ---------------------------------------------
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    acc = np.zeros(2)
    for _ in range(50):
        ev[0].record()
        f_hi, f_lo = bs.run()
        emb = make_embedding(f_hi, f_lo, dim, do_pool=args.pool)
        ev[1].record()
        _ = searcher.max_score(emb)
        ev[2].record()
        torch.cuda.synchronize()
        acc += [ev[0].elapsed_time(ev[1]), ev[1].elapsed_time(ev[2])]
    acc /= 50
    print(f"[stage] backbone+postproc {acc[0]:.3f} ms | "
          f"nn-search {acc[1]:.3f} ms")

    # ---- end-to-end ---------------------------------------------------
    print(f"[bench] {args.iters} iters ...")
    times, score = [], None
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        score = one_frame().item()          # the only D2H, 4 bytes
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    t = np.array(times)
    print("=" * 46)
    print(f"batch={args.batch}  search={args.search}  pool={args.pool}  "
          f"cuda_graph={args.cuda_graph}")
    print(f"mean  {t.mean():7.3f} ms")
    print(f"p50   {np.percentile(t, 50):7.3f} ms")
    print(f"p95   {np.percentile(t, 95):7.3f} ms")
    print(f"p99   {np.percentile(t, 99):7.3f} ms")
    print(f"FPS   {1000.0 / t.mean() * args.batch:7.1f}  (images/s)")
    print(f"anomaly score {score:.6f}")
    print("=" * 46)


if __name__ == "__main__":
    main()

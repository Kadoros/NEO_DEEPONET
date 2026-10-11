"""Step-time probe for NEO (Ours-Large architecture) training on N-point volume samples.
    usage: python bench_neo_step.py [N] [B1,B2,...]
Random inputs, repo loss (PretrainTetTraining losses), fp16 autocast, AdamW; no results are kept.

    CUDA_VISIBLE_DEVICES=<gpu> python stage2/bench_neo_step.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
for var, sub in [("MPLCONFIGDIR", "mpl"), ("XDG_CACHE_HOME", ""), ("TMPDIR", "tmp"), ("TRITON_CACHE_DIR", "triton"),
                 ("TORCH_HOME", "torch"), ("CUDA_CACHE_PATH", "nv")]:
    os.environ[var] = os.path.join(ROOT, ".cache", sub)
    os.makedirs(os.environ[var], exist_ok=True)

import open3d  # noqa: F401  -- must be imported before h5py/torch/lightning (segfault otherwise, see NEO_PILOT.md)
import torch

from g2pt.training.pretrain_tet import PretrainTetTraining

CKPT = os.path.join(os.path.dirname(ROOT), "neo_check", "ckpts", "Ours-Large.ckpt")
cfg = torch.load(CKPT, map_location="cpu", weights_only=False)["hyper_parameters"]["cfg"]
m = PretrainTetTraining(cfg).cuda()
opt = torch.optim.AdamW(m.parameters(), 1e-4)
N, K = int(sys.argv[1]) if len(sys.argv) > 1 else 2048, 96
for B in ([int(b) for b in sys.argv[2].split(",")] if len(sys.argv) > 2 else (32, 64, 112)):
    x = torch.randn(B, N, 3, device="cuda") * 0.5
    mass = torch.rand(B, N, 1, device="cuda") + 0.5
    y = torch.linalg.qr(torch.randn(B, N, K, device="cuda"))[0] / mass.sqrt()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for it in range(8):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            y_hat, y_orig = m(x, mass)
        loss = sum(w * f(pred=y_hat.float(), target=y, mass=mass, y_original=y_orig)
                   for f, w in zip(m.losses, m.weights))
        opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    print(f"B={B}: {1e3 * sum(ts[3:]) / len(ts[3:]):.0f} ms/step, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB",
          flush=True)

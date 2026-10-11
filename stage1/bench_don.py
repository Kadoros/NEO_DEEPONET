"""Throughput probe: DON-matched (p=128) training steps for a fixed wall time; prints steps/s.
Used only to decide whether two runs per GPU raise total throughput (no results are kept).

    CUDA_VISIBLE_DEVICES=<gpu> python stage1/bench_don.py <seconds>
"""
import common  # noqa: F401

import sys
import time

import numpy as np
import torch

import train as T
import train_don_fair as DF

secs = float(sys.argv[1])
D, ids, _ = T.load_all()
m = DF.DeepONetNorm(128, DF.matched_alpha(128)).to(T.DEV)
opt = torch.optim.AdamW(m.parameters(), 1e-3)
rng = np.random.default_rng(0)
for phase, dur in (("warmup", 10.0), ("measure", secs)):
    torch.cuda.synchronize()
    t0, n = time.perf_counter(), 0
    while time.perf_counter() - t0 < dur:
        b, U, M = T.batch(D, rng.choice(ids["train"], T.BATCH, replace=False), "ex")
        loss = T.rel_err(m(b), U, M).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        n += 1
    torch.cuda.synchronize()
    if phase == "measure":
        print(f"steps/s {n / (time.perf_counter() - t0):.2f}  peak_mem_GB {torch.cuda.max_memory_allocated() / 1e9:.2f}",
              flush=True)

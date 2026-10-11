"""Gate experiment, our structure: raw NEO basis + g_theta + eps (stage-1 "L-neo + r_p + eps", unchanged classes)
-> stage3/gate/runs/ours_N<N>_s<seed>/{result.json, test.csv}

Model   stage1.models.LNeoResidual(eps="rp"):  u = sum_k g_theta(lambda_k t, lambda_k/lambda_1, z) <u0, phi_k>_M phi_k
        + eps_theta(r, r_p, F, x, t),  phi = raw volume-NEO (t4096) Rayleigh-Ritz basis (256), no refinement.
Training (stage-1 schedule): phase 1 g_theta only (LearnedSpectral) 3000 steps lr 3e-3; phase 2 the full model
        initialised from phase 1, 15000 steps lr 1e-3; AdamW, OneCycle, batch 8 shapes, loss = mean relative L2;
        best checkpoint on the val split (every 250 steps).
Data    N training samples = N (shape, IC) pairs: N/2 random train shapes (seeded) x their 2 ICs.
Test    per shape (B = 1, clouds may differ in size): relative L2 (lumped mass) per (IC, tau), mean.

    CUDA_VISIBLE_DEVICES=<gpu> python stage3/gate_ours.py --N 50 --seed 0
"""
import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "stage1"))
from models import LearnedSpectral, LNeoResidual, n_params  # noqa: E402

DATA = os.path.join(ROOT, "gate", "data")
RUNS = os.path.join(ROOT, "gate", "runs")
DEV = "cuda"
BATCH = 8
PHASES = [("L", 3000, 3e-3), ("L+eps", 15000, 1e-3)]
VAL_EVERY = 250
KEYS = ("x", "m", "F", "z", "Phi", "lam", "coef", "u0", "U", "r", "rp", "t")


def load_split(split, names=None):
    out = []
    for p in sorted(glob.glob(os.path.join(DATA, split, "*.npz"))):
        name = os.path.basename(p)[:-4]
        if names is not None and name not in names:
            continue
        z = np.load(p)
        out.append((name, {k: torch.tensor(z[k].astype(np.float32)) for k in KEYS}))
    return out


def to_batch(items):
    b = {}
    for k in KEYS:
        b[k] = torch.stack([it[k] for it in items]).to(DEV)
    b["m_pc"] = b.pop("m")
    b["coef_neo"] = b["coef"]
    return b, b["U"], b["m_pc"]


def rel_err(pred, U, m):
    w = m[:, None, None, :]
    return torch.sqrt((w * (pred - U) ** 2).sum(-1) / (w * U ** 2).sum(-1))                     # (B,T,I)


@torch.no_grad()
def evaluate(model, items):
    model.eval()
    rows = []
    for name, it in items:
        b, U, m = to_batch([it])
        e = rel_err(model(b), U, m)[0].cpu().numpy()                                            # (T,I)
        rows.append((name, e))
    return rows


def train_phase(model, train, val, steps, lr, seed, log, tag):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    best, best_val = copy.deepcopy(model.state_dict()), np.mean([e.mean() for _, e in evaluate(model, val)])
    for step in range(1, steps + 1):
        model.train()
        idx = rng.choice(len(train), min(BATCH, len(train)), replace=False)
        b, U, m = to_batch([train[i][1] for i in idx])
        loss = rel_err(model(b), U, m).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % VAL_EVERY == 0 or step == steps:
            v = np.mean([e.mean() for _, e in evaluate(model, val)])
            log.append(dict(phase=tag, step=step, train=loss.item(), val=float(v)))
            if v < best_val:
                best_val, best = v, copy.deepcopy(model.state_dict())
            print(f"[{tag}] step {step} train {loss.item():.4f} val {v:.4f}", flush=True)
    model.load_state_dict(best)
    return best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    out = os.path.join(RUNS, f"ours_N{a.N}_s{a.seed}")
    os.makedirs(out, exist_ok=True)
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(DATA, "train", "*.npz")))
    pick = sorted(np.random.default_rng([a.seed, a.N]).choice(len(names), a.N // 2, replace=False))
    train = load_split("train", {names[i] for i in pick})
    val = load_split("val")
    t0 = time.time()
    log = []
    m1 = LearnedSpectral(use_eps=False).to(DEV)
    train_phase(m1, train, val, PHASES[0][1], PHASES[0][2], a.seed, log, "L")
    model = LNeoResidual(eps="rp").to(DEV)
    model.g.load_state_dict(m1.g.state_dict())
    best_val = train_phase(model, train, val, PHASES[1][1], PHASES[1][2], a.seed, log, "L+eps")
    t_train = time.time() - t0
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    rows = []
    for split in ("test_in", "test_ood", "test_real"):
        items = load_split(split)
        torch.cuda.synchronize(); ti = time.time()
        res = evaluate(model, items)
        torch.cuda.synchronize(); t_inf = (time.time() - ti) / max(len(items), 1)
        for name, e in res:
            for j in range(e.shape[0]):
                for i in range(e.shape[1]):
                    rows.append(dict(split=split, shape=name, tau_idx=j, ic=i, rel=float(e[j, i]), t_infer_per_shape_s=t_inf))
    pd.DataFrame(rows).to_csv(os.path.join(out, "test.csv"), index=False)
    df = pd.DataFrame(rows)
    summ = df.groupby("split").rel.mean().to_dict()
    json.dump(dict(model="ours", N=a.N, seed=a.seed, n_train_shapes=len(train), params=n_params(model),
                   best_val=float(best_val), t_train_s=t_train, mean_rel=summ, log=log), open(os.path.join(out, "result.json"), "w"), indent=1)
    print(summ, f"train {t_train:.0f}s params {n_params(model)}")


if __name__ == "__main__":
    main()

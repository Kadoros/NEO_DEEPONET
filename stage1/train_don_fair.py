"""DON-base fairness check (stage 1 data, stage-1 results untouched) -> results_don_fair/

  * input normalisation: u0 standardised per (shape, IC) with the point-cloud-mass-weighted mean/std;
    the network predicts the standardised field and the output is mapped back with the same statistics.
    Coordinates keep the existing area normalisation.
  * DON-base-matched: all hidden widths scaled by one factor α so the parameter count matches model (c)
    (237,458) for each latent size p.
  * search: lr ∈ {3e-4, 1e-3, 3e-3} × p ∈ {64, 128, 256} on DON-base-matched, plus the original DON-base
    architecture (p=128) with normalisation at lr 1e-3.  Every run trains until the validation error
    plateaus (same rule as the earlier DON continuation): eval every 1000 steps, ReduceLROnPlateau
    (×0.5, patience 4), stop after 20 evals without >0.5 % relative improvement (cap 400k steps).
    The configuration with the lowest validation error is reported on test.

    CUDA_VISIBLE_DEVICES=<gpu> python stage1/train_don_fair.py --runs 0,3,6,9
    python stage1/train_don_fair.py --merge
"""
import common  # noqa: F401

import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import train as T
from common import RESULTS, SEED, TAUS
from models import log_t_feature, mlp, n_params

OUT = os.path.join(os.path.dirname(RESULTS), "results_don_fair")
TARGET_PARAMS = 237_458                       # model (c) L-neo+rp+eps
EVAL_EVERY, PATIENCE, MAX_STEPS, WARMUP = 1000, 20, 400_000, 1000


class DeepONetNorm(nn.Module):
    """DON-base with per-(shape, IC) standardisation of u0 and de-standardised output.
    Widths: point MLP 4-64α-128α-256α, head (512α)-256α-p, trunk 4-128α-128α-128α-p."""

    def __init__(self, p=128, alpha=1.0):
        super().__init__()
        w = lambda n: max(4, int(round(n * alpha)))
        self.point = mlp([4, w(64), w(128), w(256)])
        self.head = mlp([2 * w(256), w(256), p])
        self.trunk = mlp([4, w(128), w(128), w(128), p])
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, b):
        B, N, _ = b["x"].shape
        I, T_ = b["u0"].shape[1], b["t"].shape[1]
        wgt = (b["m_pc"] / b["m_pc"].sum(1, keepdim=True))[:, None, :]              # (B,1,N)
        mu = (wgt * b["u0"]).sum(-1, keepdim=True)                                 # (B,I,1)
        sd = torch.sqrt((wgt * (b["u0"] - mu) ** 2).sum(-1, keepdim=True)).clamp_min(1e-8)
        u0n = (b["u0"] - mu) / sd
        h = self.point(torch.cat([b["x"][:, None].expand(B, I, N, 3), u0n[..., None]], -1))
        br = self.head(torch.cat([(wgt[..., None] * h).sum(2), h.amax(2)], -1))    # (B,I,p)
        lt = log_t_feature(b["t"])[:, :, None, None].expand(B, T_, N, 1)
        tr = self.trunk(torch.cat([b["x"][:, None].expand(B, T_, N, 3), lt], -1))   # (B,T,N,p)
        out = torch.einsum("bip,btnp->btin", br, tr) + self.bias
        return mu[:, None] + sd[:, None] * out                                     # back to physical units


def matched_alpha(p, target=TARGET_PARAMS):
    lo, hi = 0.1, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if n_params(DeepONetNorm(p, mid)) < target else (lo, mid)
    a = min((lo, hi), key=lambda a: abs(n_params(DeepONetNorm(p, a)) - target))
    return a


RUNS = [dict(name=f"DON-matched p={p} lr={lr:g}", p=p, lr=lr, matched=True)
        for p in (64, 128, 256) for lr in (3e-4, 1e-3, 3e-3)]
RUNS.append(dict(name="DON-base (정규화) p=128 lr=0.001", p=128, lr=1e-3, matched=False))


def train_to_plateau(model, D, ids, lr, log_rows, name):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4, min_lr=1e-7)
    best_val, best, stale = np.inf, copy.deepcopy(model.state_dict()), 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(1, MAX_STEPS + 1):
        if step <= WARMUP:                                          # linear warm-up
            for g in opt.param_groups:
                g["lr"] = lr * step / WARMUP
        model.train()
        b, U, m = T.batch(D, rng.choice(ids["train"], T.BATCH, replace=False), "ex")
        loss = T.rel_err(model(b), U, m).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0:
            val = T.evaluate(model, D, ids["val"], "ex").mean()
            if step > WARMUP:
                plateau.step(val)
            log_rows.append(dict(model=name, step=step, train_loss=loss.item(), val_err=val,
                                 lr=opt.param_groups[0]["lr"]))
            if val < best_val * (1 - 5e-3):
                stale = 0
            else:
                stale += 1
            if val < best_val:
                best_val, best = val, copy.deepcopy(model.state_dict())
            if step % 10000 == 0:
                print(f"  [{name}] step {step} val {val:.4f} best {best_val:.4f} "
                      f"lr {opt.param_groups[0]['lr']:.1e} stale {stale}", flush=True)
            if stale >= PATIENCE:
                break
    torch.cuda.synchronize()
    model.load_state_dict(best)
    return time.perf_counter() - t0, float(best_val), step


def run(run_ids):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "set CUDA_VISIBLE_DEVICES"
    os.makedirs(OUT, exist_ok=True)
    D, ids, index = T.load_all()
    family = index["family"]
    for i in run_ids:
        cfg = RUNS[i]
        alpha = matched_alpha(cfg["p"]) if cfg["matched"] else 1.0
        model = DeepONetNorm(cfg["p"], alpha).to(T.DEV)
        print(f"run {i} {cfg['name']}: alpha {alpha:.3f}, {n_params(model)} params", flush=True)
        log_rows = []
        t_train, best_val, steps = train_to_plateau(model, D, ids, cfg["lr"], log_rows, cfg["name"])
        e = T.evaluate(model, D, ids["test"], "ex")
        rows = [dict(model=cfg["name"], sid=int(sid), family=family[sid], ic=ic, tau_idx=ti, tau=tau,
                     err=float(e[si, ti, ic]))
                for si, sid in enumerate(ids["test"]) for ti, tau in enumerate(TAUS) for ic in range(e.shape[2])]
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"errors_test_run{i}.csv"), index=False)
        pd.DataFrame(log_rows).to_csv(os.path.join(OUT, f"train_log_run{i}.csv"), index=False)
        meta = dict(**cfg, alpha=alpha, params=n_params(model), train_s=t_train, steps=steps,
                    best_val=best_val, test_mean=float(e.mean()),
                    infer_ms=T.time_inference(model, D, ids["test"], "ex"))
        json.dump(meta, open(os.path.join(OUT, f"meta_run{i}.json"), "w"), indent=2)
        torch.save(model.state_dict(), os.path.join(OUT, f"model_run{i}.pt"))
        print(f"  run {i} done: best val {best_val:.4f}, test {e.mean():.4f}, {steps} steps, {t_train:.0f}s",
              flush=True)


def merge():
    metas = [json.load(open(os.path.join(OUT, f"meta_run{i}.json"))) for i in range(len(RUNS))
             if os.path.exists(os.path.join(OUT, f"meta_run{i}.json"))]
    M = pd.DataFrame(metas)
    M.to_csv(os.path.join(OUT, "search.csv"), index=False)
    print(M[["name", "p", "lr", "alpha", "params", "steps", "train_s", "best_val", "test_mean"]].to_string(index=False))
    E = pd.concat([pd.read_csv(os.path.join(OUT, f"errors_test_run{i}.csv")) for i in range(len(RUNS))
                   if os.path.exists(os.path.join(OUT, f"errors_test_run{i}.csv"))])
    E.to_csv(os.path.join(OUT, "errors_test.csv"), index=False)
    if "status" not in M:
        M["status"] = "plateau"
    M["status"] = M["status"].fillna("plateau")
    matched = M[M.matched & (M.status == "plateau")]
    if len(matched):
        best = matched.loc[matched.best_val.idxmin()]
        print("best DON-base-matched by val:", best["name"])
        json.dump(best.to_dict(), open(os.path.join(OUT, "best_matched.json"), "w"), indent=2, default=float)


# ------------------------------------------------------------------------------------------------
# queue worker (added after the resource check): claim-based queue, checkpoint every 10k steps,
# early stopping at 30k steps vs runs with the same p (rule fixed in advance, see RESOURCE_CHECK.md)
# ------------------------------------------------------------------------------------------------
QUEUE_ORDER = [4, 5, 7, 8, 9, 3, 6]          # promising learning rates first, so references exist early
STATE = os.path.join(OUT, "state")
CKPT_EVERY, ES_STEP, ES_FACTOR = 10_000, 30_000, 1.2


def _val30k(i):
    f = os.path.join(STATE, f"val30k_run{i}.txt")
    return float(open(f).read()) if os.path.exists(f) else None


def _es_reference(p):
    vals = [_val30k(j) for j, r in enumerate(RUNS) if r["p"] == p and r["matched"]]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def claim_next():
    os.makedirs(STATE, exist_ok=True)
    for i in QUEUE_ORDER:
        try:
            fd = os.open(os.path.join(STATE, f"claim_run{i}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode()); os.close(fd)
            return i
        except FileExistsError:
            continue
    return None


def train_resumable(model, D, ids, i, lr, name):
    ck = os.path.join(STATE, f"ckpt_run{i}.pt")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4, min_lr=1e-7)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    best_val, best, stale, start, log_rows, t_prev = np.inf, copy.deepcopy(model.state_dict()), 0, 1, [], 0.0
    if os.path.exists(ck):                                         # resume
        c = torch.load(ck, map_location=T.DEV, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); plateau.load_state_dict(c["plateau"])
        rng.bit_generator.state = c["rng"]; torch.set_rng_state(c["torch_rng"])
        best_val, best, stale, start, log_rows, t_prev = (c["best_val"], c["best"], c["stale"], c["step"] + 1,
                                                          c["log"], c["train_s"])
        print(f"  resumed run {i} at step {start - 1}", flush=True)
    status = "plateau"
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = start - 1
    for step in range(start, MAX_STEPS + 1):
        if step <= WARMUP:
            for g in opt.param_groups:
                g["lr"] = lr * step / WARMUP
        model.train()
        b, U, m = T.batch(D, rng.choice(ids["train"], T.BATCH, replace=False), "ex")
        loss = T.rel_err(model(b), U, m).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0:
            val = T.evaluate(model, D, ids["val"], "ex").mean()
            if step > WARMUP:
                plateau.step(val)
            log_rows.append(dict(model=name, step=step, train_loss=loss.item(), val_err=val,
                                 lr=opt.param_groups[0]["lr"]))
            stale = 0 if val < best_val * (1 - 5e-3) else stale + 1
            if val < best_val:
                best_val, best = val, copy.deepcopy(model.state_dict())
            if step == ES_STEP:
                open(os.path.join(STATE, f"val30k_run{i}.txt"), "w").write(f"{best_val:.8f}")
            if step >= ES_STEP:
                ref = _es_reference(RUNS[i]["p"])
                mine = _val30k(i)
                if ref is not None and mine is not None and mine > ES_FACTOR * ref:
                    status = f"early-stopped (val@30k {mine:.4f} > {ES_FACTOR}x ref {ref:.4f})"
                    break
            if step % 10000 == 0:
                print(f"  [{name}] step {step} val {val:.4f} best {best_val:.4f} "
                      f"lr {opt.param_groups[0]['lr']:.1e} stale {stale}", flush=True)
            if stale >= PATIENCE:
                break
        if step % CKPT_EVERY == 0:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), plateau=plateau.state_dict(),
                            rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), best_val=best_val,
                            best=best, stale=stale, step=step, log=log_rows,
                            train_s=t_prev + time.perf_counter() - t0), ck)
    torch.cuda.synchronize()
    model.load_state_dict(best)
    return t_prev + time.perf_counter() - t0, float(best_val), step, status, log_rows


def worker():
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "set CUDA_VISIBLE_DEVICES"
    D, ids, index = T.load_all()
    family = index["family"]
    while (i := claim_next()) is not None:
        cfg = RUNS[i]
        alpha = matched_alpha(cfg["p"]) if cfg["matched"] else 1.0
        model = DeepONetNorm(cfg["p"], alpha).to(T.DEV)
        print(f"run {i} {cfg['name']}: alpha {alpha:.3f}, {n_params(model)} params", flush=True)
        t_train, best_val, steps, status, log_rows = train_resumable(model, D, ids, i, cfg["lr"], cfg["name"])
        pd.DataFrame(log_rows).to_csv(os.path.join(OUT, f"train_log_run{i}.csv"), index=False)
        meta = dict(**cfg, alpha=alpha, params=n_params(model), train_s=t_train, steps=steps,
                    best_val=best_val, status=status)
        if status == "plateau":
            e = T.evaluate(model, D, ids["test"], "ex")
            rows = [dict(model=cfg["name"], sid=int(sid), family=family[sid], ic=ic, tau_idx=ti, tau=tau,
                         err=float(e[si, ti, ic])) for si, sid in enumerate(ids["test"])
                    for ti, tau in enumerate(TAUS) for ic in range(e.shape[2])]
            pd.DataFrame(rows).to_csv(os.path.join(OUT, f"errors_test_run{i}.csv"), index=False)
            meta.update(test_mean=float(e.mean()), infer_ms=T.time_inference(model, D, ids["test"], "ex"))
            torch.save(model.state_dict(), os.path.join(OUT, f"model_run{i}.pt"))
        json.dump(meta, open(os.path.join(OUT, f"meta_run{i}.json"), "w"), indent=2)
        print(f"  run {i} {status}: best val {best_val:.4f}, {steps} steps, {t_train:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        worker()
        sys.exit()
    if sys.argv[1:] == ["--merge"]:
        merge()
    else:
        assert len(sys.argv) == 3 and sys.argv[1] == "--runs"
        run([int(v) for v in sys.argv[2].split(",")])

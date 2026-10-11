"""ε-path v2: residual inputs + physics prior r_p (all on the NEO basis) -> results_eps/

    CUDA_VISIBLE_DEVICES=<gpu> python stage1/train_eps.py --shard c     # baselines + (c)
    CUDA_VISIBLE_DEVICES=<gpu> python stage1/train_eps.py --shard d     # (d), (d')
    CUDA_VISIBLE_DEVICES=<gpu> python stage1/train_eps.py --shard don   # DON-base until val plateau
    python stage1/train_eps.py --merge

Models (NEO basis Y = the 96 Ritz vectors of L-neo; r = u0 − Y Yᵀ M u0; r_p = (M + tL)^{-1} M r):
  (a)  L-neo              existing weights (stage 1), re-evaluated
  (b)  L-neo+rp           (a) + r_p(t); no training
  (c)  L-neo+rp+eps       (b) + ε_θ(inputs r, r_p);  g_θ (init from L-neo) and ε_θ trained jointly
  (d)  L-neo+eps(r)       (a) + ε_θ(input r only; r_p neither added nor given)
  (d') L-neo+eps(r,rp-in) (a) + ε_θ(inputs r, r_p; r_p not added)  -- supplementary
  PC-discrete             expm_multiply(−t M^{-1} cL, u0) on the full point-cloud operator (reference)
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

import train as T
from common import DATA, N_SHAPES, RESULTS, SEED, TAUS
from models import DeepONetBase, Galerkin, LNeoResidual, n_params

OUT = os.path.join(os.path.dirname(RESULTS), "results_eps")
SHARDS = ["c", "d", "don"]
T.STEPS["eps2"], T.LR["eps2"] = 15000, 1e-3
DON_MAX_STEPS, DON_EVAL, DON_PATIENCE = 400_000, 1000, 20          # stop after 20 evals w/o >0.5 % gain


def load():
    D, ids, index = T.load_all()
    r, rp = [], []
    for sid in range(N_SHAPES):
        e = np.load(os.path.join(DATA, f"eps_{sid:03d}.npz"))
        r.append(e["r"]); rp.append(e["rp"])
    D["r"] = torch.tensor(np.stack(r), device=T.DEV)
    D["rp"] = torch.tensor(np.stack(rp), device=T.DEV)
    tr = torch.as_tensor(ids["train"], device=T.DEV)
    r_rms = torch.sqrt((D["m_pc"][tr, None] * D["r"][tr] ** 2).sum() / (D["m_pc"][tr].sum() * D["r"].shape[1]))
    return D, ids, index, float(1 / r_rms)


_base_batch = T.batch


def batch_eps(D, ids, basis):
    b, U, m = _base_batch(D, ids, basis)
    it = torch.as_tensor(ids, device=T.DEV)
    b["r"], b["rp"] = D["r"][it], D["rp"][it]
    return b, U, m


T.batch = batch_eps                  # train.train / evaluate / time_inference now carry r, r_p


def l_neo(add_rp=False, eps=None, r_scale=1.0):
    m = LNeoResidual(add_rp=add_rp, eps=eps, r_scale=r_scale)
    base = torch.load(os.path.join(RESULTS, "model_L-neo.pt"), map_location="cpu")
    m.g.load_state_dict({k[2:]: v for k, v in base.items() if k.startswith("g.")})
    return m.to(T.DEV)


def long_rows(name, e, ids, family):
    return [dict(model=name, sid=int(sid), family=family[sid], ic=ic, tau_idx=ti, tau=tau,
                 err=float(e[si, ti, ic]))
            for si, sid in enumerate(ids) for ti, tau in enumerate(TAUS) for ic in range(e.shape[2])]


def run(shard):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "set CUDA_VISIBLE_DEVICES"
    os.makedirs(OUT, exist_ok=True)
    D, ids, index, r_scale = load()
    family = index["family"]
    rows, meta, log_rows = [], {}, []

    if shard == "c":
        # untrained references
        for name, model, basis in [("Oracle-mesh96", Galerkin().to(T.DEV), "mesh"),
                                   ("G-neo", Galerkin().to(T.DEV), "neo"),
                                   ("(a) L-neo", l_neo(), "neo"),
                                   ("(b) L-neo+rp", l_neo(add_rp=True), "neo")]:
            rows += long_rows(name, T.evaluate(model, D, ids["test"], basis), ids["test"], family)
            meta[name] = dict(params=n_params(model), train_s=0.0,
                              infer_ms=T.time_inference(model, D, ids["test"], basis))
        # full point-cloud operator, exact in time
        e = []
        for sid in ids["test"]:
            u = torch.tensor(np.load(os.path.join(DATA, f"pcsol_{sid:03d}.npz"))["u_pc"], device=T.DEV)
            it = torch.as_tensor([sid], device=T.DEV)
            e.append(T.rel_err(u[None], D["U"][it], D["m_mesh"][it]).cpu().numpy()[0])
        rows += long_rows("PC-discrete", np.stack(e), ids["test"], family)
        meta["PC-discrete"] = dict(params=0, train_s=0.0, infer_ms=float("nan"))
        todo = [("(c) L-neo+rp+eps", dict(add_rp=True, eps="rp"))]
    elif shard == "d":
        todo = [("(d) L-neo+eps(r)", dict(add_rp=False, eps="norp")),
                ("(d') L-neo+eps(r,rp-in)", dict(add_rp=False, eps="rp"))]
    else:
        todo = []

    for name, kw in todo:
        model = l_neo(r_scale=r_scale, **kw)
        print(f"training {name}: {n_params(model)} params", flush=True)
        t_train, best_val = T.train(model, D, ids, "neo", "eps2", log_rows, name)
        rows += long_rows(name, T.evaluate(model, D, ids["test"], "neo"), ids["test"], family)
        meta[name] = dict(params=n_params(model), train_s=t_train, best_val=float(best_val),
                          infer_ms=T.time_inference(model, D, ids["test"], "neo"))
        torch.save(model.state_dict(), os.path.join(OUT, f"model_{name.split()[0]}.pt"))
        print(f"  {name}: test mean {np.mean([r['err'] for r in rows if r['model'] == name]):.4f}", flush=True)

    if shard == "don":
        name = "DON-base (converged)"
        model = DeepONetBase().to(T.DEV)
        model.load_state_dict(torch.load(os.path.join(RESULTS, "model_DON-base.pt"), map_location=T.DEV))
        t_train = continue_don(model, D, ids, log_rows)
        rows += long_rows(name, T.evaluate(model, D, ids["test"], "ex"), ids["test"], family)
        prev = json.load(open(os.path.join(RESULTS, "models.json")))["DON-base"]["train_s"]
        meta[name] = dict(params=n_params(model), train_s=prev + t_train,
                          steps=int(20000 + log_rows[-1]["step"]),
                          infer_ms=T.time_inference(model, D, ids["test"], "ex"))
        torch.save(model.state_dict(), os.path.join(OUT, "model_DON-base-converged.pt"))

    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"errors_test_{shard}.csv"), index=False)
    pd.DataFrame(log_rows).to_csv(os.path.join(OUT, f"train_log_{shard}.csv"), index=False)
    json.dump(meta, open(os.path.join(OUT, f"models_{shard}.json"), "w"), indent=2)


def continue_don(model, D, ids, log_rows):
    """Resume DON-base from its stage-1 best checkpoint; AdamW 3e-4 + ReduceLROnPlateau;
    stop when the val error has not improved by >0.5 % (relative) for DON_PATIENCE evals."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4, min_lr=1e-6)
    best_val = T.evaluate(model, D, ids["val"], "ex").mean()
    best, stale = copy.deepcopy(model.state_dict()), 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(1, DON_MAX_STEPS + 1):
        model.train()
        b, U, m = T.batch(D, rng.choice(ids["train"], T.BATCH, replace=False), "ex")
        loss = T.rel_err(model(b), U, m).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % DON_EVAL == 0:
            val = T.evaluate(model, D, ids["val"], "ex").mean()
            sched.step(val)
            log_rows.append(dict(model="DON-base (converged)", step=step, train_loss=loss.item(), val_err=val,
                                 lr=opt.param_groups[0]["lr"]))
            if val < best_val * (1 - 5e-3):
                best_val, best, stale = val, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if val < best_val:
                    best_val, best = val, copy.deepcopy(model.state_dict())
            print(f"  [DON] step {step} val {val:.4f} best {best_val:.4f} lr {opt.param_groups[0]['lr']:.1e} "
                  f"stale {stale}", flush=True)
            if stale >= DON_PATIENCE:
                break
    torch.cuda.synchronize()
    model.load_state_dict(best)
    return time.perf_counter() - t0


def merge():
    shards = [s for s in SHARDS if os.path.exists(os.path.join(OUT, f"errors_test_{s}.csv"))]
    pd.concat([pd.read_csv(os.path.join(OUT, f"errors_test_{s}.csv")) for s in shards]).to_csv(
        os.path.join(OUT, "errors_test.csv"), index=False)
    pd.concat([pd.read_csv(os.path.join(OUT, f"train_log_{s}.csv")) for s in shards]).to_csv(
        os.path.join(OUT, "train_log.csv"), index=False)
    meta = {}
    for s in shards:
        meta.update(json.load(open(os.path.join(OUT, f"models_{s}.json"))))
    json.dump(meta, open(os.path.join(OUT, "models.json"), "w"), indent=2)
    print("merged shards:", shards)


if __name__ == "__main__":
    if sys.argv[1:] == ["--merge"]:
        merge()
    else:
        assert len(sys.argv) == 3 and sys.argv[1] == "--shard" and sys.argv[2] in SHARDS
        run(sys.argv[2])

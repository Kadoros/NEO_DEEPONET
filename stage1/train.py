"""Part 2/3: train + evaluate all models on the stage1 dataset.

    CUDA_VISIBLE_DEVICES=0 python stage1/train.py --shard exact   (one free GPU per shard,
    CUDA_VISIBLE_DEVICES=1 python stage1/train.py --shard neo      shards run in parallel)
    CUDA_VISIBLE_DEVICES=2 python stage1/train.py --shard don
    python stage1/train.py --merge

Metric (also the training loss): M-weighted relative L2 error with the mesh lumped
mass,  ||û − u||_M / ||u||_M,  per (shape, IC, τ).
"""
import common  # noqa: F401

import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from common import DATA, FAMILIES, K_MODES, N_SHAPES, RESULTS, SEED, TAUS
from models import DeepONetBase, Galerkin, LearnedSpectral, n_params

DEV = "cuda"
BATCH = 8
STEPS = {"L": 3000, "L+eps": 10000, "DON": 20000}
LR = {"L": 3e-3, "L+eps": 1e-3, "DON": 1e-3}
VAL_EVERY = 250


# ------------------------------------------------------------------- data --
def canon_sign(Phi):
    """Deterministic sign per column: largest-|entry| positive."""
    idx = np.abs(Phi).argmax(0)
    return Phi * np.sign(Phi[idx, np.arange(Phi.shape[1])])


def load_all():
    split = pd.read_csv(os.path.join(DATA, "split.csv")).set_index("sid")["split"]
    index = pd.read_csv(os.path.join(DATA, "index.csv")).set_index("sid")
    keys = ["x", "m_mesh", "m_pc", "u0", "U", "t", "Phi_ex", "lam_ex", "Phi_neo", "lam_neo",
            "Phi_mesh", "lam_mesh", "F", "z", "c_pc"]
    buf = {k: [] for k in keys}
    for sid in range(N_SHAPES):
        d = np.load(os.path.join(DATA, f"shape_{sid:03d}.npz"))
        e = np.load(os.path.join(DATA, f"neo_{sid:03d}.npz"))
        buf["x"].append(d["V"]); buf["m_mesh"].append(d["m_mesh"]); buf["m_pc"].append(d["M_pc"])
        buf["u0"].append(d["U0"]); buf["U"].append(d["U"]); buf["t"].append(d["t"])
        buf["Phi_ex"].append(canon_sign(d["Phi_ex"])); buf["lam_ex"].append(d["lam_ex"])
        buf["Phi_neo"].append(canon_sign(e["Phi_neo"])); buf["lam_neo"].append(e["lam_neo"])
        buf["Phi_mesh"].append(canon_sign(d["Phi_mesh"])); buf["lam_mesh"].append(d["lam_mesh"])
        buf["F"].append(e["F_hat"]); buf["z"].append(e["z"]); buf["c_pc"].append(d["c_pc"])
    D = {}
    for k, v in buf.items():
        a = np.stack(v)
        D[k] = torch.tensor(a, dtype=torch.float16 if k == "F" else torch.float32, device=DEV)
    # basis coefficients c = Φᵀ M u0  (M = point-cloud mass; mesh basis uses mesh mass)
    D["coef_ex"] = torch.einsum("snk,sn,sin->ski", D["Phi_ex"], D["m_pc"], D["u0"])
    D["coef_neo"] = torch.einsum("snk,sn,sin->ski", D["Phi_neo"], D["m_pc"], D["u0"])
    D["coef_mesh"] = torch.einsum("snk,sn,sin->ski", D["Phi_mesh"], D["m_mesh"], D["u0"])
    # uncalibrated operator L (not c·L): same eigenvectors, eigenvalues λ/c
    D["Phi_exuncal"], D["coef_exuncal"] = D["Phi_ex"], D["coef_ex"]
    D["lam_exuncal"] = D["lam_ex"] / D["c_pc"][:, None]
    ids = {s: np.array([i for i in range(N_SHAPES) if split[i] == s]) for s in ("train", "val", "test")}
    return D, ids, index


def batch(D, ids, basis):
    ids = torch.as_tensor(ids, device=DEV)
    b = {k: D[k][ids] for k in ("x", "m_pc", "u0", "t", "z", "coef_neo")}
    b["F"] = D["F"][ids].float()
    b["Phi"], b["lam"], b["coef"] = D[f"Phi_{basis}"][ids], D[f"lam_{basis}"][ids], D[f"coef_{basis}"][ids]
    return b, D["U"][ids], D["m_mesh"][ids]


def rel_err(pred, U, m):
    w = m[:, None, None, :]
    return torch.sqrt((w * (pred - U) ** 2).sum(-1) / (w * U ** 2).sum(-1))             # (B,T,I)


@torch.no_grad()
def evaluate(model, D, ids, basis, chunk=8):
    model.eval()
    out = []
    for i in range(0, len(ids), chunk):
        b, U, m = batch(D, ids[i:i + chunk], basis)
        out.append(rel_err(model(b), U, m).cpu().numpy())
    return np.concatenate(out)                                                            # (S,T,I)


# ---------------------------------------------------------------- training --
def train(model, D, ids, basis, kind, log_rows, name):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR[kind], weight_decay=1e-5)
    steps = STEPS[kind]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR[kind], total_steps=steps, pct_start=0.05)
    best, best_val = copy.deepcopy(model.state_dict()), evaluate(model, D, ids["val"], basis).mean()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        b, U, m = batch(D, rng.choice(ids["train"], BATCH, replace=False), basis)
        loss = rel_err(model(b), U, m).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % VAL_EVERY == 0 or step == steps:
            val = evaluate(model, D, ids["val"], basis).mean()
            log_rows.append(dict(model=name, step=step, train_loss=loss.item(), val_err=val))
            if val < best_val:
                best_val, best = val, copy.deepcopy(model.state_dict())
            print(f"  [{name}] step {step:5d} train {loss.item():.4f} val {val:.4f}", flush=True)
    torch.cuda.synchronize()
    model.load_state_dict(best)
    return time.perf_counter() - t0, best_val


@torch.no_grad()
def time_inference(model, D, ids, basis, reps=3):
    model.eval()
    for sid in ids[:3]:                                                                   # warm-up
        model(batch(D, [sid], basis)[0])
    ts = []
    for sid in ids:
        b = batch(D, [sid], basis)[0]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            model(b)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / reps)
    return 1e3 * float(np.mean(ts))


# ------------------------------------------------------------------- main --
# Shards run in parallel, one GPU each; a "+eps" model is in the same shard as its base L model.
SHARDS = {
    "exact": ["L-exact", "L-exact+eps", "L-exact-uncal"],
    "neo": ["L-neo", "L-neo+eps"],
    "don": ["DON-base"],
}
BASELINES = [("Oracle-mesh96", "mesh"), ("G-exact", "ex"), ("G-exact-uncal", "exuncal"), ("G-neo", "neo")]
SPEC = {"L-exact": ("ex", "L"), "L-exact+eps": ("ex", "L+eps"), "L-exact-uncal": ("exuncal", "L"),
        "L-neo": ("neo", "L"), "L-neo+eps": ("neo", "L+eps"), "DON-base": ("ex", "DON")}   # basis, kind


def run_shard(shard):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "set CUDA_VISIBLE_DEVICES to one free GPU"
    out = os.path.join(RESULTS, "shards")
    os.makedirs(out, exist_ok=True)
    D, ids, index = load_all()
    family = index["family"]
    log_rows, meta, errs, models = [], {}, {}, {}

    if shard == "don":                                          # untrained baselines ride along here
        for name, basis in BASELINES:
            model = Galerkin().to(DEV)
            errs[name] = evaluate(model, D, ids["test"], basis)
            meta[name] = dict(params=0, train_s=0.0, infer_ms=time_inference(model, D, ids["test"], basis))

    for name in SHARDS[shard]:
        basis, kind = SPEC[name]
        if kind == "L+eps":                                     # start from the trained L model's g_θ
            model = LearnedSpectral(use_eps=True)
            model.g.load_state_dict(models[name.replace("+eps", "")].g.state_dict())
        elif kind == "L":
            model = LearnedSpectral(use_eps=False)
        else:
            model = DeepONetBase()
        model = model.to(DEV)
        print(f"training {name}: {n_params(model)} params", flush=True)
        t_train, best_val = train(model, D, ids, basis, kind, log_rows, name)
        errs[name] = evaluate(model, D, ids["test"], basis)
        meta[name] = dict(params=n_params(model), train_s=t_train, best_val=float(best_val),
                          infer_ms=time_inference(model, D, ids["test"], basis))
        models[name] = model
        torch.save(model.state_dict(), os.path.join(RESULTS, f"model_{name}.pt"))
        print(f"  {name}: test mean {errs[name].mean():.4f}  train {t_train:.0f}s", flush=True)

    rows = []
    for name, e in errs.items():
        for si, sid in enumerate(ids["test"]):
            for ti, tau in enumerate(TAUS):
                for ic in range(e.shape[2]):
                    rows.append(dict(model=name, sid=int(sid), family=family[sid], ic=ic,
                                     tau_idx=ti, tau=tau, err=float(e[si, ti, ic])))
    pd.DataFrame(rows).to_csv(os.path.join(out, f"errors_test_{shard}.csv"), index=False)
    pd.DataFrame(log_rows).to_csv(os.path.join(out, f"train_log_{shard}.csv"), index=False)
    with open(os.path.join(out, f"models_{shard}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # g_θ on one test shape per family, for the learned-spectral models of this shard
    probe = [int(next(s for s in ids["test"] if family[s] == f)) for f in FAMILIES]
    pt = torch.as_tensor(probe, device=DEV)
    dump = {"probe_sids": np.array(probe), "taus": np.array(TAUS),
            "lam_mesh": D["lam_mesh"][pt].cpu().numpy(), "c_pc": D["c_pc"][pt].cpu().numpy()}
    for name, model in models.items():
        if not isinstance(model, LearnedSpectral):
            continue
        model.eval()
        with torch.no_grad():
            b = batch(D, probe, SPEC[name][0])[0]
            dump[f"{name}|g"] = model.g(b["lam"], b["t"], b["z"]).cpu().numpy()             # (3,T,K)
            dump[f"{name}|lam"] = b["lam"].cpu().numpy()
            dump[f"{name}|t"] = b["t"].cpu().numpy()
    np.savez(os.path.join(out, f"g_theta_probe_{shard}.npz"), **dump)


def merge():
    out = os.path.join(RESULTS, "shards")
    pd.concat([pd.read_csv(os.path.join(out, f"errors_test_{s}.csv")) for s in SHARDS]).to_csv(
        os.path.join(RESULTS, "errors_test.csv"), index=False)
    pd.concat([pd.read_csv(os.path.join(out, f"train_log_{s}.csv")) for s in SHARDS]).to_csv(
        os.path.join(RESULTS, "train_log.csv"), index=False)
    meta = {}
    for s in SHARDS:
        meta.update(json.load(open(os.path.join(out, f"models_{s}.json"))))
    json.dump(meta, open(os.path.join(RESULTS, "models.json"), "w"), indent=2)
    dump = {}
    for s in SHARDS:
        dump.update(dict(np.load(os.path.join(out, f"g_theta_probe_{s}.npz"))))
    np.savez(os.path.join(RESULTS, "g_theta_probe.npz"), **dump)
    print("merged:", sorted(meta))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--merge"]:
        merge()
    else:
        assert len(sys.argv) == 3 and sys.argv[1] == "--shard" and sys.argv[2] in SHARDS, \
            f"usage: train.py --shard {{{','.join(SHARDS)}}} | --merge"
        run_shard(sys.argv[2])

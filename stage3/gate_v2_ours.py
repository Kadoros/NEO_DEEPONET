"""Gate v2 (task B), our structure -> stage3/gate/runs_v2/ours_N<N>_s<seed>/{result.json, test.csv, kappa_hat.npz}

u_base = Phi exp(-t A_hat) Phi^T M u0,  A_hat = Phi^T K(kappa_hat) Phi = sum_e kappa_e vol_e (grad Phi_e)^T (grad Phi_e)
         (element-tensor assembly in fp32 on the GPU, exponential in fp64); Phi = raw volume-NEO basis (256).
kappa_hat(x) = exp(MLP[d(x), W F_hat(x)]) per node, kappa_e = mean over the tet's nodes; last layer zero ->
         kappa_hat = 1 and u_base = G (the kappa = 1 Galerkin baseline) at initialisation.
u = u_base + eps,  eps = stage-1 ResidualEps(r, r_p, F, x, t) with its zero-initialised output layer.
Loss (staged, stop-gradient): L = l(u_base) + l(sg(u_base) + eps),  l = mean relative L2 (lumped mass).
Selection: every 250 steps on the val split; only checkpoints whose val error is <= that of G (kappa = 1) are
eligible; if none, the initial model (= G) is kept.
Reports both outputs of the selected model: "ours" (u_base + eps) and "ours-kappa" (u_base only).

    CUDA_VISIBLE_DEVICES=<gpu> python stage3/gate_v2_ours.py --N 50 --seed 0
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
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "stage1"))
from models import ResidualEps, mlp, n_params  # noqa: E402

DATA = os.path.join(ROOT, "gate", "data_v2")
FEAT = os.path.join(ROOT, "gate", "data_feat")                        # HKS node features (gate_feat.py)
FEAT_KIND = "neo"                                                     # kappa-net input: neo | hks_d | hks
RUNS = os.path.join(ROOT, "gate", "runs_v2")
DEV = "cuda"
STEPS, LR, BATCH, VAL_EVERY = 8000, 1e-3, 4, 250
KEYS = ("x", "m", "d", "F", "z", "Phi", "coef", "u0", "U", "r", "rp", "t", "grads", "vol", "U_g_raw")


def load_split(split, names=None):
    out = []
    for p in sorted(glob.glob(os.path.join(DATA, split, "*.npz"))):
        name = os.path.basename(p)[:-4]
        if names is not None and name not in names:
            continue
        z = np.load(p)
        it = {k: torch.tensor(z[k].astype(np.float32)).to(DEV) for k in KEYS}
        it["tets"] = torch.tensor(z["tets"].astype(np.int64)).to(DEV)
        fp = os.path.join(FEAT, split, f"{name}.npz")
        if FEAT_KIND != "neo":
            it["hks"] = torch.tensor(np.load(fp)["hks"]).to(DEV)
        it["kappa"] = z["kappa"]
        out.append((name, it))
    return out


class KappaNet(nn.Module):
    """kind = neo: [depth, W F] (v2);  hks_d: [depth, HKS(6)];  hks: [HKS(6)] (no depth)."""

    def __init__(self, kind="neo", f_dim=256, f_proj=32, hidden=128):
        super().__init__()
        self.kind = kind
        self.proj = nn.Linear(f_dim, f_proj) if kind == "neo" else None
        n_in = {"neo": 1 + f_proj, "hks_d": 1 + 6, "hks": 6}[kind]
        self.net = mlp([n_in, hidden, hidden, 1], last_zero=True)

    def forward(self, it):
        d = it["d"][:, None] * 5.0
        x = {"neo": lambda: torch.cat([d, self.proj(it["F"])], -1),
             "hks_d": lambda: torch.cat([d, it["hks"] / 5.0], -1),
             "hks": lambda: it["hks"] / 5.0}[self.kind]()
        return torch.exp(self.net(x)[:, 0].clamp(-3, 3))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.kappa = KappaNet(FEAT_KIND)
        self.eps = ResidualEps(use_rp=True)

    def base(self, it):
        kap = self.kappa(it)
        kap_e = kap[it["tets"]].mean(1)
        GY = torch.einsum("tki,tim->tkm", it["grads"], it["Phi"][it["tets"]]).reshape(-1, it["Phi"].shape[1])
        w = (kap_e * it["vol"]).repeat_interleave(3)
        A = (GY * w[:, None]).T @ GY
        A = 0.5 * (A + A.T).double()
        E_ = torch.linalg.matrix_exp(-it["t"].double()[:, None, None] * A[None])        # (T, K, K)
        c = E_ @ it["coef"].double()[None]                                                # (T, K, I)
        return torch.einsum("nk,tki->tin", it["Phi"].double(), c).float(), kap           # (T, I, N)

    def eps_out(self, it):
        b = dict(x=it["x"][None], u0=it["u0"][None], m_pc=it["m"][None], r=it["r"][None], rp=it["rp"][None],
                 F=it["F"][None], t=it["t"][None])
        return self.eps(b)[0]                                                              # (T, I, N)


def rel(pred, U, m):
    return torch.sqrt((m * (pred - U) ** 2).sum(-1) / (m * U ** 2).sum(-1))              # (T, I)


@torch.no_grad()
def evaluate(model, items, keep_kappa=False):
    model.eval()
    out = []
    for name, it in items:
        ub, kap = model.base(it)
        u = ub + model.eps_out(it)
        out.append((name, rel(u, it["U"], it["m"]).cpu().numpy(), rel(ub, it["U"], it["m"]).cpu().numpy(),
                    kap.cpu().numpy() if keep_kappa else None))
    return out


def val_err(model, val):
    return float(np.mean([e.mean() for _, e, _, _ in evaluate(model, val)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--feat", default="neo", choices=["neo", "hks_d", "hks"])
    a = ap.parse_args()
    global FEAT_KIND, DATA
    FEAT_KIND = a.feat
    torch.manual_seed(a.seed)
    out = os.path.join(RUNS, f"ours_N{a.N}_s{a.seed}" + ("" if a.feat == "neo" else f"_{a.feat}"))
    os.makedirs(out, exist_ok=True)
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(DATA, "train", "*.npz")))
    pick = sorted(np.random.default_rng([a.seed, a.N]).choice(len(names), a.N // 2, replace=False))   # as gate v1
    train, val = load_split("train", {names[i] for i in pick}), load_split("val")
    g_val = float(np.mean([rel(it["U_g_raw"], it["U"], it["m"]).mean().item() for _, it in val]))
    model = Model().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.05)
    rng = np.random.default_rng(a.seed)
    best, best_val, log = copy.deepcopy(model.state_dict()), val_err(model, val), []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        model.train()
        loss = 0.0
        for i in rng.choice(len(train), min(BATCH, len(train)), replace=False):
            it = train[i][1]
            ub, _ = model.base(it)
            e = model.eps_out(it)
            loss = loss + rel(ub, it["U"], it["m"]).mean() + rel(ub.detach() + e, it["U"], it["m"]).mean()
        loss = loss / BATCH
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % VAL_EVERY == 0 or step == STEPS:
            v = val_err(model, val)
            log.append(dict(step=step, loss=float(loss.item()), val=v, g_val=g_val))
            if v <= g_val and v < best_val:
                best_val, best = v, copy.deepcopy(model.state_dict())
            print(f"step {step} loss {loss.item():.4f} val {v:.4f} (G {g_val:.4f}) best {best_val:.4f}", flush=True)
    t_train = time.time() - t0
    model.load_state_dict(best)
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    rows, kap_store = [], {}
    for split in ("test_in", "test_ood", "test_real"):
        items = load_split(split)
        torch.cuda.synchronize(); ti = time.time()
        res = evaluate(model, items, keep_kappa=True)
        torch.cuda.synchronize(); t_inf = (time.time() - ti) / max(len(items), 1)
        for (name, e_full, e_base, kap), (_, it) in zip(res, items):
            kap_store[f"{split}/{name}"] = np.stack([it["d"].cpu().numpy(), kap, it["kappa"]])
            for model_name, e in (("ours", e_full), ("ours-kappa", e_base)):
                for j in range(e.shape[0]):
                    for i in range(e.shape[1]):
                        rows.append(dict(model=model_name, split=split, shape=name, tau_idx=j, ic=i, rel=float(e[j, i]),
                                         t_infer_per_shape_s=t_inf))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "test.csv"), index=False)
    DATA = os.path.join(ROOT, "gate", "data_v2_ref")                    # test-time refined NEO basis
    rr = []
    for split in ("test_in", "test_ood", "test_real"):
        for name, e_full, e_base, _ in evaluate(model, load_split(split)):
            for model_name, e in (("ours", e_full), ("ours-kappa", e_base)):
                rr.append(dict(model=model_name, split=split, shape=name, rel=float(e.mean())))
    pd.DataFrame(rr).to_csv(os.path.join(out, "test_refined.csv"), index=False)
    np.savez_compressed(os.path.join(out, "kappa_hat.npz"), **{k.replace("/", "__"): v for k, v in kap_store.items()})
    summ = df.groupby(["model", "split"]).rel.mean().unstack().to_dict()
    json.dump(dict(model="ours", feat=a.feat, N=a.N, seed=a.seed, params=n_params(model), best_val=best_val, g_val=g_val,
                   t_train_s=t_train, mean_rel=summ, log=log), open(os.path.join(out, "result.json"), "w"), indent=1)
    print(df.groupby(["model", "split"]).rel.mean().round(4).to_string(), f"\ntrain {t_train:.0f}s params {n_params(model)}")


if __name__ == "__main__":
    main()

"""P0 (GPU part): zero-shot evaluation of the stage-1 models on the P0 sets -> results_p0/

For every point set (data_p0/T0..T3) the model-side inputs are rebuilt from the points only
(NEO basis + Rayleigh–Ritz with the calibrated pc L, r, r_p) and all trained stage-1 models are
evaluated without retraining against the fine-mesh ground truth.

    CUDA_VISIBLE_DEVICES=<gpu> python stage1/p0_eval.py
"""
import common  # noqa: F401

import json
import os
import time

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch
from robust_laplacian import point_cloud_laplacian

import train as T
import train_don_fair as DF
from common import DATA, K_MODES, NEO_CKPT, PC_NEIGHBORS, RESULTS, TAUS
from g2pt.data.transforms import normalize_pc
from g2pt.training.pretrain import PretrainTraining
from g2pt.utils.ortho_operations import qr_orthogonalization
from models import DeepONetBase, Galerkin, LNeoResidual
from report_eps import md_table

P0 = os.path.join(common.ROOT, "data_p0")
OUT = os.path.join(common.ROOT, "results_p0")
EPS = os.path.join(common.ROOT, "results_eps")
SETS = ["T0", "T1", "T2", "T3"]


def neo_inputs(model, V, c_pc, m_pc, U0, t):
    pc = normalize_pc(V, 0)
    _, Mn = point_cloud_laplacian(pc, n_neighbors=PC_NEIGHBORS)
    mass = torch.tensor(Mn.diagonal(), dtype=torch.float32, device="cuda").view(1, -1, 1)
    x = torch.tensor(pc, dtype=torch.float32, device="cuda").view(1, -1, 3)
    with torch.no_grad():
        F = model.model(x, x, mass)
        S = qr_orthogonalization(F, mass)
    Lp, _ = point_cloud_laplacian(V, n_neighbors=PC_NEIGHBORS)
    Lp = (c_pc * Lp).tocsc()
    S = S[0].double().cpu().numpy()
    lam, Y = la.eigh(S.T @ (Lp @ S), (S * m_pc[:, None]).T @ S)
    Phi = S @ Y[:, :K_MODES]
    F = F[0].double().cpu().numpy()
    s_F = np.sqrt((m_pc[:, None] * F ** 2).sum() / (m_pc.sum() * F.shape[1]))
    F_hat = F / s_F
    z = (m_pc[:, None] * F_hat).sum(0) / m_pc.sum()
    # residual + physics prior
    r = U0 - (Phi @ (Phi.T @ (m_pc[:, None] * U0.T))).T
    M = sp.diags(m_pc).tocsc()
    rp = np.stack([sla.splu((M + tj * Lp).tocsc()).solve(m_pc[:, None] * r.T).T for tj in t])
    return dict(F=F_hat, z=z, Phi_neo=T.canon_sign(Phi), lam_neo=lam[:K_MODES], r=r, rp=rp)


def load_set(name, neo):
    files = sorted(f for f in os.listdir(os.path.join(P0, name)) if f.endswith(".npz"))
    keys = ["x", "m_mesh", "m_pc", "u0", "U", "t", "Phi_ex", "lam_ex", "Phi_neo", "lam_neo", "F", "z",
            "c_pc", "r", "rp", "u_pc"]
    buf = {k: [] for k in keys}
    sids = []
    for f in files:
        d = np.load(os.path.join(P0, name, f))
        V = d["V"].astype(np.float64)
        e = neo_inputs(neo, V, float(d["c_pc"]), d["M_pc"], d["U0"].astype(np.float64), d["t"])
        buf["x"].append(d["V"]); buf["m_mesh"].append(d["m_mesh"]); buf["m_pc"].append(d["M_pc"])
        buf["u0"].append(d["U0"]); buf["U"].append(d["U"]); buf["t"].append(d["t"])
        buf["Phi_ex"].append(T.canon_sign(d["Phi_ex"])); buf["lam_ex"].append(d["lam_ex"])
        buf["c_pc"].append(d["c_pc"]); buf["u_pc"].append(d["u_pc"])
        for k in ("Phi_neo", "lam_neo", "F", "z", "r", "rp"):
            buf[k].append(e[k])
        sids.append(int(f[6:9]))
    D = {k: torch.tensor(np.stack(v), dtype=torch.float16 if k == "F" else torch.float32, device=T.DEV)
         for k, v in buf.items()}
    D["coef_ex"] = torch.einsum("snk,sn,sin->ski", D["Phi_ex"], D["m_pc"], D["u0"])
    D["coef_neo"] = torch.einsum("snk,sn,sin->ski", D["Phi_neo"], D["m_pc"], D["u0"])
    return D, sids


def batch(D, ids, basis):
    it = torch.as_tensor(ids, device=T.DEV)
    b = {k: D[k][it] for k in ("x", "m_pc", "u0", "t", "z", "coef_neo", "r", "rp")}
    b["F"] = D["F"][it].float()
    b["Phi"], b["lam"], b["coef"] = D[f"Phi_{basis}"][it], D[f"lam_{basis}"][it], D[f"coef_{basis}"][it]
    return b, D["U"][it], D["m_mesh"][it]


T.batch = batch


def r_scale_stage1():
    split = pd.read_csv(os.path.join(DATA, "split.csv"))
    num = den = 0.0
    for sid in split[split.split == "train"].sid:
        m = np.load(os.path.join(DATA, f"shape_{sid:03d}.npz"))["M_pc"]
        r = np.load(os.path.join(DATA, f"eps_{sid:03d}.npz"))["r"].astype(np.float64)
        num += (m * r ** 2).sum()
        den += m.sum() * r.shape[0]
    return float(1 / np.sqrt(num / den))


def models(r_scale):
    out = {}
    out["G-exact"] = (Galerkin(), "ex")
    out["G-neo"] = (Galerkin(), "neo")
    lneo = torch.load(os.path.join(RESULTS, "model_L-neo.pt"), map_location="cpu")
    for name, kw, path in [("(a) L-neo", {}, None), ("(b) L-neo+rp", dict(add_rp=True), None),
                           ("(c) L-neo+rp+eps", dict(add_rp=True, eps="rp"), "model_(c).pt"),
                           ("(d) L-neo+eps(r)", dict(eps="norp"), "model_(d).pt"),
                           ("(d') L-neo+eps(r,rp-in)", dict(eps="rp"), "model_(d').pt")]:
        m = LNeoResidual(r_scale=r_scale, **kw)
        if path is None:
            m.g.load_state_dict({k[2:]: v for k, v in lneo.items() if k.startswith("g.")})
        else:
            m.load_state_dict(torch.load(os.path.join(EPS, path), map_location="cpu"))
        out[name] = (m, "neo")
    don = DeepONetBase()
    don.load_state_dict(torch.load(os.path.join(EPS, "model_DON-base-converged.pt"), map_location="cpu"))
    out["DON-base (정규화 없음)"] = (don, "ex")
    best_path = os.path.join(common.ROOT, "results_don_fair", "best_matched.json")
    if os.path.exists(best_path):
        best = json.load(open(best_path))
        i = [r["name"] for r in DF.RUNS].index(best["name"])
        m = DF.DeepONetNorm(int(best["p"]), float(best["alpha"]))
        m.load_state_dict(torch.load(os.path.join(common.ROOT, "results_don_fair", f"model_run{i}.pt"),
                                     map_location="cpu"))
        out[f"DON-matched best ({best['name'].split(' ', 1)[1]})"] = (m, "ex")
    return {k: (m.to(T.DEV).eval(), b) for k, (m, b) in out.items()}


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "set CUDA_VISIBLE_DEVICES"
    os.makedirs(OUT, exist_ok=True)
    neo = PretrainTraining.load_from_checkpoint(NEO_CKPT, weights_only=False, strict=False).to("cuda").eval()
    M = models(r_scale_stage1())
    idx = pd.read_csv(os.path.join(P0, "index.csv"))
    rows = []
    for name in SETS:
        t0 = time.time()
        D, sids = load_set(name, neo)
        fam = idx[idx.set == name].set_index("sid").family
        ids = np.arange(len(sids))
        res = {"PC-discrete": T.rel_err(D["u_pc"], D["U"], D["m_mesh"]).cpu().numpy()}
        for mname, (model, basis) in M.items():
            res[mname] = T.evaluate(model, D, ids, basis)
        for mname, e in res.items():
            for si, sid in enumerate(sids):
                for ti, tau in enumerate(TAUS):
                    for ic in range(e.shape[2]):
                        rows.append(dict(set=name, model=mname, sid=sid, family=fam[sid], ic=ic,
                                         tau_idx=ti, tau=tau, err=float(e[si, ti, ic])))
        print(f"{name}: {len(sids)} shapes, {time.time() - t0:.0f}s", flush=True)
        del D
        torch.cuda.empty_cache()
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(OUT, "errors.csv"), index=False)
    small = E[E.tau <= 0.031].pivot_table(index="model", columns="set", values="err", aggfunc="mean")
    allt = E.pivot_table(index="model", columns="set", values="err", aggfunc="mean")
    order = [m for m in ["PC-discrete", "G-exact", "G-neo", "(a) L-neo", "(b) L-neo+rp", "(c) L-neo+rp+eps",
                         "(d) L-neo+eps(r)", "(d') L-neo+eps(r,rp-in)", "DON-base (정규화 없음)"] if m in small.index]
    order += [m for m in small.index if m not in order]
    with open(os.path.join(OUT, "tables.md"), "w") as fh:
        fh.write("## Mean error, τ ≤ 0.03\n\n" + md_table(small.loc[order]) + "\n\n")
        fh.write("## Mean error, all τ\n\n" + md_table(allt.loc[order]) + "\n\n")
        for name in SETS:
            tab = E[E.set == name].pivot_table(index="model", columns="tau", values="err", aggfunc="mean").loc[order]
            fh.write(f"## {name} by τ\n\n" + md_table(tab) + "\n\n")
    t0_vs_4k(E)
    print(open(os.path.join(OUT, "tables.md")).read())


def t0_vs_4k(E):
    """T0 (stage-1 test points) scored against the fine GT vs against the stage-1 4k-mesh GT:
    (c) vs PC-discrete by τ, plus the 4k-vs-fine GT discrepancy itself."""
    E4 = pd.read_csv(os.path.join(EPS, "errors_test.csv"))
    rows = []
    gt_gap = []
    for f in sorted(os.listdir(os.path.join(P0, "T0"))):
        if not f.endswith(".npz"):
            continue
        sid = int(f[6:9])
        fine = np.load(os.path.join(P0, "T0", f))
        coarse = np.load(os.path.join(DATA, f"shape_{sid:03d}.npz"))
        w = fine["m_mesh"]
        d = fine["U"].astype(np.float64) - coarse["U"].astype(np.float64)                 # (T, I, N)
        gt_gap.append(np.sqrt((w * d ** 2).sum(-1) / (w * fine["U"].astype(np.float64) ** 2).sum(-1)))
    gt_gap = np.stack(gt_gap).mean((0, 2))
    for ti, tau in enumerate(TAUS):
        r = dict(tau=tau)
        for m, key in [("(c) L-neo+rp+eps", "c"), ("PC-discrete", "pc")]:
            r[f"{key}_4kGT"] = E4[(E4.model == m) & (E4.tau_idx == ti)].err.mean()
            r[f"{key}_fineGT"] = E[(E.set == "T0") & (E.model == m) & (E.tau_idx == ti)].err.mean()
        r["ratio_c/pc_4kGT"] = r["c_4kGT"] / r["pc_4kGT"]
        r["ratio_c/pc_fineGT"] = r["c_fineGT"] / r["pc_fineGT"]
        r["4kGT_vs_fineGT"] = gt_gap[ti]
        rows.append(r)
    tab = pd.DataFrame(rows).set_index("tau")
    tab.to_csv(os.path.join(OUT, "t0_fine_vs_4k.csv"))
    small = tab[tab.index <= 0.031].mean()
    with open(os.path.join(OUT, "tables.md"), "a") as fh:
        fh.write("## T0: (c) vs PC-discrete, fine-mesh GT vs stage-1 4k-mesh GT\n\n" + md_table(tab, "{:.3g}") + "\n\n")
        fh.write("mean over τ ≤ 0.03: " + ", ".join(f"{k} {v:.3g}" for k, v in small.items()) + "\n")


if __name__ == "__main__":
    main()

"""Gate v2 supplement: the trained v2 models (no retraining) evaluated with the test-time refined NEO basis
(2 inverse-iteration steps + Rayleigh-Ritz; Phi, coef, r, r_p taken from gate/data_ref, truth from gate/data_v2)
-> runs_v2/<run>/test_refined.csv, gate/v2_refined_compare.png, table printed.
kappa_hat depends only on (d, NEO features) -> identical for both bases by construction.

    CUDA_VISIBLE_DEVICES=<gpu> python stage3/gate_v2_refined.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import gate_v2_ours as G

ROOT = os.path.dirname(os.path.abspath(__file__))
V2, REFD, OUT = (os.path.join(ROOT, "gate", d) for d in ("data_v2", "data_ref", "data_v2_ref"))


def build():
    for split in ("test_in", "test_ood", "test_real"):
        os.makedirs(os.path.join(OUT, split), exist_ok=True)
        for p in sorted(glob.glob(os.path.join(V2, split, "*.npz"))):
            q = os.path.join(OUT, split, os.path.basename(p))
            if os.path.exists(q):
                continue
            z, r = dict(np.load(p)), np.load(os.path.join(REFD, split, os.path.basename(p)))
            for k in ("Phi", "lam", "coef", "r", "rp"):
                z[k] = r[k]
            np.savez(q, **z)


def main():
    build()
    G.DATA = OUT
    rows = []
    for run in sorted(glob.glob(os.path.join(G.RUNS, "ours_N*_s*"))):
        model = G.Model().to(G.DEV)
        model.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=G.DEV))
        N, s = os.path.basename(run).split("_")[1:]
        for split in ("test_in", "test_ood", "test_real"):
            for name, e_full, e_base, _ in G.evaluate(model, G.load_split(split)):
                for mname, e in (("ours", e_full), ("ours-kappa", e_base)):
                    rows.append(dict(run=os.path.basename(run), N=int(N[1:]), seed=int(s[1:]), model=mname, split=split,
                                     shape=name, rel=float(e.mean())))
        pd.DataFrame([r for r in rows if r["run"] == os.path.basename(run)]).to_csv(os.path.join(run, "test_refined.csv"), index=False)
    R = pd.DataFrame(rows)
    raw = []
    for run in sorted(glob.glob(os.path.join(G.RUNS, "ours_N*_s*"))):
        t = pd.read_csv(os.path.join(run, "test.csv")).groupby(["model", "split", "shape"], as_index=False).rel.mean()
        N, s = os.path.basename(run).split("_")[1:]
        t["N"], t["seed"] = int(N[1:]), int(s[1:])
        raw.append(t)
    Rw = pd.concat(raw)
    thin = "thingi472052"
    for excl in (False, True):
        a = Rw[Rw["shape"] != thin] if excl else Rw
        b = R[R["shape"] != thin] if excl else R
        ta = a.groupby(["model", "N", "seed", "split"]).rel.mean().groupby(["model", "N", "split"]).mean().unstack("split")
        tb = b.groupby(["model", "N", "seed", "split"]).rel.mean().groupby(["model", "N", "split"]).mean().unstack("split")
        print(f"\n== {'excluding ' + thin if excl else 'all shapes'}: raw NEO basis | refined NEO basis ==")
        print(ta.join(tb, lsuffix="_raw", rsuffix="_refined").round(4).to_string())
    # per-shape comparison figure (ours kappa+eps, N = 200, seed mean)
    pa = Rw[(Rw.model == "ours") & (Rw.N == 200)].groupby(["split", "shape"]).rel.mean()
    pb = R[(R.model == "ours") & (R.N == 200)].groupby(["split", "shape"]).rel.mean()
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    col = {"test_in": "#2a78d6", "test_ood": "#eb6834", "test_real": "#1baf7a"}
    lab = {"test_in": "training families (unseen shapes)", "test_ood": "synthetic OOD", "test_real": "real shapes"}
    for split in ("test_in", "test_ood", "test_real"):
        x, y = pa.loc[split], pb.loc[split]
        ax.plot(x.values, y.loc[x.index].values, "o", ms=8, color=col[split], mec="white", mew=1.2, label=lab[split])
        if split == "test_real":
            for n in x.index:
                ax.annotate(n.replace("thingi", "T"), (x[n], y[n]), xytext=(4, 3), textcoords="offset points", fontsize=7, color="#52514e")
    lim = [min(pa.min(), pb.min()) * 0.8, max(pa.max(), pb.max()) * 1.2]
    ax.plot(lim, lim, color="#52514e", lw=1, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("relative L2 error, raw NEO basis", color="#52514e"); ax.set_ylabel("relative L2 error, test-time refined NEO basis", color="#52514e")
    ax.set_title("Gate v2, our model (N=200, κ̂+ε, no retraining): basis swap at test time", loc="left", fontsize=10)
    ax.grid(True, color="#e4e3df"); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "gate", "v2_refined_compare.png"), dpi=110, facecolor="#fcfcfb")
    R.to_csv(os.path.join(ROOT, "gate", "v2_refined_all.csv"), index=False)


if __name__ == "__main__":
    main()

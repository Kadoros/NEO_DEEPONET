"""Gate supplement (reference row, not part of the pre-registered design): the trained "ours" models evaluated with
the test-time REFINED NEO basis (2 inverse-iteration steps + Rayleigh-Ritz; no retraining) -> gate/runs/<run>/test_refined.csv

For each test shape the basis-dependent fields are recomputed from the refined basis: Phi, lam, coef = Phi^T M u0,
r = u0 - Phi coef, r_p = (M + t K)^{-1} M r; the NEO features F, z (network inputs) are unchanged.

    CUDA_VISIBLE_DEVICES=<gpu> python stage3/gate_refined.py [--procs 8]
"""
import argparse
import glob
import os
from multiprocessing import get_context

import numpy as np

import e1_basis as E
import gate_data as GD

REF = os.path.join(E.ROOT, "gate", "data_ref")


def refine_task(args):
    split, name, src = args
    import c_lumps_data as CL
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    import scipy.linalg as la
    z = dict(np.load(os.path.join(GD.OUT, split, f"{name}.npz")))
    Y, tets, grads, vol, m = GD.load(src)
    K = GD.stiffness(tets, grads, vol, len(Y)).tocsc()
    Q = CL.refine2(z["Phi"].astype(np.float64), K, m, float(z["lam1"]))
    A, B = Q.T @ (K @ Q), (Q * m[:, None]).T @ Q
    lam, W = la.eigh(0.5 * (A + A.T), 0.5 * (B + B.T))
    Phi = Q @ W
    u0 = z["u0"].astype(np.float64)
    coef = Phi.T @ (m[:, None] * u0.T)
    r = u0 - (Phi @ coef).T
    rp = np.stack([sla.splu((sp.diags(m) + tj * K).tocsc()).solve(m[:, None] * r.T).T for tj in z["t"].astype(np.float64)])
    z.update(Phi=Phi.astype(np.float32), lam=lam.astype(np.float32), coef=coef.astype(np.float32), r=r.astype(np.float32),
             rp=rp.astype(np.float32))
    os.makedirs(os.path.join(REF, split), exist_ok=True)
    np.savez(os.path.join(REF, split, f"{name}.npz"), **z)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    tasks = [(s, n, src) for s, n, src in GD.sources() if s in ("test_in", "test_ood", "test_real")
             and not os.path.exists(os.path.join(REF, s, f"{n}.npz"))]
    with get_context("spawn").Pool(a.procs) as pool:
        for n in pool.imap_unordered(refine_task, tasks):
            print("refined", n, flush=True)
    import pandas as pd
    import torch
    import gate_ours as G
    G.DATA = REF
    for run in sorted(glob.glob(os.path.join(G.RUNS, "ours_N*_s*"))):
        if not os.path.exists(os.path.join(run, "model.pt")):
            continue
        model = G.LNeoResidual(eps="rp").to(G.DEV)
        model.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=G.DEV))
        rows = []
        for split in ("test_in", "test_ood", "test_real"):
            for name, e in G.evaluate(model, G.load_split(split)):
                for j in range(e.shape[0]):
                    for i in range(e.shape[1]):
                        rows.append(dict(split=split, shape=name, tau_idx=j, ic=i, rel=float(e[j, i])))
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(run, "test_refined.csv"), index=False)
        print(os.path.basename(run), df.groupby("split").rel.mean().round(4).to_dict(), flush=True)


if __name__ == "__main__":
    main()

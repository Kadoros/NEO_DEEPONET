"""Shape-shift-robust local geometric node features for the kappa-network comparison -> gate/data_feat/<split>/<name>.npz

Heat kernel signature from the REFINED NEO eigenpairs (2 inverse-iteration steps of the raw NEO basis + Rayleigh-Ritz,
the same refined basis as at test time):  h_j(x) = sum_{k>=1} exp(-s_j lambda_k / lambda_1) phi_k(x)^2,
s_j in {0.1, 0.3, 1, 3, 10, 30}; each h_j divided by its mass-weighted mean over the shape (scale free), then log.
Invariant to rotation / translation; depends on local curvature, thickness and distance to the boundary (the
depth d itself is NOT an input of variant (iii)).

    python stage3/gate_feat.py [--procs 8]
"""
import argparse
import glob
import os
from multiprocessing import get_context

import numpy as np

import e1_basis as E
import gate_data as GD

OUT = os.path.join(E.ROOT, "gate", "data_feat")
S = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])


def task(args):
    split, name, src = args
    import scipy.linalg as la
    import c_lumps_data as CL
    z = np.load(os.path.join(GD.OUT, split, f"{name}.npz"))
    ref = os.path.join(E.ROOT, "gate", "data_ref", split, f"{name}.npz")
    Y, tets, grads, vol, m = GD.load(src)
    K = GD.stiffness(tets, grads, vol, len(Y)).tocsc()
    if os.path.exists(ref):
        Q = np.load(ref)["Phi"].astype(np.float64)
    else:
        Q = CL.refine2(z["Phi"].astype(np.float64), K, m, float(z["lam1"]))
    A, B = Q.T @ (K @ Q), (Q * m[:, None]).T @ Q
    lam, W = la.eigh(0.5 * (A + A.T), 0.5 * (B + B.T))
    Phi = Q @ W
    lam1 = lam[1]
    h = np.stack([(np.exp(-s * lam[1:] / lam1)[None] * Phi[:, 1:] ** 2).sum(1) for s in S], 1)     # (N, 6)
    h = h / ((m[:, None] * h).sum(0) / m.sum())
    os.makedirs(os.path.join(OUT, split), exist_ok=True)
    np.savez(os.path.join(OUT, split, f"{name}.npz"), hks=np.log(h + 1e-12).astype(np.float32))
    return f"{split}/{name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    tasks = [x for x in GD.sources() if not os.path.exists(os.path.join(OUT, x[0], f"{x[1]}.npz"))]
    with get_context("spawn").Pool(a.procs) as pool:
        for msg in pool.imap_unordered(task, tasks):
            print(msg, flush=True)


if __name__ == "__main__":
    main()

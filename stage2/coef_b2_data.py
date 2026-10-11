"""Branch B-2 data: many shapes, each with its own hidden stiff lumps -> stage2/coef/b2_data/<split>/<name>.npz

Per shape (same physics and discretisation as B-1, coef_b1.build_problem):
  N = 8192 quasi-uniform points, Delaunay-P1, 1-3 lumps (kappa = 5 inside, 1 outside, radius 0.15-0.25),
  truth u(t) = expm_multiply with the heterogeneous operator, 8 Gaussian ICs x 6 times (t = tau / lambda_1 of the
  homogeneous operator).  Stored: points, tets, vol, P1 gradients, lumped mass, 128 homogeneous eigenvectors,
  observations (full interior field), true kappa (per point and per tet; used for evaluation only), lump centres.
Families: sh_ball + superquadric (train 80 + 80, test 10 + 10 unseen shapes/lumps).

    python stage2/coef_b2_data.py [--procs 12]            (env neo, CPU)
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

import coef_b1 as B1

OUT = os.path.join(B1.ROOT, "coef", "b2_data")
FAMILIES = ["sh_ball", "superquadric"]
SPLIT = {"train": 80, "test": 10}
OOD = {"test_ood": (["torus", "cup"], 10)}   # families never seen in B-2 training (--ood)
MODES = 128


def task(args):
    split, family, idx = args
    t0 = time.time()
    fam_i = (FAMILIES + ["torus", "cup"]).index(family)
    seed = idx + {"train": 0, "test": 100000, "test_ood": 200000}[split]
    n_blobs = 1 + (idx % 3)
    try:
        pb = B1.build_problem(seed, n_blobs, family=family, key=(2026, 10, fam_i))
    except RuntimeError as e:                                         # no room for the lumps: skip, record
        return f"{family}_{idx:04d} SKIPPED ({e})", time.time() - t0
    Y, op, rng = pb["Y"], pb["op"], pb["rng"]
    m = op.m
    kappa_e = pb["kappa"](Y[op.tets].mean(1))
    K1 = op.K.tocsc()
    lam, Phi = sla.eigsh(K1, k=MODES, M=sp.diags(m).tocsc(), sigma=-1e-6)
    o = np.argsort(lam); lam, Phi = lam[o], Phi[:, o]
    t = B1.TAUS / lam[1]
    U0 = []
    for _ in range(B1.N_IC):
        cidx = rng.integers(len(Y))
        U0.append(np.exp(-((Y - Y[cidx]) ** 2).sum(1) / (2 * 0.25 ** 2)))
    U0 = np.stack(U0, 1)
    A = (sp.diags(1 / m) @ op.assemble(kappa_e).tocsc()).tocsr()
    Uobs = np.stack([sla.expm_multiply(-tj * A, U0) for tj in t])                      # (T, N, I)
    name = f"{family}_{idx:04d}"
    np.savez(os.path.join(OUT, split, name + ".npz"), Y=Y.astype(np.float32), tets=op.tets.astype(np.int32),
             vol=op.vol, grads=op.grads.astype(np.float32), m=m, Phi=Phi.astype(np.float32), lam=lam, t=t,
             U0=U0.astype(np.float32), Uobs=Uobs.astype(np.float32), kappa=pb["kappa"](Y).astype(np.float32),
             kappa_e=kappa_e.astype(np.float32), centres=pb["centres"], radii=pb["radii"])
    return name, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--ood", action="store_true", help="only the out-of-family test split (torus, cup)")
    a = ap.parse_args()
    tasks = []
    plan = {k: (fams, n) for k, (fams, n) in OOD.items()} if a.ood else {k: (FAMILIES, n) for k, n in SPLIT.items()}
    for split, (fams, n) in plan.items():
        os.makedirs(os.path.join(OUT, split), exist_ok=True)
        tasks += [(split, f, i) for f in fams for i in range(n)]
    t0 = time.time()
    with Pool(a.procs) as p:
        for name, dt in p.imap_unordered(task, tasks):
            print(f"{name} {dt:.0f}s", flush=True)
    print(f"{len(tasks)} shapes in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

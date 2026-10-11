"""C (c): hidden stiff lumps on the real shapes, in the B-2/B-4 data format -> stage2/coef/b2_data/test_real/

Per real shape (cleaned quasi-8192 operator) three configurations: 1, 2 and 3 lumps (kappa = 5 inside, 1 outside,
smooth edge 0.02).  Lump centres are drawn at mixed depths below the surface (depth = distance to the nearest
surface node; target depth drawn uniformly between r + 0.02 and the deepest point), radius U(0.15, 0.25) capped at
0.7 x the local depth (thin shapes get smaller lumps; recorded), no overlap (gap 0.05).  Physics exactly as B-2:
8 Gaussian ICs (sigma 0.25), 6 times t = TAUS / lambda_1, truth = expm_multiply with the heterogeneous operator.
Also writes the three Galerkin bases per case -> stage2/coef/b3_basis/test_real/<case>_<basis>.npz
  neo256     scalar volume NEO (t4096) + Rayleigh-Ritz (from real_shapes/cache, field Pn)
  neoinv2    the same refined by 2 steps of inverse iteration (+ Rayleigh-Ritz)
  arpack256  exact ARPACK eigenvectors of the same operator

    python stage3/c_lumps_data.py [--procs 8]
"""
import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.spatial import cKDTree

import e1_basis as E
import e1_v2 as V

S2 = E.S2
sys.path.insert(0, S2)
import coef_b1 as B1  # noqa: E402

RS = os.path.join(E.ROOT, "real_shapes")
OUT = os.path.join(S2, "coef", "b2_data", "test_real")
BAS = os.path.join(S2, "coef", "b3_basis", "test_real")
SEED = 20260929


def stiffness(op, kappa_e):
    Kl = (kappa_e * op.vol)[:, None, None] * np.einsum("tki,tkj->tij", op.grads, op.grads)
    T, n = op.tets, len(op.m)
    return sp.coo_matrix((Kl.ravel(), (np.repeat(T, 4, 1).ravel(), np.tile(T, (1, 4)).ravel())), shape=(n, n)).tocsr()


def place_lumps(Y, depth, n, rng):
    cs, rs = [], []
    dmax = depth.max()
    for _ in range(4000):
        if len(cs) == n:
            break
        r0 = rng.uniform(0.15, 0.25)
        lo = min(r0, 0.7 * dmax) + 0.02
        target = dmax if lo >= dmax else rng.uniform(lo, dmax)
        cand = np.where(np.abs(depth - target) < 0.02)[0]
        if len(cand) == 0:
            continue
        i = rng.choice(cand)
        r = min(r0, 0.7 * depth[i])
        if r < 0.05:
            continue
        if all(np.linalg.norm(Y[i] - c2) > r + r2 + 0.05 for c2, r2 in zip(cs, rs)):
            cs.append(Y[i]); rs.append(r)
    return np.array(cs), np.array(rs)


def refine2(Q, K, m, lam1):
    lu = sla.splu((K + 1e-3 * lam1 * sp.diags(m)).tocsc())
    for _ in range(2):
        Q = lu.solve(m[:, None] * Q)
        A, B = Q.T @ (K @ Q), (Q * m[:, None]).T @ Q
        _, W = la.eigh(0.5 * (A + A.T), 0.5 * (B + B.T))
        Q = Q @ W
    return Q


def task(path):
    t0 = time.time()
    shape = os.path.basename(path)[:-len("_quasi_8192.npz")]
    z = np.load(path)
    op = E.Op(z)
    Y, m = z["Y"], op.m
    K1 = op.K.tocsc()
    ws = V.surface_weights(Y, op.tets)
    depth = cKDTree(Y[ws > 0]).query(Y)[0]
    bases = {"neo256": z["Pn"], "neoinv2": refine2(z["Pn"].astype(np.float64), K1, m, float(z["lam_s"][1])),
             "arpack256": z["Ps"]}
    lam = z["lam_s"]
    t = B1.TAUS / lam[1]
    cen = Y[op.tets].mean(1)
    out = []
    if all(os.path.exists(os.path.join(OUT, f"{shape}_c{k}.npz")) for k in (1, 2, 3)):
        return shape, 0.0, ["already done"]
    for k in (1, 2, 3):
        rng = np.random.default_rng([SEED, sum(map(ord, shape)), k])
        C, R = place_lumps(Y, depth, k, rng)
        if len(C) == 0:
            out.append(f"{shape} c{k}: no room for a lump")
            continue
        kappa = lambda x: 1 + (B1.K_HI - 1) * np.clip(sum(B1.smooth_ball(x, c, r) for c, r in zip(C, R)), 0, 1)
        kappa_e = kappa(cen)
        U0 = []
        for _ in range(B1.N_IC):
            cidx = rng.integers(len(Y))
            U0.append(np.exp(-((Y - Y[cidx]) ** 2).sum(1) / (2 * 0.25 ** 2)))
        U0 = np.stack(U0, 1)
        A = (sp.diags(1 / m) @ stiffness(op, kappa_e).tocsc()).tocsr()
        Uobs = np.stack([sla.expm_multiply(-tj * A, U0) for tj in t])
        name = f"{shape}_c{k}"
        np.savez(os.path.join(OUT, name + ".npz"), Y=Y.astype(np.float32), tets=op.tets.astype(np.int32), vol=op.vol,
                 grads=op.grads.astype(np.float32), m=m, Phi=z["Ps"][:, :128].astype(np.float32), lam=lam[:128], t=t,
                 U0=U0.astype(np.float32), Uobs=Uobs.astype(np.float32), kappa=kappa(Y).astype(np.float32),
                 kappa_e=kappa_e.astype(np.float32), centres=C, radii=R,
                 lump_depth=np.array([depth[np.argmin(np.linalg.norm(Y - c, axis=1))] for c in C]))
        for b, Q in bases.items():
            np.savez(os.path.join(BAS, f"{name}_{b}.npz"), Phi=np.asarray(Q, np.float32))
        out.append(f"{name}: {len(C)}/{k} lumps, r={np.round(R, 3).tolist()}")
    return shape, time.time() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(BAS, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RS, "cache", "*_quasi_8192.npz")))
    with Pool(a.procs) as pool:
        for shape, dt, lines in pool.imap_unordered(task, files):
            print(f"{shape} {dt:.0f}s", flush=True)
            for l in lines:
                print("   ", l, flush=True)


if __name__ == "__main__":
    main()

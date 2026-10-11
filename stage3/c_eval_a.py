"""C (a): volume NEO basis quality on real shapes (no retraining) -> real_shapes/a_basis.csv

For every cached operator (shape x {quasi, random} x {8192, 16384}):
  NEO (stage2/main/t4096) 256-dim output -> M-orthonormal -> Rayleigh-Ritz with the scalar stiffness
  span loss   1 - ||Q^T M U_k||_F^2 / k   (U_k = exact ARPACK eigenvectors, k = 32 and 96)
  eig error   mean |mu_i / lambda_i - 1| over i = 1..95 and i = 1..10 (Ritz values vs exact)
The NEO basis is stored back into the cache (field Pn) for (c) / (d).
Shape indicators on the quasi-8192 operator: Korn constant (e1_v2.korn_constant), thickness d_max / V^(1/3).

    CUDA_VISIBLE_DEVICES=<gpu> python stage3/c_eval_a.py [--procs 6]
"""
import argparse
import glob
import os
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

import e1_basis as E

RS = os.path.join(E.ROOT, "real_shapes")


def neo_all(files):
    rows = []
    for p in files:
        z = dict(np.load(p))
        if "Pn" not in z:
            t0 = time.time()
            z["Pn"] = E.neo_basis(z["Y"], z["m"], E.Op(z).K.tocsc())
            z["t_neo"] = time.time() - t0
            np.savez(p, **z)
        m = z["m"]
        Q, U, lam = z["Pn"], z["Ps"], z["lam_s"]
        A = Q.T @ (E.Op(z).K @ Q)
        mu = np.linalg.eigvalsh(0.5 * (A + A.T))                        # Q is M-orthonormal (Ritz basis)
        name, kind, N = os.path.basename(p)[:-4].rsplit("_", 2)
        r = dict(shape=name, kind=kind, N=int(N), sliver=float(z["sliver"]), carved=float(z["carved"]),
                 t_neo_s=float(z["t_neo"]), t_arpack_s=float(z["t_scalar"]))
        for k in (32, 96):
            r[f"span{k}"] = float(1 - ((Q.T @ (m[:, None] * U[:, :k])) ** 2).sum() / k)
        rel = mu[1:96] / lam[1:96] - 1
        r.update(eig_relerr=float(np.abs(rel).mean()), eig_relerr_1to10=float(np.abs(rel[:10]).mean()), eig_bias=float(rel.mean()))
        rows.append(r)
        print(r, flush=True)
    return rows


def indicators(p):
    import e1_v2 as V
    z = np.load(p)
    op = E.Op(z)
    m3 = np.repeat(op.m, 3)
    C_K, _ = V.korn_constant(op, m3, z["Y"])
    ws = V.surface_weights(z["Y"], op.tets)
    _, dmax = V.depth_lumps(z["Y"], np.where(ws > 0)[0])
    return os.path.basename(p)[:-4].rsplit("_", 2)[0], C_K, dmax / float(op.vol.sum()) ** (1 / 3), dmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(RS, "cache", "*.npz")))
    ctx = get_context("spawn")                                         # CUDA in the parent -> spawn the CPU pool first
    with ctx.Pool(a.procs) as pool:
        ind_async = pool.map_async(indicators, [p for p in files if p.endswith("_quasi_8192.npz")])
        rows = neo_all(files)
        ind = ind_async.get()
    df = pd.DataFrame(rows)
    I = pd.DataFrame(ind, columns=["shape", "korn", "thick", "dmax"])
    df = df.merge(I, on="shape", how="left")
    df.to_csv(os.path.join(RS, "a_basis.csv"), index=False)
    fmt = lambda v: f"{v:.3g}"
    print(df.pivot_table(index=["shape"], columns=["kind", "N"], values="span96").to_string(float_format=fmt))
    print(df.pivot_table(index=["shape"], columns=["kind", "N"], values="eig_relerr").to_string(float_format=fmt))
    print(I.sort_values("korn").to_string(float_format=fmt))


if __name__ == "__main__":
    main()

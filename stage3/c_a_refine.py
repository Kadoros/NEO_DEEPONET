"""C (a) diagnostic: does scalar inverse-iteration refinement of the NEO basis recover the exact low modes?
V <- (K + s M)^{-1} M V (s = 1e-3 lambda_1, rank-1 constant mode kept by M-orthonormalisation), 1 and 2 steps,
Rayleigh-Ritz after each; span loss (modes 1-32, 1-96) and eigenvalue error, per cache -> real_shapes/a_refine.csv
    python stage3/c_a_refine.py [--procs 6]"""
import argparse, glob, os, time
from multiprocessing import Pool
import numpy as np, pandas as pd, scipy.linalg as la, scipy.sparse as sp, scipy.sparse.linalg as sla
import e1_basis as E
RS = os.path.join(E.ROOT, "real_shapes")


def metrics(Q, U, lam, K, m):
    A = Q.T @ (K @ Q); B = (Q * m[:, None]).T @ Q
    mu, W = la.eigh(0.5 * (A + A.T), 0.5 * (B + B.T)); Q = Q @ W
    out = {f"span{k}": float(1 - ((Q.T @ (m[:, None] * U[:, :k])) ** 2).sum() / k) for k in (32, 96)}
    rel = mu[1:96] / lam[1:96] - 1
    out.update(eig_relerr=float(np.abs(rel).mean()), eig_relerr_1to10=float(np.abs(rel[:10]).mean()))
    return out, Q


def task(p):
    z = np.load(p); op = E.Op(z); K, m = op.K.tocsc(), op.m
    name, kind, N = os.path.basename(p)[:-4].rsplit("_", 2)
    t = time.time(); lu = sla.splu((K + 1e-3 * float(z["lam_s"][1]) * sp.diags(m)).tocsc()); t_lu = time.time() - t
    rows = []
    Q = z["Pn"].astype(np.float64)
    r, Q = metrics(Q, z["Ps"], z["lam_s"], K, m); rows.append(dict(shape=name, kind=kind, N=int(N), steps=0, **r))
    for it in (1, 2):
        t = time.time(); V = lu.solve(m[:, None] * Q)
        r, Q = metrics(V, z["Ps"], z["lam_s"], K, m)
        rows.append(dict(shape=name, kind=kind, N=int(N), steps=it, t_step_s=time.time() - t, t_lu_s=t_lu, **r))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=6); a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(RS, "cache", "*.npz")))
    with Pool(a.procs) as pool:
        rows = sum(pool.map(task, files), [])
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RS, "a_refine.csv"), index=False)
    q = df[(df.kind == "quasi") & (df.N == 8192)]
    print(q.pivot_table(index="shape", columns="steps", values=["span96", "eig_relerr_1to10"]).to_string(float_format=lambda v: f"{v:.3g}"))

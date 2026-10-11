"""Locking check of the full-order operators against the Lame thick-sphere solution -> stage3/lame/

Hollow sphere a = 0.5 <= r <= b = 1, internal pressure p = 1, outer surface free, E = 1.
  u_r = A r + B / r^2,  A = p a^3 (1 - 2 nu) / (E (b^3 - a^3)),  B = p a^3 b^3 (1 + nu) / (2 E (b^3 - a^3))
Free body -> rigid modes removed by the M-orthogonality constraint (KKT), in both the numerical and the analytic
field.  Operators: plain P1 and selective nodal averaging (ANP), on the stage-2 Delaunay-P1 point tets
(quasi-uniform points, N = 8192 / 16384 / 32768).  Also: the lowest 30 eigenvalues (free, M x I3) at nu = 0.3 and
0.49 to look for non-physical near-zero modes (anything between the 6 rigid zeros and the physical spectrum).

    python stage3/lame_check.py [--procs 6]            (CPU)
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as sla

import e1_basis as E
import elastic_ops as EO
from amg_solve import amg_solve

AMG_FROM_N = 32768                  # SuperLU on the KKT system is too slow here (>2 h per task); AMG-CG gives the
                                    # same solution (checked at N = 16384: identical errors to 5 digits)

OUT = os.path.join(E.ROOT, "lame")
A_IN, B_OUT = 0.5, 1.0
NUS = [0.3, 0.45, 0.49]


def build(N, seed):
    rng = np.random.default_rng([E.SEED, 99, N, seed])
    sdf = lambda x: np.minimum(B_OUT - np.linalg.norm(x, axis=1), np.linalg.norm(x, axis=1) - A_IN)
    geo = E.D0.geometry(sdf, 1.05, rng)
    for _ in range(5):
        P, is_s, h = E.D0.sample_points(sdf, geo, N, "quasi", rng)
        op = E.delaunay_p1(P, is_s, sdf, h)
        if op.info["isolated"] == 0 and len(P) == N:
            return P, op
    raise RuntimeError("sampling failed")


def exact(P, nu):
    r = np.linalg.norm(P, axis=1)
    A = A_IN ** 3 * (1 - 2 * nu) / (B_OUT ** 3 - A_IN ** 3)
    Bc = A_IN ** 3 * B_OUT ** 3 * (1 + nu) / (2 * (B_OUT ** 3 - A_IN ** 3))
    return ((A * r + Bc / r ** 2) / r)[:, None] * P


def pressure_load(P, T):
    F, nrm = E.boundary_faces(T, P)
    inner = np.linalg.norm(P[F].mean(1), axis=1) < 0.5 * (A_IN + B_OUT)
    f = np.zeros((len(P), 3))
    for k in range(3):
        np.add.at(f, F[inner, k], -nrm[inner] / 3)                      # p = 1 pushes the body outwards (-n_body)
    return f.ravel()


def remove_rigid(u, P, m3):
    R = E.rigid_modes(P)
    return u - R @ np.linalg.solve(R.T @ (m3[:, None] * R), R.T @ (m3 * u))


def task(args):
    N, seed = args
    t0 = time.time()
    P, op = build(N, seed)
    m3 = np.repeat(op.m, 3)
    f = pressure_load(P, op.tets)
    r = np.linalg.norm(P, axis=1)
    inner = r < A_IN + 0.02
    S = EO.divergence_rows(op)
    rows = []
    for nu in NUS:
        ue = remove_rigid(exact(P, nu).ravel(), P, m3)
        for kind in ("plain", "anp"):
            K = EO.plain_stiffness(op, np.ones(len(op.vol)), nu) if kind == "plain" else EO.anp_stiffness(op, np.ones(len(op.vol)), nu, S)
            u = amg_solve(K, f, P, m3)[0] if N >= AMG_FROM_N else E.full_solve(K, f, P, m3)
            e = u - ue
            ur_num = (u.reshape(-1, 3) * P).sum(1) / r
            ur_ex = (ue.reshape(-1, 3) * P).sum(1) / r
            row = dict(N=N, seed=seed, nu=nu, op=kind, n_tets=len(op.vol), sliver=op.info["sliver_frac"],
                       relL2=float(np.sqrt((m3 * e ** 2).sum() / (m3 * ue ** 2).sum())),
                       inner_ur_ratio=float(ur_num[inner].mean() / ur_ex[inner].mean()))
            if nu in (0.3, 0.49) and N == 8192:
                lam = np.sort(sla.eigsh(K.tocsc(), k=36, M=sp.diags(m3).tocsc(), sigma=-1e-6)[0])
                row.update(n_zero=int((lam < 1e-8 * lam[6]).sum()),
                           **{f"lam{i + 1}": float(lam[i]) for i in range(6, 36)})
            rows.append(row)
    for rw in rows:
        rw["t_s"] = time.time() - t0
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    parts = os.path.join(OUT, "parts")                                  # one csv per finished task -> resumable
    os.makedirs(parts, exist_ok=True)
    part = lambda t: os.path.join(parts, f"N{t[0]}_s{t[1]}.csv")
    tasks = [(N, s) for N in (8192, 16384, 32768) for s in (0, 1)]
    todo = [t for t in tasks if not os.path.exists(part(t))]
    print(f"{len(tasks) - len(todo)} tasks already done, {len(todo)} to run", flush=True)
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(task, todo):
            pd.DataFrame(r).to_csv(part((r[0]["N"], r[0]["seed"])), index=False)
            print(r[0]["N"], r[0]["seed"], f"{r[0]['t_s']:.0f}s", flush=True)
    df = pd.concat([pd.read_csv(part(t)) for t in tasks])
    df.to_csv(os.path.join(OUT, "lame.csv"), index=False)
    print(df.pivot_table(index=["N"], columns=["op", "nu"], values="relL2").to_string(float_format=lambda v: f"{v:.3g}"))
    print(df.pivot_table(index=["N"], columns=["op", "nu"], values="inner_ur_ratio").to_string(float_format=lambda v: f"{v:.3g}"))
    ev = df[df.N == 8192].dropna(subset=["lam7"])
    print(ev.groupby(["op", "nu"])[["n_zero", "lam7", "lam8", "lam12", "lam20", "lam36"]].mean().to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()

"""C (b): volume heat diffusion on the real shapes: Galerkin (G) with three bases vs fine FEM -> real_shapes/b_heat.csv

Problem  u_t = Laplace u in the solid, Neumann boundary, 3 Gaussian ICs (sigma 0.25, random centres), 6 times
         tau = t lambda_1 in {0.01, 0.03, 0.1, 0.3, 1, 3} (lambda_1 of the cloud operator; a shift-invert solve on
         the fine mesh only for lambda_1 cost minutes per shape).
Truth    fine P1 FEM: fTetWild mesh with edge_length_r = FINE_R and no coarsening, lumped mass, shift-invert block Krylov in time (fine_heat);
         interpolated (barycentric) to the point cloud.  Everything in the NEO-normalised frame of the cloud; the
         cloud's normalisation (centre, scale) is recovered by replaying c_volumize's seeded sampling.
Models on the point cloud (quasi and random, N = 8192, cleaned operators):
  PC-discrete   full Delaunay-P1 operator of the cloud, exact in time (shift-invert Krylov, discretisation floor)
  G-<basis>     u(t) = Q exp(-t Lambda) Q^T M u0 with the 256 Rayleigh-Ritz pairs of basis Q:
                neo256 (raw NEO), neoinv2 (NEO + 2 inverse-iteration steps), arpack256 (exact)
  (The learned stage-1 L model (g_theta) exists only for surfaces; no volume L model has been trained.)
Error: relative L2 (cloud lumped mass) against the interpolated fine FEM solution.

    python stage3/c_heat.py [--procs 6]
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

import e1_basis as E
import c_volumize as CV
import c_lumps_data as CL

sys.path.insert(0, E.S2)
import opcheck_span_fem as SF  # noqa: E402
from volop import p1_elements  # noqa: E402

RS = os.path.join(E.ROOT, "real_shapes")
FINE_R = 0.01
TAUS = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
N_IC = 3


def fit_frame(sdf, Y, is_s, init=None):
    """Recover the cloud normalisation Y = (P - c) s from the surface samples, which lie on sdf(P) = 0 (they were
    Newton-projected to |sdf| < 1e-6): least squares over (c, log s), started from `init` = (c0, s0) given by the
    volume ratio and the volume centroids (a start at c = 0, s = 1 fell into wrong local minima on near-symmetric
    shapes, e.g. thingi53405)."""
    from scipy.optimize import least_squares
    Ys = Y[is_s]
    f = lambda q: sdf(Ys / np.exp(q[3]) + q[:3])
    x0 = np.r_[np.zeros(3), 0.0] if init is None else np.r_[init[0], np.log(init[1])]
    r = least_squares(f, x0, x_scale=np.r_[0.1, 0.1, 0.1, 0.1], xtol=1e-12, ftol=1e-12, max_nfev=200)
    return r.x[:3], float(np.exp(r.x[3])), float(np.abs(r.fun).max())


def fine_mesh(name, kind_file, path):
    import wildmeshing as wm
    out = os.path.join(RS, "tet", f"{name}_fine.npz")
    if os.path.exists(out):
        z = np.load(out)
        return z["V"], z["T"]
    V, F = CV.load_surface(kind_file, path)
    c = 0.5 * (V.min(0) + V.max(0))
    V = (V - c) / np.abs(V - c).max()                                # same frame as the coarse tet mesh (SDF)
    tw = wm.Tetrahedralizer(stop_quality=10, epsilon=1e-3, edge_length_r=FINE_R, max_threads=4, coarsen=False)
    tw.set_mesh(V, F)
    tw.tetrahedralize()
    Vt, Tt = tw.get_tet_mesh()[:2]
    Vt, Tt = np.asarray(Vt, np.float64), np.asarray(Tt, np.int64)
    used, Tt = np.unique(Tt, return_inverse=True)
    Vt, Tt = Vt[used], Tt.reshape(-1, 4)
    np.savez(out, V=Vt, T=Tt)
    return Vt, Tt


def p1(P, T):
    """P1 stiffness / lumped mass; nodes left without any tet (degenerate tets dropped by p1_elements) are removed,
    returning the kept node coordinates too."""
    T, G, vol = p1_elements(P, T)
    used, T = np.unique(T, return_inverse=True)
    T = T.reshape(-1, 4)
    P = P[used]
    n = len(P)
    Kl = vol[:, None, None] * np.einsum("tki,tkj->tij", G, G)
    K = sp.coo_matrix((Kl.ravel(), (np.repeat(T, 4, 1).ravel(), np.tile(T, (1, 4)).ravel())), shape=(n, n)).tocsr()
    m = np.zeros(n)
    np.add.at(m, T.ravel(), np.repeat(vol / 4, 4))
    return K, m, T, P


def fine_heat(K, m, U0, t, lam1, steps=(40, 60)):
    """exp(-t M^{-1} K) U0 on the fine mesh by shift-invert block Krylov: one sparse LU of K + s M (s = lam1), an
    M-orthonormal block Krylov basis of (K + s M)^{-1} M started at U0, exact exponential of the projected
    operator (Rayleigh-Ritz).  expm_multiply is unusable here: tiny lumped masses of the uncoarsened fTetWild mesh
    make ||M^{-1} K|| huge.  Convergence check: basis sizes steps[0] and steps[1] (returned relative change)."""
    lu = sla.splu((K + lam1 * sp.diags(m)).tocsc())
    def basis(nb):
        V = [U0 / np.sqrt((m[:, None] * U0 ** 2).sum(0))]
        Q = np.linalg.qr(np.sqrt(m)[:, None] * V[0])[0] / np.sqrt(m)[:, None]
        blocks = [Q]
        for _ in range(nb - 1):
            W = lu.solve(m[:, None] * blocks[-1])
            for _ in range(2):                                           # block Gram-Schmidt (twice), M-inner product
                for B in blocks:
                    W = W - B @ (B.T @ (m[:, None] * W))
            Q = np.linalg.qr(np.sqrt(m)[:, None] * W)[0] / np.sqrt(m)[:, None]
            blocks.append(Q)
        return np.hstack(blocks)
    out = []
    for nb in steps:
        Qk = basis(nb)
        out.append(galerkin(Qk, K, m, U0, t))
    rel = float(np.sqrt(((out[1] - out[0]) ** 2 * m[None, :, None]).sum() / (out[1] ** 2 * m[None, :, None]).sum()))
    return out[1], rel


def galerkin(Q, K, m, u0, t):
    A, B = Q.T @ (K @ Q), (Q * m[:, None]).T @ Q
    lam, W = la.eigh(0.5 * (A + A.T), 0.5 * (B + B.T))
    Qr = Q @ W
    c0 = Qr.T @ (m[:, None] * u0)
    return np.stack([Qr @ (np.exp(-tj * lam)[:, None] * c0) for tj in t])


def task(args):
    name, kind_file, path = args
    t0 = time.time()
    shape_sdf = CV.make_sdf(os.path.join(RS, "tet", f"{name}.npz"))
    Pf, Tf = fine_mesh(name, kind_file, path)
    t_fine_mesh = time.time() - t0
    rows = []
    for kind in ("quasi", "random"):
        cache = os.path.join(RS, "cache", f"{name}_{kind}_8192.npz")
        z = np.load(cache)
        Y = z["Y"]
        from e1_v2 import surface_weights
        is_s = surface_weights(Y, E.Op(z).tets) > 0                    # boundary nodes of the cloud operator
        # boundary nodes include sampled surface points (on sdf = 0); interior nodes carved to the boundary would
        # bias the fit, so keep the nodes with the smallest |sdf| under the initial guess (80 %)
        s0 = np.abs(shape_sdf(Y[is_s]))
        keep = np.zeros(len(Y), bool); keep[np.where(is_s)[0][s0 <= np.quantile(s0, 0.8)]] = True
        op0 = E.Op(z)
        tz = np.load(os.path.join(RS, "tet", f"{name}.npz"))
        Vt, Tt = tz["V"], tz["T"]
        vt = np.abs(np.einsum("ij,ij->i", np.cross(Vt[Tt[:, 1]] - Vt[Tt[:, 0]], Vt[Tt[:, 2]] - Vt[Tt[:, 0]]), Vt[Tt[:, 3]] - Vt[Tt[:, 0]])) / 6
        C_true = (vt[:, None] * Vt[Tt].mean(1)).sum(0) / vt.sum()
        C_Y = (op0.vol[:, None] * Y[op0.tets].mean(1)).sum(0) / op0.vol.sum()
        s0 = (op0.vol.sum() / vt.sum()) ** (1 / 3)
        c, s, mis = fit_frame(shape_sdf, Y, keep, init=(C_true - C_Y / s0, s0))
        Pn = (Pf - c) * s                                              # fine mesh in the cloud's frame
        Kf, mf, Tn, Pn = p1(Pn, Tf)
        t = TAUS / float(z["lam_s"][1])
        rng = np.random.default_rng([CL.SEED + 1, sum(map(ord, name)), kind == "random"])
        ctr = Y[rng.choice(len(Y), N_IC, replace=False)]
        ic = lambda X: np.stack([np.exp(-((X - cc) ** 2).sum(1) / (2 * 0.25 ** 2)) for cc in ctr], 1)
        tf = time.time()
        Uf, kry_err = fine_heat(Kf, mf, ic(Pn), t, float(z["lam_s"][1]))  # (T, n_fine, I)
        t_fem = time.time() - tf
        I, _ = SF.interp(Pn, Tn, Y)
        Utrue = np.stack([I @ Uf[j] for j in range(len(t))])
        op = E.Op(z)
        K, m = op.K.tocsc(), op.m
        u0 = ic(Y)
        tp = time.time()
        Upc, pc_err = fine_heat(K, m, u0, t, float(z["lam_s"][1]))    # same shift-invert Krylov (exact in time)
        t_pc = time.time() - tp
        preds = {"PC-discrete": (Upc, t_pc)}
        bases = {"neo256": z["Pn"].astype(np.float64), "neoinv2": CL.refine2(z["Pn"].astype(np.float64), K, m, float(z["lam_s"][1])),
                 "arpack256": z["Ps"]}
        for b, Q in bases.items():
            tg = time.time()
            preds[f"G-{b}"] = (galerkin(Q, K, m, u0, t), time.time() - tg)
        for model, (U, dt) in preds.items():
            for j, tau in enumerate(TAUS):
                e = U[j] - Utrue[j]
                rows.append(dict(shape=name, kind=kind, model=model, tau=tau, n_fine=len(Pf), n_fine_tets=len(Tf),
                                 relL2=float(np.sqrt((m[:, None] * e ** 2).sum() / (m[:, None] * Utrue[j] ** 2).sum())),
                                 t_model_s=dt, t_fine_fem_s=t_fem, t_fine_mesh_s=t_fine_mesh, frame_fit_res=mis, pc_krylov_change=pc_err,
                                 krylov_rel_change=kry_err))
    return rows


def safe_task(args):
    try:
        return task(args)
    except Exception as e:
        return f"{args[0]} FAILED: {e!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    shapes = CV.shape_list()
    have = {os.path.basename(p)[:-len("_quasi_8192.npz")] for p in glob.glob(os.path.join(RS, "cache", "*_quasi_8192.npz"))}
    tasks = [(n, k, p) for n, (k, p) in shapes.items() if n in have and (not a.only or n in a.only.split(","))]
    import pandas as pd
    parts = os.path.join(RS, "b_heat_parts")                            # one csv per shape -> resumable
    os.makedirs(parts, exist_ok=True)
    todo = [t for t in tasks if not os.path.exists(os.path.join(parts, f"{t[0]}.csv"))]
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(safe_task, todo):
            if isinstance(r, str):
                print(r, flush=True)
                continue
            pd.DataFrame(r).to_csv(os.path.join(parts, f"{r[0]['shape']}.csv"), index=False)
            print(r[0]["shape"], "done", flush=True)
    df = pd.concat([pd.read_csv(os.path.join(parts, f"{t[0]}.csv")) for t in tasks
                    if os.path.exists(os.path.join(parts, f"{t[0]}.csv"))])
    df.to_csv(os.path.join(RS, "b_heat.csv"), index=False)
    print(df.pivot_table(index=["shape", "kind"], columns="model", values="relL2", aggfunc="mean").to_string(float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()

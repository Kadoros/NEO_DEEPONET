"""Part 1: procedural dataset.

Per shape (data/shape_XXX.npz):
  V, F                  mesh (area-normalised, rotated); points = V (connectivity is never given to models)
  m_mesh                lumped mesh mass;  lam_mesh, Phi_mesh: first K_MODES mesh eigenpairs (oracle only)
  U0 (N_IC, N), U (8, N_IC, N), t (8)   ground truth u(t) = expm_multiply(-t M^{-1} L_cot, u0)
  M_pc, c_pc, lam_ex, Phi_ex            point-cloud Laplacian (n_neighbors=20), calibrated c·L, ARPACK 96 modes

    python stage1/make_dataset.py        (env neo; CPU only)
"""
import common  # noqa: F401  (sets cache dirs)

import json
import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from robust_laplacian import point_cloud_laplacian

from common import (AREA, DATA, FAMILIES, K_MODES, N_IC, N_SHAPES, N_VERT, PC_NEIGHBORS, SEED, SPLIT, TAUS)
from shapes import angles, cotan_laplacian, make_shape


def initial_conditions(V, rng):
    """N_IC fields, each a sum of 1-3 Gaussians (Euclidean distance)."""
    U0, spec = [], []
    for _ in range(N_IC):
        ng = rng.integers(1, 4)
        c_idx = rng.choice(len(V), ng, replace=False)
        sig = rng.uniform(0.15, 0.5, ng)
        amp = rng.uniform(0.5, 1.5, ng)
        u = sum(a * np.exp(-((V - V[i]) ** 2).sum(1) / (2 * s ** 2)) for i, s, a in zip(c_idx, sig, amp))
        U0.append(u)
        spec.append(dict(centers=c_idx.tolist(), sigma=sig.tolist(), amp=amp.tolist()))
    return np.stack(U0), spec


def build(sid):
    path = os.path.join(DATA, f"shape_{sid:03d}.npz")
    if os.path.exists(path):
        return None
    fam = FAMILIES[sid * len(FAMILIES) // N_SHAPES]
    rng = np.random.default_rng([SEED, sid])
    for attempt in range(10):
        V, F, ok, chi, params = make_shape(fam, N_VERT, AREA, rng)
        if ok:
            break
    else:
        raise RuntimeError(f"shape {sid}: no valid mesh after 10 attempts")

    # ground truth operator
    L, m = cotan_laplacian(V, F)
    lam_mesh, Phi_mesh = sla.eigsh(L.tocsc(), k=K_MODES + 1, M=sp.diags(m).tocsc(), sigma=-1e-8)
    o = np.argsort(lam_mesh)
    lam_mesh, Phi_mesh = lam_mesh[o], Phi_mesh[:, o]
    lam1 = lam_mesh[1]

    U0, ic_spec = initial_conditions(V, rng)
    A = sp.diags(1 / m) @ L
    t = np.array(TAUS) / lam1
    t0 = time.perf_counter()
    U = np.stack([sla.expm_multiply(-tj * A, U0.T).T for tj in t])       # (8, N_IC, N)
    t_gt = time.perf_counter() - t0
    mass_drift = np.abs(U @ m - U0 @ m).max() / np.abs(U0 @ m).max()

    # point-cloud basis (no connectivity): robust Laplacian, surface calibration, ARPACK
    t0 = time.perf_counter()
    Lp, Mp = point_cloud_laplacian(V, n_neighbors=PC_NEIGHBORS)
    Lp, Mp = Lp.tocsc(), Mp.tocsc()
    c = 2 * Mp.diagonal().sum() / sum(V[:, i] @ (Lp @ V[:, i]) for i in range(3))
    lam_ex, Phi_ex = sla.eigsh(c * Lp, k=K_MODES, M=Mp, sigma=-1e-8)
    t_pc = time.perf_counter() - t0
    o = np.argsort(lam_ex)
    lam_ex, Phi_ex = lam_ex[o], Phi_ex[:, o]

    ang = angles(V, F).min(1)
    np.savez(path, V=V.astype(np.float32), F=F.astype(np.int32), m_mesh=m, lam_mesh=lam_mesh[:K_MODES],
             Phi_mesh=Phi_mesh[:, :K_MODES].astype(np.float32), U0=U0.astype(np.float32),
             U=U.astype(np.float32), t=t, lam1=lam1, M_pc=Mp.diagonal(), c_pc=c,
             lam_ex=lam_ex, Phi_ex=Phi_ex.astype(np.float32))
    return dict(sid=sid, family=fam, attempts=attempt + 1, lam1=lam1, c_pc=c,
                pc_mass_over_area=Mp.diagonal().sum() / AREA,
                pc_lam_bias_1to10=float(np.mean(lam_ex[1:11] / lam_mesh[1:11] - 1)),
                pc_lam_bias_80to95=float(np.mean(lam_ex[80:96] / lam_mesh[80:96] - 1)),
                min_angle=float(ang.min()), gt_mass_drift=mass_drift, t_gt_s=t_gt, t_pc_basis_s=t_pc,
                params=json.dumps({k: float(v) for k, v in params.items()}), ics=json.dumps(ic_spec))


def main():
    os.makedirs(DATA, exist_ok=True)
    t0 = time.time()
    with Pool(24) as p:
        rows = [r for r in p.imap_unordered(build, range(N_SHAPES)) if r is not None]
    if rows:
        new = pd.DataFrame(rows)
        idx_path = os.path.join(DATA, "index.csv")
        if os.path.exists(idx_path):
            new = pd.concat([pd.read_csv(idx_path), new])
        new.sort_values("sid").to_csv(idx_path, index=False)

    # split by shape, stratified by family: 160/20/20 each
    rng = np.random.default_rng([SEED, 999])
    split = {}
    per = N_SHAPES // len(FAMILIES)
    for f in range(len(FAMILIES)):
        ids = rng.permutation(np.arange(f * per, (f + 1) * per))
        a, b = SPLIT[0], SPLIT[0] + SPLIT[1]
        split.update({int(i): "train" for i in ids[:a]} | {int(i): "val" for i in ids[a:b]}
                     | {int(i): "test" for i in ids[b:]})
    pd.DataFrame({"sid": list(split), "split": list(split.values())}).sort_values("sid").to_csv(
        os.path.join(DATA, "split.csv"), index=False)
    print(f"dataset done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

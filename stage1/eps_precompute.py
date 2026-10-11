"""ε-path inputs (CPU) -> data/eps_XXX.npz, and the full point-cloud discrete solution for
val/test shapes -> data/pcsol_XXX.npz.

  L      calibrated point-cloud Laplacian c·L_pc (n_neighbors=20) — the same operator the NEO
         Rayleigh–Ritz used;  M = its lumped mass M_pc
  Y      the 96 NEO Ritz vectors used by L-neo (M-orthonormal), so u0 = Y Yᵀ M u0 + r exactly
  r      = u0 − Y Yᵀ M u0                         (N_IC, N)
  r_p(t) = (M + t L)^{-1} M r                     (8, N_IC, N)   one sparse factorisation per t
  u_pc(t)= expm_multiply(−t M^{-1} L, u0)         (8, N_IC, N)   val/test only (reference baseline)

    python stage1/eps_precompute.py      (env neo)
"""
import common  # noqa: F401

import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from robust_laplacian import point_cloud_laplacian

from common import DATA, N_SHAPES, PC_NEIGHBORS


def work(args):
    sid, need_pcsol = args
    d = np.load(os.path.join(DATA, f"shape_{sid:03d}.npz"))
    e = np.load(os.path.join(DATA, f"neo_{sid:03d}.npz"))
    V = d["V"].astype(np.float64)
    L, _ = point_cloud_laplacian(V, n_neighbors=PC_NEIGHBORS)
    L = (d["c_pc"] * L).tocsc()
    m = d["M_pc"]
    M = sp.diags(m).tocsc()
    U0 = d["U0"].astype(np.float64)                          # (I, N)
    Y = e["Phi_neo"].astype(np.float64)                      # (N, 96)
    r = U0 - (Y @ (Y.T @ (m[:, None] * U0.T))).T

    t0 = time.perf_counter()
    rp = np.stack([sla.splu((M + tj * L).tocsc()).solve((m[:, None] * r.T)).T for tj in d["t"]])
    t_rp = time.perf_counter() - t0
    np.savez(os.path.join(DATA, f"eps_{sid:03d}.npz"), r=r.astype(np.float32), rp=rp.astype(np.float32))

    row = dict(sid=sid, t_rp_s=t_rp, r_rel=float(np.sqrt((m * r ** 2).sum(1) / (m * U0 ** 2).sum(1)).mean()))
    if need_pcsol:
        A = (sp.diags(1 / m) @ L).tocsr()
        t0 = time.perf_counter()
        u_pc = np.stack([sla.expm_multiply(-tj * A, U0.T).T for tj in d["t"]])
        row["t_pcsol_s"] = time.perf_counter() - t0
        np.savez(os.path.join(DATA, f"pcsol_{sid:03d}.npz"), u_pc=u_pc.astype(np.float32))
    return row


def main():
    split = pd.read_csv(os.path.join(DATA, "split.csv")).set_index("sid")["split"]
    tasks = [(sid, split[sid] in ("val", "test")) for sid in range(N_SHAPES)]
    tasks.sort(key=lambda a: not a[1])                        # expensive ones first
    t0 = time.time()
    with Pool(24) as p:
        rows = list(p.imap_unordered(work, tasks))
    pd.DataFrame(rows).sort_values("sid").to_csv(os.path.join(DATA, "eps_index.csv"), index=False)
    print(f"eps precompute done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

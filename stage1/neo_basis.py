"""Part 2 prerequisite: NEO (Ours-Large) basis for every shape -> data/neo_XXX.npz

  input      normalize_pc(V) (centre, max|coord| = 1, as in NEO's infer.py) and its
             point-cloud mass (n_neighbors=20, the NEO training setting)
  F          raw network output model.model(x, x, mass)  (N × 256, before M-orthonormalisation)
  S          qr_orthogonalization(F, mass)  (NEO span)
  Φ_neo, λ   Rayleigh–Ritz of the *same* calibrated point-cloud L as G-exact (dense, float64),
             lowest K_MODES Ritz pairs
  F_hat, z   F / s_F with s_F = sqrt(Σ m_i |F_i|² / (Σ m_i · 256)) (per shape);  z = Σ m_i F_hat_i / Σ m_i

    CUDA_VISIBLE_DEVICES=<free gpu> python stage1/neo_basis.py      (env neo)
"""
import common  # noqa: F401

import os
import time

import numpy as np
import pandas as pd
import scipy.linalg as la
import torch
from robust_laplacian import point_cloud_laplacian

from common import DATA, K_MODES, N_SHAPES, NEO_CKPT, PC_NEIGHBORS
from g2pt.data.transforms import normalize_pc
from g2pt.training.pretrain import PretrainTraining
from g2pt.utils.ortho_operations import qr_orthogonalization


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "set CUDA_VISIBLE_DEVICES to one free GPU"
    model = PretrainTraining.load_from_checkpoint(NEO_CKPT, weights_only=False, strict=False)
    model = model.to(device="cuda", dtype=torch.float32).eval()
    rows = []
    for sid in range(N_SHAPES):
        out = os.path.join(DATA, f"neo_{sid:03d}.npz")
        d = np.load(os.path.join(DATA, f"shape_{sid:03d}.npz"))
        V = d["V"].astype(np.float64)

        t0 = time.perf_counter()
        pc = normalize_pc(V, 0)
        _, Mn = point_cloud_laplacian(pc, n_neighbors=PC_NEIGHBORS)
        mass = torch.tensor(Mn.diagonal(), dtype=torch.float32, device="cuda").view(1, -1, 1)
        x = torch.tensor(pc, dtype=torch.float32, device="cuda").view(1, -1, 3)
        t_in = time.perf_counter() - t0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            F = model.model(x, x, mass)
            S = qr_orthogonalization(F, mass)
        torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t0

        # Rayleigh–Ritz with the same calibrated L as G-exact (area-normalised coordinates)
        t0 = time.perf_counter()
        Lp, _ = point_cloud_laplacian(V, n_neighbors=PC_NEIGHBORS)
        Lp = d["c_pc"] * Lp
        m_pc = d["M_pc"]                    # same M as G-exact (Φ_ex is M_pc-orthonormal)
        S = S[0].double().cpu().numpy()
        lam, Y = la.eigh(S.T @ (Lp @ S), (S * m_pc[:, None]).T @ S)
        lam, Phi = lam[:K_MODES], S @ Y[:, :K_MODES]
        t_rr = time.perf_counter() - t0

        F = F[0].double().cpu().numpy()
        m = m_pc
        s_F = np.sqrt((m[:, None] * F ** 2).sum() / (m.sum() * F.shape[1]))
        F_hat = F / s_F
        z = (m[:, None] * F_hat).sum(0) / m.sum()

        Phi_ex, lam_ex = d["Phi_ex"].astype(np.float64), d["lam_ex"]
        Q = S @ Y                           # all 256 Ritz vectors: M_pc-orthonormal span of NEO
        span_loss = 1 - ((Q.T @ (m_pc[:, None] * Phi_ex)) ** 2).sum() / K_MODES
        np.savez(out, F_hat=F_hat.astype(np.float16), z=z.astype(np.float32), lam_neo=lam,
                 Phi_neo=Phi.astype(np.float32))
        rows.append(dict(sid=sid, span_loss_vs_ex=span_loss,
                         lam_relerr_1to95=float(np.mean(lam[1:] / lam_ex[1:] - 1)),
                         lam_relerr_vs_mesh_1to95=float(np.mean(lam[1:] / d["lam_mesh"][1:] - 1)),
                         t_input_s=t_in, t_forward_s=t_fwd, t_rr_s=t_rr))
        if sid % 50 == 0:
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "neo_index.csv"), index=False)


if __name__ == "__main__":
    main()

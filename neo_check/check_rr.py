"""Diagnostic: is NEO's ~0.77% flat eigenvalue error a property of the learned
subspace, or of the Rayleigh-Ritz step in infer.py (which keeps only diag(SᵀMS))?

Same model call / L, M as exp/pretrain/infer.py; compares, against ARPACK:
  (a) repo RR:  solve_gev_from_subspace_cuda (diag(SᵀMS), float32)
  (b) full RR:  dense SᵀLS, SᵀMS, scipy.linalg.eigh, float64

    CUDA_VISIBLE_DEVICES=<free gpu> conda run -n neo python neo_check/check_rr.py
"""
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import torch
from robust_laplacian import point_cloud_laplacian

from g2pt.data.transforms import normalize_pc
from g2pt.training.pretrain import PretrainTraining
from g2pt.utils.gev import solve_gev_from_subspace_cuda, solve_gev_ground_truth

HERE = Path(__file__).resolve().parent
K = 96

model = PretrainTraining.load_from_checkpoint(HERE / "ckpts/Ours-Large.ckpt", weights_only=False, strict=False)
model = model.to(device="cuda", dtype=torch.float32).eval()

lines = []
for case in ["sphere", "suzanne", "torus"]:
    pc = normalize_pc(np.load(HERE / f"samples/{case}/sample_points.npy"), 0)
    L, M = point_cloud_laplacian(pc)
    mass = M.diagonal().astype(np.float32)
    L = sp.csr_matrix(L).astype(np.float64)
    M = sp.csr_matrix(M).astype(np.float64)
    pc_t = torch.tensor(pc, dtype=torch.float32, device="cuda").view(1, -1, 3)
    m_t = torch.tensor(mass, dtype=torch.float32, device="cuda").view(1, -1, 1)
    with torch.no_grad():
        S_t = model.forward(pc_t, m_t, False, return_time=False)[0][0].float()

    lam_gt, _ = solve_gev_ground_truth(L=L, M=M, k=K)
    lam_gt = np.sort(np.asarray(lam_gt))

    lam_a, _ = solve_gev_from_subspace_cuda(S_t, pc_t[0], m_t[0], k=model.targ_dim_model, L=L, M=M, precision="32")
    lam_a = np.sort(lam_a.cpu().numpy().astype(np.float64))[:K]

    S = S_t.cpu().numpy().astype(np.float64)
    Ms = S.T @ (M @ S)
    lam_b = sla.eigh(S.T @ (L @ S), Ms, eigvals_only=True)[:K]

    off = np.abs(Ms - np.diag(np.diag(Ms))).max()
    for name, lam in [("repo RR (diag SᵀMS, fp32)", lam_a), ("full RR (dense SᵀMS, fp64)", lam_b)]:
        r = lam[1:] / lam_gt[1:] - 1
        lines.append(f"{case:8s} {name:28s} signed relerr i=1..95: mean {r.mean():+.2e}  "
                     f"min {r.min():+.2e}  max {r.max():+.2e}")
    lines.append(f"{case:8s} SᵀMS: diag mean {np.diag(Ms).mean():.4f}, max |offdiag| {off:.2e}")

out = "\n".join(lines)
print(out)
(HERE / "results/check_rr.txt").write_text(out + "\n")

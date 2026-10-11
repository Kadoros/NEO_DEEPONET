"""C timing table: pipeline stage costs by shape size -> real_shapes/timing.csv

Per real shape and cloud (quasi N = 8192 and 16384; cleaned operators), sequential and alone on the machine:
  neo_gpu          NEO forward + M-orthonormalisation on the GPU (warm; median of 3)
  neo_rr           scalar Rayleigh-Ritz of the 256 NEO vectors (CPU)
  scal_lu / scal_step   scalar inverse iteration: sparse LU of K + s M, one block step (256 columns)
  el_assemble      ANP elastic stiffness assembly (nu = 0.3)
  el_lu / el_step  elastic inverse iteration: sparse LU of the shifted ANP K, one block step (~980 columns)
  el_orth_rr       orthonormalisation + Rayleigh-Ritz of the ~980-dim elastic basis
  static_corr      one static-correction vector = one full elastic solve (AMG-CG, smooth two-spot load)
  red_cpu          reduced solve for a new coefficient field E(x): Q^T K(E) Q (sparse x dense, CPU) + dense solve
  red_gpu          the same with the element-tensor assembly A = sum_e E_e vol_e (BQ)_e^T D (BQ)_e on the GPU
  full_amg_el      full-order elastic solve for a new E(x), AMG-CG (the fair baseline for red_*)
  full_amg_heat    full-order scalar solve (K + M/dt) u = b, AMG-CG (one implicit heat step)
Columns also give N and the number of tets.  Break-even counts: (basis cost) / (full solve - reduced solve).

    CUDA_VISIBLE_DEVICES=<gpu> OMP_NUM_THREADS=4 python stage3/c_timing.py
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import pyamg
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

import e1_basis as E
import e1_v2 as V
import elastic_ops as EO
from amg_solve import amg_solve
from e1_rot import rotation_fields

RS = os.path.join(E.ROOT, "real_shapes")


def tm(fn, rep=1):
    ts, out = [], None
    for _ in range(rep):
        t = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t)
    return out, float(np.median(ts))


def neo_gpu(Y, m):
    import torch
    from g2pt.training.pretrain_tet import PretrainTetTraining
    from g2pt.utils.ortho_operations import qr_orthogonalization
    global _NEO
    if "_NEO" not in globals():
        _NEO = PretrainTetTraining.load_from_checkpoint(E.CKPT, weights_only=False).model.to("cuda").eval()
    x = torch.tensor(Y, dtype=torch.float32, device="cuda")[None]
    mass = torch.tensor(m, dtype=torch.float32, device="cuda").view(1, -1, 1)

    def run():
        with torch.no_grad():
            S = qr_orthogonalization(_NEO(x, x, mass), mass)[0]
        torch.cuda.synchronize()
        return S.double().cpu().numpy()
    run()
    return tm(run, 3)


def red_gpu_time(op, Q, E_e, nu, f, m3):
    import torch
    B = torch.tensor(EO.b_matrices(op.grads), dtype=torch.float32, device="cuda")                  # (T, 6, 12)
    dof = torch.tensor(EO.dofs(op.tets), device="cuda")
    Qt = torch.tensor(Q, dtype=torch.float32, device="cuda")
    BQ = torch.einsum("tij,tjk->tik", B, Qt[dof])                                                  # (T, 6, m)
    D = torch.tensor(EO.d_full(nu), dtype=torch.float32, device="cuda")
    w = torch.tensor(E_e * op.vol, dtype=torch.float32, device="cuda")
    b = torch.tensor(Q.T @ f, dtype=torch.float64, device="cuda")

    X = BQ.reshape(-1, BQ.shape[-1])                                                              # (T*6, m)

    def run():
        DX = torch.einsum("ij,tjn->tin", D, BQ).reshape(-1, BQ.shape[-1])
        A = ((w.repeat_interleave(6))[:, None] * X).T @ DX                                          # one GEMM
        A = A.double()
        c = torch.linalg.solve(A + 1e-10 * torch.eye(A.shape[0], device="cuda", dtype=torch.float64), b)
        torch.cuda.synchronize()
        return c
    run()
    return tm(run, 3)[1]


def main():
    rows = []
    only = os.environ.get("TIMING_ONLY", "")
    for p in sorted(glob.glob(os.path.join(RS, "cache", "*_quasi_*.npz"))):
        if only and os.path.basename(p)[:-4] not in only.split(","):
            continue
        name, kind, N = os.path.basename(p)[:-4].rsplit("_", 2)
        z = np.load(p)
        op = E.Op(z)
        Y, m = z["Y"], op.m
        K = op.K.tocsc()
        m3 = np.repeat(m, 3)
        r = dict(shape=name, N=len(Y), n_tets=len(op.tets))
        S, r["neo_gpu"] = neo_gpu(Y, m)
        _, r["neo_rr"] = tm(lambda: la.eigh(*(lambda A, B: (0.5 * (A + A.T), 0.5 * (B + B.T)))(S.T @ (K @ S), (S * m[:, None]).T @ S)))
        lu, r["scal_lu"] = tm(lambda: sla.splu((K + 1e-3 * float(z["lam_s"][1]) * sp.diags(m)).tocsc()))
        _, r["scal_step"] = tm(lambda: lu.solve(m[:, None] * S))
        nu = 0.3
        Sd = EO.divergence_rows(op)
        Kel, r["el_assemble"] = tm(lambda: EO.anp_stiffness(op, np.ones(len(op.vol)), nu, Sd))
        raw = np.hstack([E.replicate(z["Pn"]), E.affine_fields(Y), rotation_fields(Y, z["Pn"], 64)])
        Q, r["el_orth_rr"] = tm(lambda: E.ritz(E.m_orthonormal(raw, m3), Kel)[1])
        lam7 = float(sla.eigsh(Kel.tocsc(), k=7, M=sp.diags(m3).tocsc(), sigma=-1e-6)[0].max())
        lue, r["el_lu"] = tm(lambda: sla.splu((Kel + 1e-3 * lam7 * sp.diags(m3)).tocsc()))
        _, r["el_step"] = tm(lambda: lue.solve(m3[:, None] * Q))
        rng = np.random.default_rng(0)
        f = E.pinch_load(Y, op.tets, rng)["smooth"]
        (u_hom, info, its), r["static_corr"] = tm(lambda: amg_solve(Kel, f, Y, m3))
        r["static_corr_iters"] = its
        Qc = E.m_orthonormal(np.hstack([Q, u_hom[:, None]]), m3)
        cen = Y[op.tets].mean(1)
        E_e = 1 + 4 / (1 + np.exp(-(0.15 - np.linalg.norm(cen - Y[len(Y) // 3], axis=1)) / 0.02))
        Kl = EO.anp_stiffness(op, E_e, nu, Sd)
        _, r["red_cpu"] = tm(lambda: E.reduced_solve(Qc, Kl, f, Y, m3))
        r["red_gpu"] = red_gpu_time(op, Qc, E_e, nu, f, m3)
        _, r["full_amg_el"] = tm(lambda: amg_solve(Kl, f, Y, m3))
        dt = 0.1 / float(z["lam_s"][1])
        A = (K + sp.diags(m) / dt).tocsr()
        b = m * np.random.default_rng(1).random(len(m))

        def heat():
            ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric")
            return sla.cg(A, b, M=ml.aspreconditioner(), rtol=1e-10)[0]
        _, r["full_amg_heat"] = tm(heat)
        r["basis_scalar"] = r["neo_gpu"] + r["neo_rr"] + r["scal_lu"] + 2 * r["scal_step"]
        r["basis_elastic_c"] = r["el_orth_rr"] + r["el_lu"] + 2 * r["el_step"] + r["static_corr"]
        for red in ("red_cpu", "red_gpu"):
            gain = r["full_amg_el"] - r[red]
            r[f"breakeven_{red}"] = float(np.ceil(r["basis_elastic_c"] / gain)) if gain > 0 else float("inf")
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RS, "timing.csv" if not only else "timing_smoke.csv"), index=False)
    print(df.sort_values("N").to_string(float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()

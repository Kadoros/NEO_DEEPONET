"""Gate v2 data (task B, hidden conductivity by a common rule) -> stage3/gate/data_v2/<split>/<name>.npz

Same shapes, ICs, times, NEO features and basis fields as gate v1 (gate/data); only the truth changes:
  kappa(x) = 1 + 3 sigmoid((d(x) - 0.12) / 0.03),  d = distance to the nearest boundary node (soft skin ~0.1 thick,
  4x stiffer core); per tet kappa_e = mean of its 4 nodal values.  The model is never given kappa.
  U = exact discrete solution exp(-t M^{-1} K(kappa)) u0 (shift-invert block Krylov).
Added fields: tets, grads (fp32), vol (element data for the reduced assembly), kappa (nodal truth, figures only),
  U_g_oracle_raw / U_g_oracle_ref (Galerkin with the TRUE kappa in the raw / refined NEO basis; reference ceilings),
  U_g_raw / U_g_ref (Galerkin with kappa = 1: the G baselines).
r, r_p stay the kappa = 1 versions (the model does not know kappa).

    python stage3/gate_v2_data.py [--procs 8]
"""
import argparse
import os
from multiprocessing import get_context

import numpy as np

import e1_basis as E
import gate_data as GD

OUT = os.path.join(E.ROOT, "gate", "data_v2")
REF = os.path.join(E.ROOT, "gate", "data_ref")


def kappa_rule(d):
    return 1 + 3 / (1 + np.exp(-(d - 0.12) / 0.03))


def galerkin(Phi, K, m, u0, t):
    import scipy.linalg as la
    A, B = Phi.T @ (K @ Phi), (Phi * m[:, None]).T @ Phi
    lam, W = la.eigh(0.5 * (A + A.T), 0.5 * (B + B.T))
    Q = Phi @ W
    c0 = Q.T @ (m[:, None] * u0.T)                                     # (K, I)
    return np.stack([(Q @ (np.exp(-tj * lam)[:, None] * c0)).T for tj in t])   # (T, I, N)


def task(args):
    split, name, src = args
    import c_heat as CH
    z = dict(np.load(os.path.join(GD.OUT, split, f"{name}.npz")))
    Y, tets, grads, vol, m = GD.load(src)
    kap = kappa_rule(z["d"].astype(np.float64))
    kap_e = kap[tets].mean(1)
    K1 = GD.stiffness(tets, grads, vol, len(Y)).tocsc()
    Kk = GD.stiffness(tets, grads, vol * kap_e, len(Y)).tocsc()
    t = z["t"].astype(np.float64)
    u0 = z["u0"].astype(np.float64)
    U, rel = CH.fine_heat(Kk, m, u0.T, t, float(z["lam1"]))
    z["U"] = U.transpose(0, 2, 1).astype(np.float32)
    Phi_raw = z["Phi"].astype(np.float64)
    out = dict(U_g_raw=galerkin(Phi_raw, K1, m, u0, t), U_g_oracle_raw=galerkin(Phi_raw, Kk, m, u0, t))
    rp = os.path.join(REF, split, f"{name}.npz")
    if os.path.exists(rp):
        Phi_ref = np.load(rp)["Phi"].astype(np.float64)
        out.update(U_g_ref=galerkin(Phi_ref, K1, m, u0, t), U_g_oracle_ref=galerkin(Phi_ref, Kk, m, u0, t))
    z.update({k: v.astype(np.float32) for k, v in out.items()})
    z.update(tets=tets.astype(np.int32), grads=grads.astype(np.float32), vol=vol.astype(np.float32),
             kappa=kap.astype(np.float32), krylov_change=rel)
    os.makedirs(os.path.join(OUT, split), exist_ok=True)
    np.savez(os.path.join(OUT, split, f"{name}.npz"), **z)
    return f"{split}/{name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    tasks = [x for x in GD.sources() if not os.path.exists(os.path.join(OUT, x[0], f"{x[1]}.npz"))]
    with get_context("spawn").Pool(a.procs) as pool:
        for msg in pool.imap_unordered(task, tasks):
            print(msg, flush=True)


if __name__ == "__main__":
    main()

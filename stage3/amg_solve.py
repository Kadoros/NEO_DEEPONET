"""Free-body elastic solve without the KKT saddle system: consistent right-hand side (rigid part of the load
removed with the same multiplier the KKT system would use), CG preconditioned by pyamg smoothed aggregation with
the 6 rigid modes as near-null space, then the M-rigid component removed.  Equals e1_basis.full_solve's KKT
solution up to the CG tolerance."""
import numpy as np
import pyamg
import scipy.sparse.linalg as sla

import e1_basis as E


def amg_solve(K, f, Y, m3, tol=1e-10, maxiter=2000):
    R = E.rigid_modes(Y)
    MR = m3[:, None] * R
    g = f - MR @ np.linalg.solve(R.T @ MR, R.T @ f)                   # consistent RHS (KKT multiplier removed)
    ml = pyamg.smoothed_aggregation_solver(K.tocsr(), B=R, symmetry="symmetric", max_coarse=500)
    it = [0]
    u, info = sla.cg(K, g, M=ml.aspreconditioner(cycle="V"), rtol=tol, maxiter=maxiter,
                     callback=lambda _: it.__setitem__(0, it[0] + 1))
    u = u - R @ np.linalg.solve(R.T @ MR, MR.T @ u)                    # M-orthogonal to the rigid modes
    return u, info, it[0]

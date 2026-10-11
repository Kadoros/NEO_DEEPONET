"""Full-order linear elasticity operators on the stage-2 Delaunay-P1 tets (DOF order 3*i + c).

  plain_stiffness   standard P1 (constant strain):  K = sum_e vol_e B_e^T D(E_e, nu) B_e
  anp_stiffness     selective nodal averaging (average-nodal-pressure tetrahedron, Bonet & Burton 1998 family):
                      deviatoric part per element      K_dev = sum_e vol_e B_e^T D_dev(E_e, nu) B_e
                      volumetric strain nodally averaged  theta_n = (1/V_n) sum_{e ni n} (vol_e / 4) div_e(u)
                      K_vol = W^T diag(V_n k_n) W,  W = M^{-1} S,  V_n = lumped mass,  k_n = nodal volume average of
                      the element bulk moduli k_e = E_e / (3 (1 - 2 nu))  (linear in E -> K(E) stays linear in E)
Both return a scipy CSR matrix; both are linear in the element moduli E_e.
"""
import numpy as np
import scipy.sparse as sp

VOIGT_M = np.array([1, 1, 1, 0, 0, 0], float)


def b_matrices(G):
    """G[t, d, a] = d phi_a / d x_d  ->  B (T, 6, 12), Voigt xx yy zz yz xz xy, engineering shear."""
    nT = G.shape[0]
    B = np.zeros((nT, 6, 12))
    for a in range(4):
        gx, gy, gz = G[:, 0, a], G[:, 1, a], G[:, 2, a]
        B[:, 0, 3 * a] = gx; B[:, 1, 3 * a + 1] = gy; B[:, 2, 3 * a + 2] = gz
        B[:, 3, 3 * a + 1] = gz; B[:, 3, 3 * a + 2] = gy
        B[:, 4, 3 * a] = gz; B[:, 4, 3 * a + 2] = gx
        B[:, 5, 3 * a] = gy; B[:, 5, 3 * a + 1] = gx
    return B


def lame(nu):
    """Per unit E: lambda, mu, bulk modulus."""
    lam, mu = nu / ((1 + nu) * (1 - 2 * nu)), 1 / (2 * (1 + nu))
    return lam, mu, lam + 2 * mu / 3


def d_full(nu):
    lam, mu, _ = lame(nu)
    return lam * np.outer(VOIGT_M, VOIGT_M) + mu * np.diag([2, 2, 2, 1, 1, 1])


def d_dev(nu):
    _, mu, _ = lame(nu)
    return mu * (np.diag([2, 2, 2, 1, 1, 1]) - (2 / 3) * np.outer(VOIGT_M, VOIGT_M))


def dofs(T):
    return (3 * T[:, :, None] + np.arange(3)[None, None]).reshape(len(T), 12)


def assemble(T, Ke, n):
    d = dofs(T)
    return sp.coo_matrix((Ke.ravel(), (np.repeat(d, 12, 1).ravel(), np.tile(d, (1, 12)).ravel())), shape=(n, n)).tocsr()


def plain_stiffness(op, E_e, nu):
    B = b_matrices(op.grads)
    Ke = (E_e * op.vol)[:, None, None] * np.einsum("tia,ij,tjb->tab", B, d_full(nu), B)
    return assemble(op.tets, Ke, 3 * len(op.m))


def divergence_rows(op):
    """S (N x 3N): S[n, :] u = sum_{e ni n} (vol_e / 4) div_e(u)."""
    T, G, vol = op.tets, op.grads, op.vol
    d = dofs(T)                                                         # (T, 12)
    bdiv = np.zeros((len(T), 12))
    for a in range(4):
        for c in range(3):
            bdiv[:, 3 * a + c] = G[:, c, a]                             # div_e = sum_a grad_c(phi_a) u_{a,c}
    rows = np.repeat(T, 12, 1)                                          # node n of e, repeated over 12 dofs
    cols = np.tile(d, (1, 4))
    vals = np.tile((vol / 4)[:, None] * bdiv, (1, 4))
    return sp.coo_matrix((vals.ravel(), (rows.ravel(), cols.ravel())), shape=(len(op.m), 3 * len(op.m))).tocsr()


def anp_stiffness(op, E_e, nu, S=None):
    B = b_matrices(op.grads)
    Kd = assemble(op.tets, (E_e * op.vol)[:, None, None] * np.einsum("tia,ij,tjb->tab", B, d_dev(nu), B), 3 * len(op.m))
    S = divergence_rows(op) if S is None else S
    _, _, kap = lame(nu)
    Vn = op.m                                                           # lumped mass = sum vol_e / 4
    kn = np.zeros(len(Vn))
    np.add.at(kn, op.tets.ravel(), np.repeat(E_e * op.vol / 4, 4))
    kn = kap * kn / Vn                                                  # nodal volume average of bulk modulus
    W = sp.diags(1 / Vn) @ S
    Kv = W.T @ sp.diags(Vn * kn) @ W
    return (Kd + Kv).tocsr()


def stiffness(kind, op, E_e, nu):
    return {"plain": plain_stiffness, "anp": anp_stiffness}[kind](op, E_e, nu)

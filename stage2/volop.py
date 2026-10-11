"""Point-based volume operator (DESIGN.md A1/A4, scalar implementation).

Validated in REPORT_opcheck.md: Delaunay of the points -> SDF carving -> boundary-sliver removal -> P1.
The result is returned as an `Operator` (K, lumped M, dof_dim) so that consumers (eigenbasis, Rayleigh-Ritz,
residual paths, coefficient models) do not depend on the scalar Laplacian; a stage-3 elasticity
implementation will return the same structure with dof_dim = 3.
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.spatial import Delaunay

EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
SLIVER_Q = 1e-3


@dataclass
class Operator:
    K: sp.csr_matrix                   # (n*d) x (n*d) symmetric PSD stiffness
    m: np.ndarray                      # lumped mass, length n*d
    dof_dim: int = 1
    tets: np.ndarray = None            # (T, 4) element connectivity used for assembly
    grads: np.ndarray = None           # (T, 3, 4) P1 basis gradients per element
    vol: np.ndarray = None             # (T,) element volumes
    info: dict = field(default_factory=dict)

    def assemble(self, coef_e):
        """Re-assemble K with per-element coefficients (e.g. hidden conductivity kappa_e)."""
        Kl = (coef_e * self.vol)[:, None, None] * np.einsum("tki,tkj->tij", self.grads, self.grads)
        n = len(self.m) // self.dof_dim
        T = self.tets
        return sp.coo_matrix((Kl.ravel(), (np.repeat(T, 4, 1).ravel(), np.tile(T, (1, 4)).ravel())),
                             shape=(n, n)).tocsr()

    def eigs(self, k, sigma=-1e-6):
        lam, vec = sla.eigsh(self.K.tocsc(), k=k, M=sp.diags(self.m).tocsc(), sigma=sigma)
        o = np.argsort(lam)
        return lam[o], vec[:, o]


def p1_elements(P, T):
    D = np.stack([P[T[:, k]] - P[T[:, 0]] for k in (1, 2, 3)], -1)
    vol = np.abs(np.linalg.det(D)) / 6
    ok = vol > 1e-14 * np.median(vol)
    D, vol, T = D[ok], vol[ok], T[ok]
    G = np.transpose(np.linalg.inv(D), (0, 2, 1))
    G = np.concatenate([-G.sum(2, keepdims=True), G], 2)          # (T, 3, 4)
    return T, G, vol


def delaunay_p1(P, is_surface, sdf, h, carve=True, sliver_q=SLIVER_Q):
    """Delaunay-P1 volume operator on points P (n, 3).

    sdf: callable (m, 3) -> (m,), > 0 inside.  h: local point spacing.
    Carving removes tets with centroid SDF < 0 or an edge midpoint SDF < -h/4; boundary slivers
    (>= 3 surface vertices and vol / l_max^3 < sliver_q) are always removed.  Points left without a tet
    get zero mass (callers drop them)."""
    T = Delaunay(P).simplices
    n_all = len(T)
    if carve:
        keep = sdf(P[T].mean(1)) >= 0
        for a, b in EDGES:
            keep &= sdf((P[T[:, a]] + P[T[:, b]]) / 2) >= -0.25 * h
        T = T[keep]
    n_carved = n_all - len(T)
    D = np.stack([P[T[:, k]] - P[T[:, 0]] for k in (1, 2, 3)], -1)
    vol = np.abs(np.linalg.det(D)) / 6
    lmax = np.max([np.linalg.norm(P[T[:, a]] - P[T[:, b]], axis=1) for a, b in EDGES], 0)
    sliver = (is_surface[T].sum(1) >= 3) & (vol / lmax ** 3 < sliver_q)
    T = T[~sliver]
    T, G, vol = p1_elements(P, T)
    n = len(P)
    Kl = vol[:, None, None] * np.einsum("tki,tkj->tij", G, G)
    K = sp.coo_matrix((Kl.ravel(), (np.repeat(T, 4, 1).ravel(), np.tile(T, (1, 4)).ravel())), shape=(n, n)).tocsr()
    m = np.zeros(n)
    np.add.at(m, T.ravel(), np.repeat(vol / 4, 4))
    return Operator(K=K, m=m, dof_dim=1, tets=T, grads=G, vol=vol,
                    info=dict(carved_frac=n_carved / n_all, sliver_frac=float(sliver.mean()),
                              isolated=int((m <= 0).sum())))

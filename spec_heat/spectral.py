"""Generalised eigendecomposition L φ = λ M φ and the truncated spectral
heat-equation solution  u(t) = Φ_K diag(exp(-α λ t)) Φ_Kᵀ M u0."""
import numpy as np
import scipy.sparse.linalg as sla


def eigendecompose(L, M, k=256, sigma=-1e-8):
    """Smallest k eigenpairs via shift-invert; Φ is M-orthonormal (ΦᵀMΦ = I)."""
    lam, Phi = sla.eigsh(L, k=k, M=M, sigma=sigma)
    order = np.argsort(lam)
    lam, Phi = lam[order], Phi[:, order]
    # ARPACK already returns M-orthonormal vectors; renormalise defensively
    Phi = Phi / np.sqrt(np.einsum("ij,ij->j", Phi, M @ Phi))
    return lam, Phi


def spectral_coefficients(Phi, M, u0):
    """c = Φᵀ M u0  (u0: (N,) or (N, n_ic))."""
    return Phi.T @ (M @ u0)


def spectral_solution(lam, Phi, coeffs, times, K, alpha=1.0):
    """u(t) for every t in `times` using the first K modes.

    coeffs: (k,) or (k, n_ic).  Returns array of shape (len(times), N[, n_ic]).
    """
    lamK, PhiK, cK = lam[:K], Phi[:, :K], coeffs[:K]
    out = []
    for t in times:
        decay = np.exp(-alpha * lamK * t)
        out.append(PhiK @ (decay[:, None] * cK if cK.ndim == 2 else decay * cK))
    return np.stack(out)


def m_norm(v, Mdiag):
    """||v||_M for a diagonal mass matrix; v may carry trailing axes."""
    return np.sqrt(np.tensordot(Mdiag, v ** 2, axes=(0, 0)))


def rel_error(u, u_ref, Mdiag):
    """||u - u_ref||_M / ||u_ref||_M along axis 0 (points)."""
    return m_norm(u - u_ref, Mdiag) / m_norm(u_ref, Mdiag)

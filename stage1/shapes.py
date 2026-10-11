"""Procedural closed surfaces with near-uniform triangle meshes (no downloads).

Families
--------
sh_sphere     r(d) = 1 + δ(d), δ = Σ_{l=2..6} a_lm Y_lm(d), max|δ| ~ U[0.15, 0.4]
superquadric  superellipsoid, semi-axes ~ U[0.6, 1.4], exponents e1, e2 ~ U[0.5, 1.5]
torus         R = 1, r/R ~ U[0.2, 0.6]

Meshing: dense candidate points on the surface -> farthest point sampling
(FPS) of N vertices (near-uniform spacing) -> triangulation through a
parameterisation: convex hull of the unit directions for the star-shaped
families, periodic Delaunay in conformal (u, w) coordinates for the torus.
Every mesh is then centred (area centroid), scaled to total area AREA and
randomly rotated.
"""
import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation
from scipy.special import sph_harm_y

_ICO = np.array(trimesh.creation.icosphere(subdivisions=6).vertices)   # 40962 unit directions


# ------------------------------------------------------------------ helpers --
def fps(P, n, rng):
    """Farthest point sampling (Euclidean) of n indices from candidates P."""
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(len(P))
    d = ((P - P[idx[0]]) ** 2).sum(1)
    for i in range(1, n):
        idx[i] = np.argmax(d)
        d = np.minimum(d, ((P - P[idx[i]]) ** 2).sum(1))
    return idx


def _orient(V, F, outward):
    """Flip faces whose normal points against `outward` (per-face vectors)."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    flip = (n * outward).sum(1) < 0
    F = F.copy()
    F[flip] = F[flip][:, [0, 2, 1]]
    return F


def check_mesh(V, F, euler):
    """Closed 2-manifold with the expected Euler characteristic, all vertices used."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, counts = np.unique(E, axis=0, return_counts=True)
    ok = (counts == 2).all() and len(np.unique(F)) == len(V)
    chi = len(V) - len(counts) + len(F)
    return bool(ok and chi == euler), chi


def angles(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    def ang(p, q, r):
        u, v = q - p, r - p
        return np.degrees(np.arccos(np.clip((u * v).sum(1) / (np.linalg.norm(u, axis=1)
                                                               * np.linalg.norm(v, axis=1)), -1, 1)))
    return np.stack([ang(a, b, c), ang(b, c, a), ang(c, a, b)], 1)


# ---------------------------------------------------------- star-shaped ----
def _real_sh(l, m, d):
    theta = np.arccos(np.clip(d[:, 2], -1, 1))
    phi = np.arctan2(d[:, 1], d[:, 0])
    y = sph_harm_y(l, abs(m), theta, phi)
    return y.real if m >= 0 else y.imag


def sh_sphere_radius(rng):
    coeffs = [(l, m, rng.normal() / l ** 2) for l in range(2, 7) for m in range(-l, l + 1)]
    delta = sum(a * _real_sh(l, m, _ICO) for l, m, a in coeffs)
    amp = rng.uniform(0.15, 0.4) / np.abs(delta).max()

    def radius(d):
        return 1 + amp * sum(a * _real_sh(l, m, d) for l, m, a in coeffs)
    return radius, dict(amp=float(amp * np.abs(delta).max()))


def superquadric_radius(rng):
    a, b, c = rng.uniform(0.6, 1.4, 3)
    e1, e2 = rng.uniform(0.5, 1.5, 2)

    def radius(d):
        x, y, z = np.abs(d[:, 0]) / a, np.abs(d[:, 1]) / b, np.abs(d[:, 2]) / c
        F = (x ** (2 / e2) + y ** (2 / e2)) ** (e2 / e1) + z ** (2 / e1)
        return F ** (-e1 / 2)
    return radius, dict(a=a, b=b, c=c, e1=e1, e2=e2)


def star_mesh(radius, n, rng):
    cand = _ICO * radius(_ICO)[:, None]
    idx = fps(cand, n, rng)
    D = _ICO[idx]
    F = ConvexHull(D).simplices
    V = cand[idx]
    F = _orient(V, F, V[F].mean(1))
    return V, F, 2


# ------------------------------------------------------------------ torus ----
def torus_mesh(rng, n):
    R, r = 1.0, rng.uniform(0.2, 0.6)
    m = 60000
    u = rng.uniform(0, 2 * np.pi, 3 * m)
    v = rng.uniform(0, 2 * np.pi, 3 * m)
    keep = rng.uniform(0, R + r, 3 * m) < R + r * np.cos(v)          # area element ∝ R + r cos v
    u, v = u[keep][:m], v[keep][:m]
    P = np.stack([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)], 1)
    idx = fps(P, n, rng)
    u, v, V = u[idx], v[idx], P[idx]

    # conformal coordinate w(v) = ∫ r/(R + r cos v) dv  -> metric ∝ du² + dw²
    s, k = np.sqrt(R ** 2 - r ** 2), np.sqrt((R - r) / (R + r))
    w = (2 * r / s) * np.arctan2(k * np.sin(v / 2), np.cos(v / 2))    # v in [0, 2π) -> w in [0, W)
    W = 2 * np.pi * r / s
    uv = np.stack([u, w], 1)
    tiles = [uv + np.array([du, dw]) for du in (-2 * np.pi, 0, 2 * np.pi) for dw in (-W, 0, W)]
    T = Delaunay(np.concatenate(tiles)).simplices
    cen = np.concatenate(tiles)[T].mean(1)
    T = T[(cen[:, 0] >= 0) & (cen[:, 0] < 2 * np.pi) & (cen[:, 1] >= 0) & (cen[:, 1] < W)] % n
    core = np.stack([np.cos(u), np.sin(u), np.zeros_like(u)], 1) * R
    out = (V - core)[T].mean(1)
    F = _orient(V, T, out)
    return V, F, 0, dict(r_over_R=r)


# ------------------------------------------------------------- normalise ----
def face_areas(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def normalise(V, F, area, rng):
    A = face_areas(V, F)
    V = V - (A[:, None] * V[F].mean(1)).sum(0) / A.sum()
    V = V * np.sqrt(area / A.sum())
    return V @ Rotation.random(random_state=rng.integers(2 ** 31)).as_matrix().T


def make_shape(family, n, area, rng):
    if family == "torus":
        V, F, euler, params = torus_mesh(rng, n)
    else:
        radius, params = (sh_sphere_radius if family == "sh_sphere" else superquadric_radius)(rng)
        V, F, euler = star_mesh(radius, n, rng)
    ok, chi = check_mesh(V, F, euler)
    return normalise(V, F, area, rng), F, ok, chi, params


# --------------------------------------------------------------- operators ---
def cotan_laplacian(V, F):
    """P1 stiffness (cotan) matrix L (PSD) and lumped mass (area/3 per vertex)."""
    n = len(V)
    rows, cols, vals = [], [], []
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        a, b, c = V[F[:, i]], V[F[:, j]], V[F[:, k]]
        u, v = a - c, b - c                                   # angle at vertex k, opposite edge (i, j)
        cot = (u * v).sum(1) / np.linalg.norm(np.cross(u, v), axis=1)
        rows += [F[:, i], F[:, j]]
        cols += [F[:, j], F[:, i]]
        vals += [-0.5 * cot, -0.5 * cot]
    L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    L = L - sp.diags(np.asarray(L.sum(1)).ravel())
    m = np.zeros(n)
    np.add.at(m, F.ravel(), np.repeat(face_areas(V, F) / 3, 3))
    return L.tocsr(), m

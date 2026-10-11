"""Geometries, point-cloud sampling and the robust point-cloud Laplacian.

Shapes
------
square       : [0,1]^2
disk         : unit disk (radius 1, centre at origin)
square_hole  : [0,1]^2 minus the disk of radius 0.2 centred at (0.5, 0.5)

Each shape knows its signed distance (positive inside), area and boundary
curves (parametrised by arc length); the gmsh geometry lives in fem_reference.py.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import robust_laplacian
from scipy.spatial import Delaunay

DENSITY_RATIO = 5.0  # non-uniform sampling: dense half has 5x the point density
HOLE_C, HOLE_R = np.array([0.5, 0.5]), 0.2


# --------------------------------------------------------------------------- #
# boundary curves: each is (length, param(s) -> xy) with s in [0, length)
# --------------------------------------------------------------------------- #
def _segment(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ln = np.linalg.norm(b - a)
    return ln, lambda s: a + (s[:, None] / ln) * (b - a)


def _circle(c, r):
    c = np.asarray(c, float)
    return 2 * np.pi * r, lambda s: c + r * np.stack([np.cos(s / r), np.sin(s / r)], 1)


_SQUARE_EDGES = [_segment((0, 0), (1, 0)), _segment((1, 0), (1, 1)),
                 _segment((1, 1), (0, 1)), _segment((0, 1), (0, 0))]


@dataclass
class Shape:
    name: str
    area: float
    bbox: tuple                      # (xmin, xmax, ymin, ymax)
    sdf: Callable                    # signed distance, >0 inside
    curves: list                     # boundary curves
    x_mid: float                     # x < x_mid is the dense side (non-uniform)

    @property
    def perimeter(self):
        return sum(c[0] for c in self.curves)

    def inside(self, x, margin=0.0):
        return self.sdf(x) > margin


def _sdf_square(x):
    return np.minimum.reduce([x[:, 0], 1 - x[:, 0], x[:, 1], 1 - x[:, 1]])


def _sdf_disk(x):
    return 1.0 - np.linalg.norm(x, axis=1)


def _sdf_hole(x):
    return np.minimum(_sdf_square(x), np.linalg.norm(x - HOLE_C, axis=1) - HOLE_R)


SHAPES = {
    "square": Shape("square", 1.0, (0, 1, 0, 1), _sdf_square, _SQUARE_EDGES, 0.5),
    "disk": Shape("disk", np.pi, (-1, 1, -1, 1), _sdf_disk, [_circle((0, 0), 1.0)], 0.0),
    "square_hole": Shape("square_hole", 1.0 - np.pi * HOLE_R ** 2, (0, 1, 0, 1), _sdf_hole,
                         _SQUARE_EDGES + [_circle(HOLE_C, HOLE_R)], 0.5),
}


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def _weight(shape, x, sampling):
    """Relative point density w(x): 1 (uniform) or 5 on the x < x_mid half."""
    if sampling == "uniform":
        return np.ones(len(x))
    return np.where(x[:, 0] < shape.x_mid, DENSITY_RATIO, 1.0)


def _rejection(n, propose, accept_prob, rng):
    out, have = [], 0
    while have < n:
        x = propose(max(2 * (n - have), 64))
        x = x[rng.random(len(x)) < accept_prob(x)]
        out.append(x)
        have += len(x)
    return np.concatenate(out)[:n]


def sample_point_cloud(shape, N, sampling, rng):
    """N points: iid interior points with density ∝ w(x) plus iid boundary
    points with linear density ∝ sqrt(w) (matching the local spacing).

    The interior/boundary split solves  N = rho*∫w dA + sqrt(rho)*∫sqrt(w) ds,
    i.e. boundary spacing ≈ interior spacing.  All three shapes are mirror
    symmetric about x = x_mid, so the dense/sparse halves have equal area and
    equal boundary length.
    """
    wmax = DENSITY_RATIO if sampling == "nonuniform" else 1.0
    A_eff = shape.area * (1 + wmax) / 2
    P_eff = shape.perimeter * (1 + np.sqrt(wmax)) / 2
    s = (-P_eff + np.sqrt(P_eff ** 2 + 4 * A_eff * N)) / (2 * A_eff)  # sqrt(rho)
    n_b = int(round(s * P_eff))
    n_i = N - n_b

    xmin, xmax, ymin, ymax = shape.bbox

    def propose_interior(m):
        return np.stack([rng.uniform(xmin, xmax, m), rng.uniform(ymin, ymax, m)], 1)

    interior = _rejection(
        n_i, propose_interior,
        lambda x: shape.inside(x) * _weight(shape, x, sampling) / wmax, rng)

    lengths = np.array([c[0] for c in shape.curves])
    cum = np.concatenate([[0], np.cumsum(lengths)])

    def propose_boundary(m):
        s_all = rng.uniform(0, cum[-1], m)
        idx = np.searchsorted(cum, s_all, side="right") - 1
        pts = np.empty((m, 2))
        for i, (_, f) in enumerate(shape.curves):
            sel = idx == i
            pts[sel] = f(s_all[sel] - cum[i])
        return pts

    boundary = _rejection(
        n_b, propose_boundary,
        lambda x: np.sqrt(_weight(shape, x, sampling) / wmax), rng)

    pts = np.concatenate([interior, boundary])
    is_boundary = np.r_[np.zeros(n_i, bool), np.ones(n_b, bool)]
    return pts, is_boundary


# --------------------------------------------------------------------------- #
# Laplacian
# --------------------------------------------------------------------------- #
def build_laplacian(pts2d):
    """Robust point-cloud Laplacian (Sharp & Crane 2020) of a planar cloud.

    Returns (L, M): L is the positive semi-definite weak Laplacian (stiffness),
    M the diagonal (lumped) mass matrix, both scipy.sparse.
    """
    X = np.c_[pts2d, np.zeros(len(pts2d))]
    L, M = robust_laplacian.point_cloud_laplacian(X)
    return L.tocsc(), M.tocsc()


def build_delaunay_laplacian(pts2d, shape):
    """Diagnostic baseline: cotan Laplacian + lumped mass on the Delaunay
    triangulation of the same points (triangles whose centroid lies outside
    the shape, i.e. inside the hole, are removed)."""
    tri = Delaunay(pts2d).simplices
    tri = tri[shape.inside(pts2d[tri].mean(1))]
    X = np.c_[pts2d, np.zeros(len(pts2d))]
    L, M = robust_laplacian.mesh_laplacian(np.ascontiguousarray(X), np.ascontiguousarray(tri))
    return L.tocsc(), M.tocsc()

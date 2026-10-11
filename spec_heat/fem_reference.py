"""FEM reference solution: gmsh mesh -> meshio -> scikit-fem P1, Crank–Nicolson.

* The mesh conforms to the step-IC circle (OCC fragment) and is graded to
  h/4 in a band around it, so the discontinuous IC is represented exactly
  element-wise; u0 is the L2 projection  M u0 = ∫ f φ.
* Time stepping: Crank–Nicolson with piecewise-constant dt (n steps between
  consecutive output times).  The first two CN steps are replaced by four
  backward-Euler half steps (Rannacher start-up) to damp the non-smooth
  modes of the step IC — the half-step BE matrix M + dt/2 K is the CN
  left-hand side, so one factorisation serves both.
"""
import os
import tempfile

# gmsh's bundled FLTK writes $HOME/.fltk/fltk.org/fltk.prefs; point HOME at a
# project folder (affects this Python process only) before gmsh is imported.
os.environ["HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gmsh_home")
os.makedirs(os.environ["HOME"], exist_ok=True)

import gmsh  # noqa: E402
import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import skfem
from matplotlib.tri import Triangulation
from skfem.models.poisson import laplace, mass

from laplacian import HOLE_C, HOLE_R


# --------------------------------------------------------------------------- #
# mesh
# --------------------------------------------------------------------------- #
def build_mesh(shape_name, step_center, step_radius, h, band=0.02, grow=0.08):
    """Triangular mesh of the shape, conforming to the step circle.

    Size field: h/4 within `band` of the step circle, growing linearly to h
    over a further distance `grow`.  Returns skfem.MeshTri.
    """
    cx, cy = step_center
    hf = h / 4
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(shape_name)
        occ = gmsh.model.occ
        if shape_name == "square":
            s = occ.addRectangle(0, 0, 0, 1, 1)
        elif shape_name == "disk":
            s = occ.addDisk(0, 0, 0, 1, 1)
        elif shape_name == "square_hole":
            r = occ.addRectangle(0, 0, 0, 1, 1)
            c = occ.addDisk(HOLE_C[0], HOLE_C[1], 0, HOLE_R, HOLE_R)
            out, _ = occ.cut([(2, r)], [(2, c)])
            s = out[0][1]
        else:
            raise ValueError(shape_name)
        d = occ.addDisk(cx, cy, 0, step_radius, step_radius)
        occ.fragment([(2, s)], [(2, d)])
        occ.synchronize()

        dist = f"abs(sqrt((x-({cx}))^2+(y-({cy}))^2)-{step_radius})"
        expr = f"min({h}, max({hf}, {hf} + {(h - hf) / grow}*({dist}-{band})))"
        f = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(f, "F", expr)
        gmsh.model.mesh.field.setAsBackgroundMesh(f)
        for opt in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints",
                    "Mesh.MeshSizeFromCurvature"):
            gmsh.option.setNumber(opt, 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(2)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mesh.msh")
            gmsh.write(path)
            m = meshio.read(path)
    finally:
        gmsh.finalize()

    tri = np.concatenate([c.data for c in m.cells if c.type == "triangle"])
    used, tri = np.unique(tri, return_inverse=True)
    tri = tri.reshape(-1, 3)
    p = m.points[used, :2]
    # counter-clockwise orientation
    e1, e2 = p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]]
    flip = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return skfem.MeshTri(p.T.copy(), tri.T.copy())


# --------------------------------------------------------------------------- #
# P1 interpolation from a mesh to arbitrary points
# --------------------------------------------------------------------------- #
class P1Interpolator:
    """Builds sparse interpolation matrices (n_pts x n_nodes) for a P1 mesh.

    Points outside the (polygonal) mesh — e.g. cloud points on a curved
    boundary between two mesh vertices — are projected onto the nearest
    boundary edge (distance ~h^2/8R, negligible).
    """

    def __init__(self, mesh):
        self.p = mesh.p.T
        self.t = mesh.t.T
        self.tri = Triangulation(self.p[:, 0], self.p[:, 1], self.t)
        self.finder = self.tri.get_trifinder()
        self.bedges = mesh.facets[:, mesh.boundary_facets()].T

    def matrix(self, pts):
        pts = np.asarray(pts, float)
        n = len(pts)
        el = self.finder(pts[:, 0], pts[:, 1])
        rows, cols, vals = [], [], []

        ok = np.where(el >= 0)[0]
        if len(ok):
            v = self.t[el[ok]]
            a, b, c = self.p[v[:, 0]], self.p[v[:, 1]], self.p[v[:, 2]]
            x = pts[ok]
            det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
            l1 = ((x[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (x[:, 1] - a[:, 1])) / det
            l2 = ((b[:, 0] - a[:, 0]) * (x[:, 1] - a[:, 1]) - (x[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / det
            lam = np.stack([1 - l1 - l2, l1, l2], 1)
            rows.append(np.repeat(ok, 3)); cols.append(v.ravel()); vals.append(lam.ravel())

        out = np.where(el < 0)[0]
        if len(out):
            A, B = self.p[self.bedges[:, 0]], self.p[self.bedges[:, 1]]
            AB = B - A
            for i in out:
                s = np.clip(((pts[i] - A) * AB).sum(1) / (AB ** 2).sum(1), 0, 1)
                d = np.linalg.norm(A + s[:, None] * AB - pts[i], axis=1)
                j = np.argmin(d)
                rows.append([i, i]); cols.append(self.bedges[j]); vals.append([1 - s[j], s[j]])

        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n, len(self.p)))


# --------------------------------------------------------------------------- #
# heat solver
# --------------------------------------------------------------------------- #
def assemble(mesh):
    basis = skfem.Basis(mesh, skfem.ElementTriP1(), intorder=4)
    return basis, laplace.assemble(basis).tocsc(), mass.assemble(basis).tocsc()


def l2_project(basis, M, funcs):
    """Columns: L2 projections of each f(x: (n,2)) -> (n,) onto P1."""
    cols = []
    for f in funcs:
        @skfem.LinearForm
        def rhs(v, w, f=f):
            x = np.asarray(w.x)
            return f(x.reshape(2, -1).T).reshape(x.shape[1:]) * v
        cols.append(sla.spsolve(M, rhs.assemble(basis)))
    return np.stack(cols, 1)


def crank_nicolson(K, M, U0, times, n_steps, alpha=1.0, rannacher=2):
    """Returns array (len(times), n_nodes, n_ic)."""
    U, t_prev, out = U0.copy(), 0.0, []
    for i, T in enumerate(times):
        dt = (T - t_prev) / n_steps
        lu = sla.splu((M + 0.5 * dt * alpha * K).tocsc())
        B = (M - 0.5 * dt * alpha * K).tocsr()
        for k in range(n_steps):
            if i == 0 and k < rannacher:        # 2 BE half steps == 1 step of dt
                U = lu.solve(M @ U)
                U = lu.solve(M @ U)
            else:
                U = lu.solve(B @ U)
        out.append(U.copy())
        t_prev = T
    return np.stack(out)


def compute_reference(shape_name, ic_funcs, step_center, step_radius, times, h, n_steps):
    """Solve the heat equation for every IC on a mesh of size h.

    Returns dict with the mesh, consistent mass matrix and solutions
    U (len(times), n_nodes, n_ic).
    """
    mesh = build_mesh(shape_name, step_center, step_radius, h)
    basis, K, M = assemble(mesh)
    U0 = l2_project(basis, M, ic_funcs)
    U = crank_nicolson(K, M, U0, times, n_steps)
    return dict(mesh=mesh, M=M, U=U, U0=U0)


def self_convergence(coarse, fine):
    """||u_coarse - u_fine||_M / ||u_fine||_M on the fine mesh.

    Returns (len(times), n_ic)."""
    P = P1Interpolator(coarse["mesh"]).matrix(fine["mesh"].p.T)
    M = fine["M"]
    err = np.empty(coarse["U"].shape[::2])
    for i in range(err.shape[0]):
        d = P @ coarse["U"][i] - fine["U"][i]
        err[i] = np.sqrt(np.einsum("ij,ij->j", d, M @ d) /
                         np.einsum("ij,ij->j", fine["U"][i], M @ fine["U"][i]))
    return err

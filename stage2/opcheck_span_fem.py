"""Supplement to opcheck: span loss for torus and cup against fine gmsh tet P1 FEM eigenvectors
interpolated to the point positions -> results_opcheck/span_fem.csv

  reference   gmsh tet mesh h=0.03 (P1, eigsh 48 modes with eigenvectors); h=0.04 mesh gives the
              reference floor (span loss between the two references at the same points)
  K_ref       largest count <= 32 of complete eigenvalue groups (relative gap < 2e-3 = same group;
              the torus has rotational pairs)
  interpolation  point location in the fine tet mesh (k nearest tet centroids as candidates,
              barycentric coordinates; points just outside the polyhedral mesh -> clamped weights)
  operators   delaunay (SDF carving + boundary-sliver removal), delaunay_nocarve; same sampling and seeds
              as opcheck.py

    python stage2/opcheck_span_fem.py        (env neo, CPU)
"""
import os
import sys
import time

import opcheck as O                     # sets HOME / cache dirs (project-local) on import

from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.spatial import cKDTree

OUT = O.OUT
K_EIG = 48
GROUP_TOL = 2e-3


def fem_mesh_eig(dom, h):
    path = os.path.join(OUT, f"{dom}_femvec_h{h:g}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["P"], d["T"], d["lam"], d["vec"]
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    if dom == "torus":
        gmsh.model.occ.addTorus(0, 0, 0, O.TORUS_R, O.TORUS_r)
    else:
        outer = gmsh.model.occ.addCylinder(0, 0, -0.5, 0, 0, 1.0, 1.0)
        cavity = gmsh.model.occ.addCylinder(0, 0, O.CUP_ZB, 0, 0, 0.8 - O.CUP_ZB, O.CUP_RI)
        gmsh.model.occ.cut([(3, outer)], [(3, cavity)])
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.model.mesh.generate(3)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    _, conn = gmsh.model.mesh.getElementsByType(4)
    gmsh.finalize()
    lut = np.empty(tags.max() + 1, np.int64)
    lut[tags] = np.arange(len(tags))
    T = lut[conn.reshape(-1, 4)]
    used, T = np.unique(T, return_inverse=True)
    T = T.reshape(-1, 4)
    P = coords.reshape(-1, 3)[used]
    K, m, _, _ = O.tet_p1(P, T)
    lam, vec = sla.eigsh(K.tocsc(), k=K_EIG, M=sp.diags(m).tocsc(), sigma=-1e-6)
    o = np.argsort(lam)
    np.savez(path, P=P, T=T, lam=lam[o], vec=vec[:, o])
    return P, T, lam[o], vec[:, o]


def k_ref_from_eigs(lam, kmax=32, tol=GROUP_TOL):
    bounds = [i for i in range(1, len(lam)) if (lam[i] - lam[i - 1]) > tol * max(lam[i], 1e-12)]
    return max(b for b in bounds if b <= kmax)


def interp(P_mesh, T, X, k=16):
    """Sparse (len(X) x len(P_mesh)) barycentric interpolation in the tet mesh."""
    cen = P_mesh[T].mean(1)
    _, cand = cKDTree(cen).query(X, k)
    best_min = np.full(len(X), -np.inf)
    best_t = np.zeros(len(X), int)
    best_w = np.zeros((len(X), 4))
    for j in range(k):
        t = cand[:, j]
        V = P_mesh[T[t]]                                                # (n, 4, 3)
        A = np.stack([V[:, 1] - V[:, 0], V[:, 2] - V[:, 0], V[:, 3] - V[:, 0]], -1)
        lam3 = np.linalg.solve(A, (X - V[:, 0])[..., None])[..., 0]
        w = np.c_[1 - lam3.sum(1), lam3]
        mn = w.min(1)
        better = mn > best_min
        best_min[better], best_t[better], best_w[better] = mn[better], t[better], w[better]
    w = np.clip(best_w, 0, None)
    w /= w.sum(1, keepdims=True)
    rows = np.repeat(np.arange(len(X)), 4)
    return sp.csr_matrix((w.ravel(), (rows, T[best_t].ravel())), shape=(len(X), len(P_mesh))), best_min


def span(Phi, U, m):
    G = U.T @ (m[:, None] * U)
    U = np.linalg.solve(np.linalg.cholesky(0.5 * (G + G.T)), U.T).T
    return float(1 - ((Phi.T @ (m[:, None] * U)) ** 2).sum() / U.shape[1])


def task(args):
    dom, kind, N, seed, op = args
    rng = np.random.default_rng([2026, ["ball", "cube", "shell", "torus", "cup"].index(dom),
                                 0 if kind == "random" else 1, N, seed])
    P, is_s, h = O.sample(dom, N, kind, rng)
    L, m, _ = O.op_delaunay(P, dom, h, is_s, carve=(op == "delaunay"))
    keep = m > 0
    L, m, P = L[keep][:, keep], m[keep], P[keep]
    Pm3, T3, lam3, vec3 = fem_mesh_eig(dom, 0.03)
    Pm4, T4, lam4, vec4 = fem_mesh_eig(dom, 0.04)
    kr = k_ref_from_eigs(lam3)
    I3, out3 = interp(Pm3, T3, P)
    I4, _ = interp(Pm4, T4, P)
    U3, U4 = I3 @ vec3[:, :kr], I4 @ vec4[:, :kr]
    lam, Phi = sla.eigsh(L.tocsc(), k=kr + 8, M=sp.diags(m).tocsc(), sigma=-1e-6)
    Phi = Phi[:, np.argsort(lam)][:, :kr]
    # reference floor: M-orthonormal h=0.04 reference vs h=0.03 reference at the same points
    G4 = U4.T @ (m[:, None] * U4)
    Q4 = np.linalg.solve(np.linalg.cholesky(0.5 * (G4 + G4.T)), U4.T).T
    return dict(domain=dom, sampling=kind, N=N, seed=seed, op=op, K_ref=kr, span_loss_fem=span(Phi, U3, m),
                ref_floor=span(Q4, U3, m), frac_points_outside_mesh=float((out3 < -1e-9).mean()))


def main():
    for dom in ("torus", "cup"):
        for h in (0.03, 0.04):
            P, T, lam, _ = fem_mesh_eig(dom, h)
            print(f"{dom} h={h}: {len(P)} verts, K_ref={k_ref_from_eigs(lam)}", flush=True)
    tasks = [(dom, kind, N, s, op) for dom in ("torus", "cup") for kind in ("quasi", "random")
             for N in O.NS for s in O.SEEDS for op in ("delaunay", "delaunay_nocarve")]
    tasks.sort(key=lambda t: -t[2])
    t0 = time.time()
    with Pool(12) as p:
        rows = list(p.imap_unordered(task, tasks, chunksize=1))
    df = pd.DataFrame(rows).sort_values(["domain", "op", "sampling", "N", "seed"])
    df.to_csv(os.path.join(OUT, "span_fem.csv"), index=False)
    print(df.groupby(["domain", "op", "sampling", "N"])[["span_loss_fem", "ref_floor", "frac_points_outside_mesh"]]
          .mean().to_string(float_format=lambda v: f"{v:.2e}"))
    print(f"{len(df)} runs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

"""P0 (CPU part): re-evaluation sets for stage 1 -> data_p0/<set>/shape_XXX.npz

Sets (N = 4000 points each):
  T0  test shapes, FPS mesh vertices (as in stage 1)             -- reference
  T1  test shapes, uniform random samples on the surface
  T2  test shapes, non-uniform samples (5x density on the half-space x·n > 0, n random per shape)
  T3  held-out family "dumbbell" (smooth union of two balls), FPS mesh vertices, 60 new shapes

Ground truth for every set: a fine mesh = the 4k mesh subdivided twice (1→4 midpoint) with new vertices
projected onto the true surface (~64k vertices); cotan + lumped mass; Crank–Nicolson with Rannacher
start at the stage-1 times; interpolated to the points (barycentric on the closest fine triangle).
The 4k vertices are vertices of the fine mesh, so T0/T3 need no interpolation.
Evaluation weights: fine lumped masses assigned to the nearest point (restricted-Voronoi area);
used only for the metric.  Model-side quantities use points only (pc Laplacian nn=20 + c, ARPACK 96,
PC-discrete = expm_multiply on the calibrated pc operator).

    python stage1/p0_make.py [--validate]      (env neo, CPU)
"""
import common  # noqa: F401

import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import trimesh
from robust_laplacian import point_cloud_laplacian
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import shapes as S
from common import AREA, DATA, K_MODES, N_VERT, PC_NEIGHBORS, SEED, TAUS

OUT = os.path.join(common.ROOT, "data_p0")
N_PTS = 4000
CN_STEPS = 200                 # CN steps per output interval on the fine mesh
N_HELDOUT = 60
DENSITY_RATIO = 5.0


# --------------------------------------------------------------- geometry --
def dumbbell_radius(rng):
    """Smooth union of two balls that both contain the origin (star-shaped w.r.t. the origin)."""
    e = rng.normal(size=3)
    e /= np.linalg.norm(e)
    a1, a2 = rng.uniform(0.3, 0.6, 2)
    R1, R2 = rng.uniform(0.65, 0.9, 2)
    cs, Rs = [a1 * e, -a2 * e], [R1, R2]
    k = 12.0

    def radius(d):
        ts = []
        for c, R in zip(cs, Rs):
            dc = d @ c
            ts.append(dc + np.sqrt(dc ** 2 - c @ c + R ** 2))
        ts = np.stack(ts)
        return np.log(np.exp(k * ts).sum(0)) / k
    return radius, dict(a1=a1, a2=a2, R1=R1, R2=R2)


def replay(family, rng):
    """Re-run the stage-1 generator, returning the normalised 4k mesh, the raw mesh, the normalising
    transform and a projector onto the true (raw-frame) surface."""
    if family == "torus":
        V, F, euler, params = S.torus_mesh(rng, N_VERT)
        R, r = 1.0, params["r_over_R"]

        def project(p):
            q = np.c_[p[:, :2], np.zeros(len(p))]
            q = R * q / np.linalg.norm(q, axis=1, keepdims=True)
            return q + r * (p - q) / np.linalg.norm(p - q, axis=1, keepdims=True)
    else:
        radius, params = {"sh_sphere": S.sh_sphere_radius, "superquadric": S.superquadric_radius,
                          "dumbbell": dumbbell_radius}[family](rng)
        V, F, euler = S.star_mesh(radius, N_VERT, rng)

        def project(p):
            d = p / np.linalg.norm(p, axis=1, keepdims=True)
            return d * radius(d)[:, None]
    ok, _ = S.check_mesh(V, F, euler)
    A = S.face_areas(V, F)
    c = (A[:, None] * V[F].mean(1)).sum(0) / A.sum()
    s = np.sqrt(AREA / A.sum())
    Rm = Rotation.random(random_state=rng.integers(2 ** 31)).as_matrix()
    fwd = lambda P: ((P - c) * s) @ Rm.T
    inv = lambda Q: (Q @ Rm) / s + c
    return dict(V=fwd(V), F=F, V_raw=V, fwd=fwd, inv=inv, project=project, ok=ok, params=params)


def fine_mesh(g, levels=2):
    V, F = g["V_raw"], g["F"]
    for _ in range(levels):
        V, F = trimesh.remesh.subdivide(V, F)
        V = g["project"](V)
    return g["fwd"](V), F


# ------------------------------------------------------------ heat solver --
def crank_nicolson(L, m, U0, times, n_steps=CN_STEPS):
    """(M + dt/2 L) u+ = (M − dt/2 L) u, Rannacher start (4 BE half steps). U0 (N, I)."""
    M = sp.diags(m).tocsc()
    U, tp, out = U0.copy(), 0.0, []
    for i, T in enumerate(times):
        dt = (T - tp) / n_steps
        lu = sla.splu((M + 0.5 * dt * L).tocsc())
        B = (M - 0.5 * dt * L).tocsr()
        for k in range(n_steps):
            if i == 0 and k < 2:
                U = lu.solve(M @ U)
                U = lu.solve(M @ U)
            else:
                U = lu.solve(B @ U)
        out.append(U.copy())
        tp = T
    return np.stack(out)                                                   # (T, N, I)


def gaussians(P, spec, centers_xyz):
    out = []
    for s, cxyz in zip(spec, centers_xyz):
        out.append(sum(a * np.exp(-((P - c) ** 2).sum(1) / (2 * sg ** 2))
                       for c, sg, a in zip(cxyz, s["sigma"], s["amp"])))
    return np.stack(out)                                                    # (I, N)


# ------------------------------------------------------------- per shape --
def sample_surface(Vf, Ff, g, n, rng, weight=None):
    """Area-uniform samples on the fine mesh (optionally rejection-weighted), projected to the true surface."""
    A = S.face_areas(Vf, Ff)
    pts = []
    while sum(len(p) for p in pts) < n:
        m = 4 * n
        f = rng.choice(len(Ff), m, p=A / A.sum())
        b = rng.dirichlet(np.ones(3), m)
        P = (Vf[Ff[f]] * b[:, :, None]).sum(1)
        if weight is not None:
            P = P[rng.uniform(0, 1, m) < weight(P)]
        pts.append(P)
    P = np.concatenate(pts)[:n]
    return g["fwd"](g["project"](g["inv"](P)))


def _closest_on_triangles(p, a, b, c):
    """Exact closest point on triangles (a, b, c) to points p (Ericson, Real-Time Collision Detection 5.1.5).
    All arrays (n, 3); returns barycentric weights (n, 3)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    w = np.zeros((len(p), 3))
    denom = va + vb + vc
    with np.errstate(divide="ignore", invalid="ignore"):
        v_in, w_in = vb / denom, vc / denom
        cases = [
            ((d1 <= 0) & (d2 <= 0), lambda: np.stack([np.ones_like(d1), 0 * d1, 0 * d1], 1)),
            ((d3 >= 0) & (d4 <= d3), lambda: np.stack([0 * d1, np.ones_like(d1), 0 * d1], 1)),
            ((d6 >= 0) & (d5 <= d6), lambda: np.stack([0 * d1, 0 * d1, np.ones_like(d1)], 1)),
            ((vc <= 0) & (d1 >= 0) & (d3 <= 0), lambda: (lambda v: np.stack([1 - v, v, 0 * v], 1))(d1 / (d1 - d3))),
            ((vb <= 0) & (d2 >= 0) & (d6 <= 0), lambda: (lambda v: np.stack([1 - v, 0 * v, v], 1))(d2 / (d2 - d6))),
            ((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
             lambda: (lambda v: np.stack([0 * v, 1 - v, v], 1))((d4 - d3) / ((d4 - d3) + (d5 - d6)))),
        ]
        done = np.zeros(len(p), bool)
        for mask, fn in cases:
            m = mask & ~done
            if m.any():
                w[m] = fn()[m]
            done |= m
        w[~done] = np.stack([1 - v_in - w_in, v_in, w_in], 1)[~done]
    return w


def interp_matrix(Vf, Ff, P, k=12):
    """Barycentric interpolation at the closest point of the fine mesh (k nearest triangle centroids
    as candidates, exact point-triangle projection)."""
    tri_c = Vf[Ff].mean(1)
    _, cand = cKDTree(tri_c).query(P, k)                                      # (n, k)
    best_d = np.full(len(P), np.inf)
    best_t = np.zeros(len(P), int)
    best_w = np.zeros((len(P), 3))
    for j in range(k):
        t = cand[:, j]
        a, b, c = Vf[Ff[t, 0]], Vf[Ff[t, 1]], Vf[Ff[t, 2]]
        w = _closest_on_triangles(P, a, b, c)
        q = w[:, :1] * a + w[:, 1:2] * b + w[:, 2:] * c
        d = np.linalg.norm(q - P, axis=1)
        better = d < best_d
        best_d[better], best_t[better], best_w[better] = d[better], t[better], w[better]
    rows = np.repeat(np.arange(len(P)), 3)
    return sp.csr_matrix((best_w.ravel(), (rows, Ff[best_t].ravel())), shape=(len(P), len(Vf)))


def pc_side(P, U0, t):
    L, M = point_cloud_laplacian(P, n_neighbors=PC_NEIGHBORS)
    L, M = L.tocsc(), M.tocsc()
    m = M.diagonal()
    c = 2 * m.sum() / sum(P[:, i] @ (L @ P[:, i]) for i in range(3))
    lam, Phi = sla.eigsh(c * L, k=K_MODES, M=M, sigma=-1e-8)
    o = np.argsort(lam)
    A = (sp.diags(1 / m) @ (c * L)).tocsr()
    u_pc = np.stack([sla.expm_multiply(-tj * A, U0.T).T for tj in t])
    return dict(M_pc=m, c_pc=c, lam_ex=lam[o], Phi_ex=Phi[:, o].astype(np.float32), u_pc=u_pc.astype(np.float32))


def build(task):
    kind, sid = task                                  # kind: "test" (T0-T2) or "heldout" (T3)
    if kind == "test":
        idx = pd.read_csv(os.path.join(DATA, "index.csv")).set_index("sid")
        fam = idx.loc[sid, "family"]
        g = replay(fam, np.random.default_rng([SEED, sid]))
        d = np.load(os.path.join(DATA, f"shape_{sid:03d}.npz"))
        assert np.abs(g["V"] - d["V"]).max() < 1e-5, f"replay mismatch for {sid}"
        spec = json.loads(idx.loc[sid, "ics"])
        t, lam1 = d["t"], float(d["lam1"])
        centers = [g["V"][s["centers"]] for s in spec]
        sets = ["T0", "T1", "T2"]
    else:
        fam = "dumbbell"
        rng = np.random.default_rng([SEED, 10000 + sid])
        g = replay(fam, rng)
        assert g["ok"], f"held-out {sid}: invalid mesh"
        L4, m4 = S.cotan_laplacian(g["V"], g["F"])
        lam1 = float(np.sort(sla.eigsh(L4.tocsc(), k=2, M=sp.diags(m4).tocsc(), sigma=-1e-8)[0])[1])
        t = np.array(TAUS) / lam1
        spec = []
        for _ in range(4):                            # same IC recipe as make_dataset.initial_conditions
            ng = rng.integers(1, 4)
            spec.append(dict(centers=rng.choice(N_VERT, ng, replace=False).tolist(),
                             sigma=rng.uniform(0.15, 0.5, ng).tolist(), amp=rng.uniform(0.5, 1.5, ng).tolist()))
        centers = [g["V"][s["centers"]] for s in spec]
        sets = ["T3"]

    t0 = time.perf_counter()
    Vf, Ff = fine_mesh(g)
    Lf, mf = S.cotan_laplacian(Vf, Ff)
    Uf = crank_nicolson(Lf.tocsc(), mf, gaussians(Vf, spec, centers).T, t)          # (T, Nf, I)
    t_gt = time.perf_counter() - t0
    tree_f = None
    rows = []
    for name in sets:
        rng = np.random.default_rng([SEED, 777, sid, sets.index(name)])
        if name in ("T0", "T3"):
            P = g["V"]
            I = sp.csr_matrix((np.ones(N_VERT), (np.arange(N_VERT), np.arange(N_VERT))), shape=(N_VERT, len(Vf)))
        else:
            if name == "T1":
                P = sample_surface(Vf, Ff, g, N_PTS, rng)
            else:
                n_dir = rng.normal(size=3)
                n_dir /= np.linalg.norm(n_dir)
                P = sample_surface(Vf, Ff, g, N_PTS, rng,
                                   weight=lambda X: np.where(X @ n_dir > 0, 1.0, 1 / DENSITY_RATIO))
            I = interp_matrix(Vf, Ff, P)
        # evaluation weights: fine lumped mass -> nearest point
        nearest = cKDTree(P).query(Vf)[1]
        w = np.bincount(nearest, weights=mf, minlength=len(P))
        U = np.stack([(I @ Uf[k]).T for k in range(len(t))])                       # (T, I, N)
        U0 = gaussians(P, spec, centers)                                             # exact IC at points
        pc = pc_side(P, U0, t)
        os.makedirs(os.path.join(OUT, name), exist_ok=True)
        np.savez(os.path.join(OUT, name, f"shape_{sid:03d}.npz"), V=P.astype(np.float32), m_mesh=w,
                 U0=U0.astype(np.float32), U=U.astype(np.float32), t=t, lam1=lam1, **pc)
        rows.append(dict(set=name, sid=sid, family=fam, t_gt_s=t_gt, n_fine=len(Vf),
                         pc_mass_over_area=pc["M_pc"].sum() / AREA, c_pc=pc["c_pc"],
                         min_nn_dist=float(cKDTree(P).query(P, 2)[0][:, 1].min())))
    return rows


def validate():
    """CN on a fine mesh vs expm_multiply (one level of subdivision, sid of the first test shape)."""
    split = pd.read_csv(os.path.join(DATA, "split.csv"))
    sid = int(split[split.split == "test"].sid.iloc[0])
    idx = pd.read_csv(os.path.join(DATA, "index.csv")).set_index("sid")
    g = replay(idx.loc[sid, "family"], np.random.default_rng([SEED, sid]))
    d = np.load(os.path.join(DATA, f"shape_{sid:03d}.npz"))
    print("replay max |ΔV| vs stored:", np.abs(g["V"] - d["V"]).max())
    Vf, Ff = fine_mesh(g, levels=1)
    L, m = S.cotan_laplacian(Vf, Ff)
    spec = json.loads(idx.loc[sid, "ics"])
    U0 = gaussians(Vf, spec, [g["V"][s["centers"]] for s in spec]).T
    Ucn = crank_nicolson(L.tocsc(), m, U0, d["t"])
    A = (sp.diags(1 / m) @ L).tocsr()
    for k in (0, 3, 7):
        ex = sla.expm_multiply(-d["t"][k] * A, U0)
        e = np.sqrt((m[:, None] * (Ucn[k] - ex) ** 2).sum(0) / (m[:, None] * ex ** 2).sum(0))
        print(f"tau idx {k}: CN vs expm_multiply rel err max {e.max():.2e}")
    # fine (2 levels) vs 4k-mesh ground truth at the 4k vertices
    Vf2, Ff2 = fine_mesh(g, levels=2)
    L2, m2 = S.cotan_laplacian(Vf2, Ff2)
    U2 = crank_nicolson(L2.tocsc(), m2, gaussians(Vf2, spec, [g["V"][s["centers"]] for s in spec]).T, d["t"])
    for k in range(len(d["t"])):
        a, b = U2[k][:N_VERT].T, d["U"][k]
        e = np.sqrt((d["m_mesh"] * (a - b) ** 2).sum(1) / (d["m_mesh"] * b ** 2).sum(1))
        print(f"tau {TAUS[k]:.3g}: fine(64k) vs stage-1 4k mesh GT at 4k vertices, rel diff mean {e.mean():.2e}")


def main():
    split = pd.read_csv(os.path.join(DATA, "split.csv"))
    test = split[split.split == "test"].sid.astype(int).tolist()
    done = lambda st, s: os.path.exists(os.path.join(OUT, st, f"shape_{s:03d}.npz"))
    tasks = [("test", s) for s in test if not (done("T1", s) and done("T2", s))]
    tasks += [("heldout", j) for j in range(N_HELDOUT) if not done("T3", j)]
    t0 = time.time()
    with Pool(16) as p:
        rows = sum(p.map(build, tasks, chunksize=1), [])
    have = {(r["set"], r["sid"]) for r in rows}
    fam = pd.read_csv(os.path.join(DATA, "index.csv")).set_index("sid").family
    for st in ("T0", "T1", "T2", "T3"):
        for f in sorted(os.listdir(os.path.join(OUT, st))):
            sid = int(f[6:9])
            if (st, sid) not in have:
                d = np.load(os.path.join(OUT, st, f))
                rows.append(dict(set=st, sid=sid, family="dumbbell" if st == "T3" else fam[sid],
                                 pc_mass_over_area=d["M_pc"].sum() / AREA, c_pc=float(d["c_pc"])))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "index.csv"), index=False)
    print(f"P0 data done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    validate() if "--validate" in sys.argv else main()

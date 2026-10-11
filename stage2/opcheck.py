"""Stage 2, decisions 1-2: validation of point-based volume Laplacians (CPU only) -> stage2/results_opcheck/

Domains (all Neumann):
  ball    unit ball                        analytic: j_l'(k) = 0, eigenfunctions j_l(kr) Y_lm
  cube    [-1/2, 1/2]^3                    analytic: pi^2 (a^2+b^2+c^2), products of cosines
  shell   0.5 <= r <= 1 (concave cavity)   analytic: j_l'(ka) y_l'(kb) - j_l'(kb) y_l'(ka) = 0
  torus   solid torus R=0.7, r=0.3         reference: fine tet P1 FEM (gmsh), eigenvalues only
  cup     open cup: cylinder r=1, z in [-.5,.5] minus cavity r=0.65, z in [-.2, top] (open concavity)
          reference: fine tet P1 FEM (gmsh), eigenvalues only
Operators (points only, + SDF of the domain for carving):
  delaunay        3D Delaunay of the points, tets removed if centroid SDF < 0 or any edge midpoint
                  SDF < -h/4 (h = local spacing), then P1 stiffness + lumped mass
  delaunay_nocarve  same without SDF removal (concave domains only; shows why carving is needed)
  (both Delaunay variants also drop boundary slivers: tets with >= 3 surface vertices and vol/l_max^3 < 1e-3)
  knn16 / knn32   graph Laplacian, self-tuning Gaussian weights w_ij = exp(-d^2/(s_i s_j)),
                  s_i = distance to the (k/2)-th neighbour; mass m_i = (4/3) pi d_k,i^3 / k
All operators are rescaled by c3 = 3 sum(M) / sum_i x_i^T L x_i (linear precision in 3D).
Sampling: "random" (iid in volume + iid on the surface) and "quasi" (jittered lattice inside + FPS on
the surface); surface point count matched to the interior spacing.
Metrics: sum(M)/V, low-mode eigenvalue bias (i = 1..10, before/after c3), eigenvalue error up to K_ref,
span loss of the first K_ref reference modes (K_ref = largest complete degeneracy group count <= 32),
fraction of negative cotan weights, carved-tet fraction.

    python stage2/opcheck.py [--quick]        (env neo)
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["HOME"] = os.path.join(ROOT, ".gmsh_home")          # gmsh/FLTK prefs stay in the project
os.makedirs(os.environ["HOME"], exist_ok=True)
for var, sub in [("MPLCONFIGDIR", "mpl"), ("XDG_CACHE_HOME", ""), ("TMPDIR", "tmp")]:
    os.environ[var] = os.path.join(ROOT, ".cache", sub)
    os.makedirs(os.environ[var], exist_ok=True)
os.environ.setdefault("OMP_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import brentq
from scipy.spatial import Delaunay, cKDTree
from scipy.special import sph_harm_y, spherical_jn, spherical_yn

OUT = os.path.join(ROOT, "results_opcheck")
NS = [2000, 4000, 8000, 16000, 32000]
SEEDS = [0, 1, 2]
K_MAX_REF = 32
TORUS_R, TORUS_r, SHELL_A = 0.7, 0.3, 0.5
CUP_RI, CUP_ZB = 0.65, -0.2                 # cavity radius, cavity floor height


# ------------------------------------------------------------------ domains --
def sdf(dom, x):
    r = np.linalg.norm(x, axis=1)
    if dom == "ball":
        return 1 - r
    if dom == "cube":
        return np.min(0.5 - np.abs(x), axis=1)
    if dom == "shell":
        return np.minimum(1 - r, r - SHELL_A)
    if dom == "torus":
        return TORUS_r - np.sqrt((np.linalg.norm(x[:, :2], axis=1) - TORUS_R) ** 2 + x[:, 2] ** 2)
    if dom == "cup":
        rho = np.linalg.norm(x[:, :2], axis=1)
        outer = np.minimum(1 - rho, 0.5 - np.abs(x[:, 2]))
        zc, hh = (CUP_ZB + 0.7) / 2, (0.7 - CUP_ZB) / 2           # cavity extends above the rim (open)
        cavity = np.minimum(CUP_RI - rho, hh - np.abs(x[:, 2] - zc))
        return np.minimum(outer, -cavity)


VOL = {"ball": 4 / 3 * np.pi, "cube": 1.0, "shell": 4 / 3 * np.pi * (1 - SHELL_A ** 3),
       "torus": 2 * np.pi ** 2 * TORUS_R * TORUS_r ** 2,
       "cup": np.pi - np.pi * CUP_RI ** 2 * (0.5 - CUP_ZB)}
AREA = {"ball": 4 * np.pi, "cube": 6.0, "shell": 4 * np.pi * (1 + SHELL_A ** 2),
        "torus": 4 * np.pi ** 2 * TORUS_R * TORUS_r,
        "cup": 2 * np.pi + np.pi + np.pi * (1 - CUP_RI ** 2) + 2 * np.pi * CUP_RI * (0.5 - CUP_ZB) + np.pi * CUP_RI ** 2}
BOX = {"ball": 1.0, "cube": 0.5, "shell": 1.0, "torus": TORUS_R + TORUS_r, "cup": 1.0}


def surface_random(dom, n, rng):
    if dom in ("ball", "shell"):
        if dom == "ball":
            radii = np.ones(n)
        else:
            radii = np.where(rng.uniform(size=n) < 1 / (1 + SHELL_A ** 2), 1.0, SHELL_A)
        d = rng.normal(size=(n, 3))
        return d / np.linalg.norm(d, axis=1, keepdims=True) * radii[:, None]
    if dom == "cube":
        face = rng.integers(0, 6, n)
        p = rng.uniform(-0.5, 0.5, (n, 3))
        ax, sgn = face // 2, np.where(face % 2 == 0, -0.5, 0.5)
        p[np.arange(n), ax] = sgn
        return p
    if dom == "cup":
        parts = [("side", 2 * np.pi), ("bottom", np.pi), ("rim", np.pi * (1 - CUP_RI ** 2)),
                 ("iside", 2 * np.pi * CUP_RI * (0.5 - CUP_ZB)), ("floor", np.pi * CUP_RI ** 2)]
        w = np.array([a for _, a in parts])
        which = rng.choice(len(parts), n, p=w / w.sum())
        P = np.empty((n, 3))
        for k, (name, _) in enumerate(parts):
            sel = which == k
            c = int(sel.sum())
            th = rng.uniform(0, 2 * np.pi, c)
            if name == "side":
                P[sel] = np.c_[np.cos(th), np.sin(th), rng.uniform(-0.5, 0.5, c)]
            elif name == "iside":
                P[sel] = np.c_[CUP_RI * np.cos(th), CUP_RI * np.sin(th), rng.uniform(CUP_ZB, 0.5, c)]
            else:
                lo, hi, z = {"bottom": (0, 1, -0.5), "rim": (CUP_RI, 1, 0.5), "floor": (0, CUP_RI, CUP_ZB)}[name]
                r = np.sqrt(rng.uniform(lo ** 2, hi ** 2, c))
                P[sel] = np.c_[r * np.cos(th), r * np.sin(th), np.full(c, z)]
        return P
    if dom == "torus":
        out = []
        while sum(len(o) for o in out) < n:
            u, v = rng.uniform(0, 2 * np.pi, (2, 4 * n))
            keep = rng.uniform(0, TORUS_R + TORUS_r, 4 * n) < TORUS_R + TORUS_r * np.cos(v)
            u, v = u[keep], v[keep]
            out.append(np.stack([(TORUS_R + TORUS_r * np.cos(v)) * np.cos(u),
                                 (TORUS_R + TORUS_r * np.cos(v)) * np.sin(u), TORUS_r * np.sin(v)], 1))
        return np.concatenate(out)[:n]


def fps(P, n, rng):
    idx = np.empty(n, np.int64)
    idx[0] = rng.integers(len(P))
    d = ((P - P[idx[0]]) ** 2).sum(1)
    for i in range(1, n):
        idx[i] = np.argmax(d)
        d = np.minimum(d, ((P - P[idx[i]]) ** 2).sum(1))
    return P[idx]


def sample(dom, N, kind, rng):
    """Interior + surface points; surface spacing matched to interior spacing."""
    V, S = VOL[dom], AREA[dom]
    h = (V / N) ** (1 / 3)
    for _ in range(3):
        n_s = int(round(S / h ** 2))
        h = (V / (N - n_s)) ** (1 / 3)
    n_i = N - n_s
    b = BOX[dom]
    if kind == "random":
        pts = []
        while sum(len(p) for p in pts) < n_i:
            x = rng.uniform(-b, b, (4 * n_i, 3))
            pts.append(x[sdf(dom, x) > 0])
        inner = np.concatenate(pts)[:n_i]
        surf = surface_random(dom, n_s, rng)
    else:
        g = np.arange(-b + h / 2, b, h)
        X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        X = X + rng.uniform(-0.35 * h, 0.35 * h, X.shape)
        inner = X[sdf(dom, X) > 0.35 * h]
        surf = fps(surface_random(dom, 10 * n_s, rng), n_s, rng)
    P = np.concatenate([inner, surf])
    return P, np.r_[np.zeros(len(inner), bool), np.ones(len(surf), bool)], h


# --------------------------------------------------------------- operators --
def tet_p1(P, T):
    """P1 stiffness (K) and lumped mass for tets T; also the fraction of positive off-diagonals
    of K (= negative cotan weights)."""
    D = np.stack([P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]], P[T[:, 3]] - P[T[:, 0]]], -1)
    det = np.linalg.det(D)
    vol = np.abs(det) / 6
    ok = vol > 1e-14 * np.median(vol)
    D, vol, T = D[ok], vol[ok], T[ok]
    G = np.transpose(np.linalg.inv(D), (0, 2, 1))
    G = np.concatenate([-G.sum(2, keepdims=True), G], 2)                       # (T, 3, 4) gradients
    Kl = vol[:, None, None] * np.einsum("tki,tkj->tij", G, G)
    n = len(P)
    K = sp.coo_matrix((Kl.ravel(), (np.repeat(T, 4, 1).ravel(), np.tile(T, (1, 4)).ravel())), shape=(n, n)).tocsr()
    m = np.zeros(n)
    np.add.at(m, T.ravel(), np.repeat(vol / 4, 4))
    off = K - sp.diags(K.diagonal())
    neg_w = float((off.data > 1e-14).mean()) if off.nnz else 0.0
    return K, m, neg_w, len(T)


SLIVER_Q = 1e-3          # quality vol / (longest edge)^3 below which a boundary tet counts as a sliver


def op_delaunay(P, dom, h, is_surface, carve=True):
    """Delaunay-P1.  (1) SDF carving (optional) removes tets filling concavities;
    (2) boundary slivers are always removed: tets with >= 3 surface vertices and quality
    vol/l_max^3 < SLIVER_Q.  They lie flat on the boundary (4 surface points on a flat or gently curved
    patch, or 3 surface points + an interior point very close to the face, which random interior
    sampling produces), have ~zero volume and near-singular gradients (found on the cube, torus, cup)."""
    T = Delaunay(P).simplices
    n_all = len(T)
    if carve:
        cen = P[T].mean(1)
        keep = sdf(dom, cen) >= 0
        for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            keep &= sdf(dom, (P[T[:, a]] + P[T[:, b]]) / 2) >= -0.25 * h
        T = T[keep]
    n_carved = n_all - len(T)
    D = np.stack([P[T[:, k]] - P[T[:, 0]] for k in (1, 2, 3)], -1)
    vol = np.abs(np.linalg.det(D)) / 6
    lmax = np.max([np.linalg.norm(P[T[:, a]] - P[T[:, b]], axis=1)
                   for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]], 0)
    sliver = (is_surface[T].sum(1) >= 3) & (vol / lmax ** 3 < SLIVER_Q)
    T = T[~sliver]
    K, m, neg_w, n_t = tet_p1(P, T)
    return K, m, dict(neg_weight_frac=neg_w, carved_frac=n_carved / n_all, sliver_frac=float(sliver.mean()))


def op_knn(P, k):
    tree = cKDTree(P)
    d, j = tree.query(P, k + 1)
    d, j = d[:, 1:], j[:, 1:]
    s = d[:, k // 2 - 1]
    w = np.exp(-d ** 2 / (s[:, None] * s[j]))
    n = len(P)
    W = sp.coo_matrix((w.ravel(), (np.repeat(np.arange(n), k), j.ravel())), shape=(n, n)).tocsr()
    W = 0.5 * (W + W.T)
    L = sp.diags(np.asarray(W.sum(1)).ravel()) - W
    m = 4 / 3 * np.pi * d[:, -1] ** 3 / k
    return L.tocsr(), m, dict(neg_weight_frac=0.0, carved_frac=0.0)


# --------------------------------------------------------- analytic refs ----
def _roots(f, kmax=22.0, step=2e-3):
    ks = np.arange(step, kmax, step)
    v = f(ks)
    idx = np.where(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]
    return [brentq(f, ks[i], ks[i + 1]) for i in idx]


def _real_sh(l, x):
    r = np.linalg.norm(x, axis=1)
    theta = np.arccos(np.clip(x[:, 2] / np.maximum(r, 1e-15), -1, 1))
    phi = np.arctan2(x[:, 1], x[:, 0])
    out = []
    for m in range(0, l + 1):
        y = sph_harm_y(l, m, theta, phi)
        out.append(y.real)
        if m > 0:
            out.append(y.imag)
    return np.stack(out, 1)                                                    # (n, 2l+1)


def reference(dom):
    """List of (lambda, multiplicity, fn(x) -> (n, mult))."""
    modes = [(0.0, 1, lambda x: np.ones((len(x), 1)))]
    if dom == "cube":
        trip = {}
        for a in range(6):
            for b in range(6):
                for c in range(6):
                    trip.setdefault(a * a + b * b + c * c, []).append((a, b, c))
        for s in sorted(trip)[1:]:
            abc = trip[s]
            modes.append((np.pi ** 2 * s, len(abc),
                          lambda x, abc=abc: np.stack([np.cos(a * np.pi * (x[:, 0] + .5)) * np.cos(b * np.pi * (x[:, 1] + .5))
                                                       * np.cos(c * np.pi * (x[:, 2] + .5)) for a, b, c in abc], 1)))
    elif dom in ("ball", "shell"):
        for l in range(0, 12):
            if dom == "ball":
                ks = _roots(lambda k: spherical_jn(l, k, derivative=True))
                rad = lambda x, k, l=l: spherical_jn(l, k * np.linalg.norm(x, axis=1))
            else:
                a, b = SHELL_A, 1.0
                f = lambda k, l=l: (spherical_jn(l, k * a, True) * spherical_yn(l, k * b, True)
                                    - spherical_jn(l, k * b, True) * spherical_yn(l, k * a, True))
                ks = _roots(f)
                rad = (lambda x, k, l=l: spherical_jn(l, k * np.linalg.norm(x, axis=1)) * spherical_yn(l, k * a, True)
                       - spherical_yn(l, k * np.linalg.norm(x, axis=1)) * spherical_jn(l, k * a, True))
            for k in ks:
                if k < 1e-6:
                    continue
                modes.append((k * k, 2 * l + 1, lambda x, k=k, l=l, rad=rad: rad(x, k)[:, None] * _real_sh(l, x)))
    modes.sort(key=lambda t: t[0])
    return modes


def k_ref(modes, kmax=K_MAX_REF):
    c = 0
    for _, mult, _ in modes:
        if c + mult > kmax:
            break
        c += mult
    return c


_TORUS = {}


def torus_reference(h):
    return fem_reference("torus", h)


def fem_reference(dom, h):
    """Fine tet P1 FEM eigenvalues (gmsh) of the torus or the cup, cached on disk."""
    path = os.path.join(OUT, f"{dom}_ref_h{h:g}.npy")
    if os.path.exists(path):
        return np.load(path)
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    if dom == "torus":
        gmsh.model.occ.addTorus(0, 0, 0, TORUS_R, TORUS_r)
    else:
        outer = gmsh.model.occ.addCylinder(0, 0, -0.5, 0, 0, 1.0, 1.0)
        cavity = gmsh.model.occ.addCylinder(0, 0, CUP_ZB, 0, 0, 0.8 - CUP_ZB, CUP_RI)
        gmsh.model.occ.cut([(3, outer)], [(3, cavity)])
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.model.mesh.generate(3)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    _, conn = gmsh.model.mesh.getElementsByType(4)
    gmsh.finalize()
    P = coords.reshape(-1, 3)
    lut = np.empty(tags.max() + 1, np.int64)
    lut[tags] = np.arange(len(tags))
    T = lut[conn.reshape(-1, 4)]
    used, T = np.unique(T, return_inverse=True)
    T = T.reshape(-1, 4)
    K, m, _, _ = tet_p1(P[used], T)
    lam = np.sort(sla.eigsh(K.tocsc(), k=48, M=sp.diags(m).tocsc(), sigma=-1e-6)[0])
    np.save(path, lam)
    np.save(os.path.join(OUT, f"{dom}_ref_h{h:g}_nverts.npy"), np.array([len(used)]))
    return lam


# ------------------------------------------------------------------- task ---
def task(args):
    dom, kind, N, seed, op = args
    rng = np.random.default_rng([2026, ["ball", "cube", "shell", "torus", "cup"].index(dom), 0 if kind == "random" else 1, N, seed])
    P, is_s, h = sample(dom, N, kind, rng)
    t0 = time.perf_counter()
    if op.startswith("delaunay"):
        L, m, info = op_delaunay(P, dom, h, is_s, carve=(op == "delaunay"))
    else:
        L, m, info = op_knn(P, int(op[3:]))
    iso = m <= 0
    if iso.any():                                          # points with no remaining tet: drop them
        keep = ~iso
        L = L[keep][:, keep]
        m, P, is_s = m[keep], P[keep], is_s[keep]
    c3 = 3 * m.sum() / sum(P[:, i] @ (L @ P[:, i]) for i in range(3))
    if dom in ("torus", "cup"):
        lam_ref = fem_reference(dom, 0.03)
        kr = 32
        modes = None
    else:
        modes = reference(dom)
        kr = k_ref(modes)
        lam_ref = np.concatenate([[lam] * mult for lam, mult, _ in modes])[: kr + 8]
    ke = min(kr + 8, len(lam_ref))
    lam, Phi = sla.eigsh(L.tocsc(), k=ke, M=sp.diags(m).tocsc(), sigma=-1e-6)
    o = np.argsort(lam)
    lam, Phi = lam[o], Phi[:, o]
    t_op = time.perf_counter() - t0
    rel_raw = lam[1:kr] / lam_ref[1:kr] - 1
    rel_cal = c3 * lam[1:kr] / lam_ref[1:kr] - 1
    row = dict(domain=dom, sampling=kind, N=N, seed=seed, op=op, n_points=len(P), n_isolated=int(iso.sum()),
               surface_frac=float(is_s.mean()), mass_ratio=m.sum() / VOL[dom], c3=c3, K_ref=kr,
               low_bias_raw=float(rel_raw[:10].mean()), low_bias_cal=float(rel_cal[:10].mean()),
               abs_err_cal_1toK=float(np.abs(rel_cal).mean()), max_abs_err_cal_1toK=float(np.abs(rel_cal).max()),
               t_s=t_op, **info)
    if modes is not None:
        U = np.concatenate([fn(P) for _, _, fn in modes], 1)[:, :kr]
        G = U.T @ (m[:, None] * U)
        Lc = np.linalg.cholesky(0.5 * (G + G.T))
        U = np.linalg.solve(Lc, U.T).T                     # M-orthonormal reference basis on the points
        row["span_loss"] = float(1 - ((Phi[:, :kr].T @ (m[:, None] * U)) ** 2).sum() / kr)
    return row


def main(quick=False, domains=("ball", "cube", "shell", "torus")):
    os.makedirs(OUT, exist_ok=True)
    if domains == ("cup",):
        for h in (0.03, 0.04):
            fem_reference("cup", h)
        r3, r4 = fem_reference("cup", 0.03), fem_reference("cup", 0.04)
        print("  cup ref rel. diff h=0.04 vs 0.03 (i=1..31):", np.round(np.abs(r4[1:32] / r3[1:32] - 1).mean(), 5),
              " n_verts h=0.03:", int(np.load(os.path.join(OUT, "cup_ref_h0.03_nverts.npy"))[0]), flush=True)
    if "torus" in domains:
      print("torus reference (gmsh, h=0.03 and 0.04) ...", flush=True)
      ref3, ref4 = torus_reference(0.03), torus_reference(0.04)
      print("  torus ref rel. diff h=0.04 vs 0.03 (i=1..31):", np.round(np.abs(ref4[1:32] / ref3[1:32] - 1).mean(), 5),
            " n_verts h=0.03:", int(np.load(os.path.join(OUT, "torus_ref_h0.03_nverts.npy"))[0]), flush=True)
    ns, seeds = ([2000, 8000], [0]) if quick else (NS, SEEDS)
    tasks = []
    for dom in domains:
        ops = ["delaunay", "knn16", "knn32"] + (["delaunay_nocarve"] if dom in ("shell", "torus", "cup") else [])
        for kind in ["quasi", "random"]:
            for N in ns:
                for s in seeds:
                    for op in ops:
                        tasks.append((dom, kind, N, s, op))
    tasks.sort(key=lambda t: -t[2])
    t0 = time.time()
    with Pool(20) as p:
        rows = list(p.imap_unordered(task, tasks, chunksize=1))
    df = pd.DataFrame(rows).sort_values(["domain", "op", "sampling", "N", "seed"])
    name = "opcheck_quick.csv" if quick else ("opcheck_cup.csv" if domains == ("cup",) else "opcheck.csv")
    df.to_csv(os.path.join(OUT, name), index=False)
    print(f"{len(df)} runs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv, domains=("cup",) if "--cup" in sys.argv else ("ball", "cube", "shell", "torus"))

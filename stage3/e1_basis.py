"""Stage 3 / E1: does "scalar basis x 3 components + 12 affine fields" contain the elastic modes and static
solutions?  (DESIGN.md s3a; decision-free part: no Dirichlet method / tool-patch BC is used here)

Operator   linear isotropic elasticity, plain P1 on the stage-2 Delaunay-P1 tets (volop), lumped mass M x I3,
           DOF order 3*i + c.  E = 1 (homogeneous) for the modes; the static test also uses stiff lumps (E = 5).
Bases      (all + 12 exact affine fields, (M x I3)-orthonormalised, Rayleigh-Ritz with the homogeneous K(nu))
             arpack128x3, arpack256x3   scalar ARPACK eigenvectors replicated per component
             neo256x3                   scalar volume NEO (stage2 main/t4096) + scalar Rayleigh-Ritz, replicated
             elastic256                 exact elastic eigenvectors (ARPACK), the best modal basis of that size (reference)
Tests      (1) span loss of the exact elastic eigenvectors: 6 rigid modes, and the first complete groups up to 96
               non-rigid modes -> criterion <= 1e-3
           (2) static solution: free body under an equilibrated two-patch load (rigid-mode components removed);
               full-order P1 solution (KKT, rigid modes pinned) vs the Galerkin solution in the basis (the
               energy-best approximation) -> relative L2 (M) and energy error, criterion <= 1 %;
               homogeneous E and stiff lumps (E = 5, 1-2 lumps).  Loads:
                 pinch   uniform pressure on two opposite discs (radius 0.15, sharp edges)
                 smooth  Gaussian pressure (sigma 0.15) centred at the same two points (no edge singularity)
               Diagnostic "+sc": the basis plus ONE static-correction vector u_hom = K_hom^+ f of that load
               (modal truncation augmentation), applied to the lump case (tests whether the load-local part,
               not the basis, is what the modes miss).
           Recorded for nu in {0.3, 0.45, 0.49}: reduced-model locking = growth of the static error with nu;
           also the smallest non-rigid eigenvalues of the full P1 operator (spurious near-zero modes check).
Shapes     2 shapes per family: sh_ball, superquadric, torus, cup, shell, elongated, multihole, c_tube (concave:
           torus, cup, shell cavity, multihole, c_tube), N = 8192 quasi-uniform, seeds disjoint from stage-2 sets.

    python stage3/e1_basis.py prep [--procs 8]                  shapes + scalar ARPACK 256 -> e1/cache/ (CPU)
    CUDA_VISIBLE_DEVICES=<gpu> python stage3/e1_basis.py neo    scalar NEO 256 + RR -> e1/cache/ (GPU; the repo's
                                                                 fused kernels are Triton/CUDA only)
    python stage3/e1_basis.py run [--nu 0.3] [--procs 8]        elastic tests (CPU)
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
S2 = os.path.join(os.path.dirname(ROOT), "stage2")
sys.path.insert(0, S2)
for var, sub in [("MPLCONFIGDIR", "mpl"), ("XDG_CACHE_HOME", ""), ("TMPDIR", "tmp"), ("TORCH_HOME", "torch"),
                 ("TRITON_CACHE_DIR", "triton")]:
    os.environ[var] = os.path.join(ROOT, ".cache", sub)
    os.makedirs(os.environ[var], exist_ok=True)

import open3d  # noqa: F401,E402  -- must precede h5py/torch/lightning in the neo env

from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.linalg as la  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as sla  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import neo_vol_data as D0  # noqa: E402
import neo_vol_data2 as D2  # noqa: E402
from volop import delaunay_p1  # noqa: E402

OUT = os.path.join(ROOT, "e1")
CACHE = os.path.join(OUT, "cache")
FAMILIES = ["sh_ball", "superquadric", "torus", "cup", "shell", "elongated", "multihole", "c_tube"]
N = 8192
SEED = 20260927 + 3                                     # stage-3 stream, disjoint from stage-2 data seeds
K_MODES = 96                                            # non-rigid elastic modes checked (complete groups)
CKPT = os.path.join(S2, "main", "t4096", "best.ckpt")
GROUP_TOL = 1e-3


# ---------------------------------------------------------------- geometry / operators
def build_shape(family, idx):
    rng = np.random.default_rng([SEED, FAMILIES.index(family), idx])
    sdf0, bound = D2.make_sdf(family, rng)
    R = Rotation.random(random_state=rng.integers(2 ** 31)).as_matrix()
    sdf = D0.rotated(sdf0, R)
    geo = D0.geometry(sdf, bound * 1.05, rng)
    for _ in range(5):
        P, is_s, h = D0.sample_points(sdf, geo, N, "quasi", rng)
        c = P.mean(0)
        s = 1.0 / np.abs(P - c).max()
        Y = (P - c) * s
        sdf_n = lambda y, s=s, c=c: s * sdf(y / s + c)
        op = delaunay_p1(Y, is_s, sdf_n, h * s)
        if op.info["isolated"] == 0 and len(Y) == N:
            return Y, op, sdf_n, rng
    raise RuntimeError("sampling failed")


def elastic_stiffness(op, E_e, nu):
    """Plain P1: K = sum_e vol_e B_e^T D(E_e, nu) B_e, DOF 3*i + c (Voigt xx yy zz yz xz xy, engineering shear)."""
    G, T, vol = op.grads, op.tets, op.vol                       # G[t, d, a] = d phi_a / d x_d
    nT = len(T)
    B = np.zeros((nT, 6, 12))
    for a in range(4):
        gx, gy, gz = G[:, 0, a], G[:, 1, a], G[:, 2, a]
        B[:, 0, 3 * a] = gx; B[:, 1, 3 * a + 1] = gy; B[:, 2, 3 * a + 2] = gz
        B[:, 3, 3 * a + 1] = gz; B[:, 3, 3 * a + 2] = gy
        B[:, 4, 3 * a] = gz; B[:, 4, 3 * a + 2] = gx
        B[:, 5, 3 * a] = gy; B[:, 5, 3 * a + 1] = gx
    lam, mu = nu / ((1 + nu) * (1 - 2 * nu)), 1 / (2 * (1 + nu))  # per unit E
    Dm = lam * np.outer([1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]) + mu * np.diag([2, 2, 2, 1, 1, 1])
    Ke = (E_e * vol)[:, None, None] * np.einsum("tia,ij,tjb->tab", B, Dm, B)
    dof = (3 * T[:, :, None] + np.arange(3)[None, None]).reshape(nT, 12)
    n = 3 * len(op.m)
    return sp.coo_matrix((Ke.ravel(), (np.repeat(dof, 12, 1).ravel(), np.tile(dof, (1, 12)).ravel())),
                         shape=(n, n)).tocsr()


def affine_fields(Y):
    """12 exact affine displacement fields (3 translations + 9 linear x_j e_c), DOF 3*i + c."""
    n = len(Y)
    F = []
    for c in range(3):
        f = np.zeros((n, 3)); f[:, c] = 1; F.append(f.ravel())
        for j in range(3):
            f = np.zeros((n, 3)); f[:, c] = Y[:, j]; F.append(f.ravel())
    return np.stack(F, 1)


def rigid_modes(Y):
    n = len(Y)
    R = []
    for c in range(3):
        f = np.zeros((n, 3)); f[:, c] = 1; R.append(f.ravel())
    for a, b in ((1, 2), (2, 0), (0, 1)):                      # rotations
        f = np.zeros((n, 3)); f[:, a] = -Y[:, b]; f[:, b] = Y[:, a]; R.append(f.ravel())
    return np.stack(R, 1)


def replicate(Phi):
    n, k = Phi.shape
    V = np.zeros((3 * n, 3 * k))
    for c in range(3):
        V[c::3, c::3] = Phi                                     # column 3*j + c = phi_j e_c
    return V


def m_orthonormal(V, m3, rtol=1e-10):
    Gm = V.T @ (m3[:, None] * V)
    w, U = la.eigh(0.5 * (Gm + Gm.T))
    keep = w > rtol * w.max()
    return V @ (U[:, keep] / np.sqrt(w[keep]))


def ritz(Q, K):
    A = Q.T @ (K @ Q)
    mu, W = la.eigh(0.5 * (A + A.T))
    return mu, Q @ W


def span_loss(Q, U, m3):
    """1 - ||Q^T M U||_F^2 / k for M-orthonormalised U (Q M-orthonormal)."""
    Gm = U.T @ (m3[:, None] * U)
    U = np.linalg.solve(np.linalg.cholesky(0.5 * (Gm + Gm.T)), U.T).T
    return float(1 - ((Q.T @ (m3[:, None] * U)) ** 2).sum() / U.shape[1])


def k_ref(lam, start, kmax, tol=GROUP_TOL):
    """Largest index <= start + kmax that ends a complete eigenvalue group."""
    b = [i for i in range(start + 1, len(lam)) if lam[i] - lam[i - 1] > tol * max(lam[i], 1e-12) and i <= start + kmax]
    return max(b)


# ---------------------------------------------------------------- loads
def boundary_faces(T, P):
    F = np.concatenate([T[:, [1, 2, 3]], T[:, [0, 3, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 1]]])
    opp = np.concatenate([T[:, 0], T[:, 1], T[:, 2], T[:, 3]])
    key = np.sort(F, 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    b = cnt[inv.ravel()] == 1
    F, opp = F[b], opp[b]
    nrm = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]) / 2      # area-weighted
    flip = ((P[F].mean(1) - P[opp]) * nrm).sum(1) < 0                          # make outward
    nrm[flip] *= -1
    return F, nrm


def pinch_load(Y, T, rng, rho=0.15):
    """Inward pressure on two opposite surface spots; rigid-mode components removed (equilibrated).
    Returns {"pinch": uniform pressure on discs of radius rho, "smooth": Gaussian pressure, sigma = rho}."""
    F, nrm = boundary_faces(T, Y)
    cen = Y[F].mean(1)
    d = rng.normal(size=3); d /= np.linalg.norm(d)
    p1, p2 = cen[np.argmax(cen @ d)], cen[np.argmin(cen @ d)]
    out = {}
    for name in ("pinch", "smooth"):
        f = np.zeros((len(Y), 3))
        for pc in (p1, p2):
            dist = np.linalg.norm(cen - pc, axis=1)
            w = (dist < rho).astype(float) if name == "pinch" else np.exp(-dist ** 2 / (2 * rho ** 2))
            for k in range(3):
                np.add.at(f, F[:, k], -w[:, None] * nrm / 3)                    # inward traction
        out[name] = equilibrate(f.ravel(), Y)
    return out


def equilibrate(f, Y):
    R = rigid_modes(Y)
    f -= R @ np.linalg.lstsq(R, f, rcond=None)[0]                               # l2-orthogonal to rigid modes
    return f / np.linalg.norm(f)


def full_solve(K, f, Y, m3):
    """K u = f on the free body; u M-orthogonal to the rigid modes (KKT)."""
    C = sp.csr_matrix((m3[:, None] * rigid_modes(Y)))
    A = sp.bmat([[K, C], [C.T, None]]).tocsc()
    return sla.spsolve(A, np.concatenate([f, np.zeros(6)]))[:len(f)]


def reduced_solve(Q, K, f, Y, m3):
    """Galerkin in span(Q) with the rigid-mode components constrained to zero (Q contains the rigid modes)."""
    A = Q.T @ (K @ Q)
    Cr = Q.T @ (m3[:, None] * rigid_modes(Y))                                  # (m, 6)
    KKT = np.block([[0.5 * (A + A.T), Cr], [Cr.T, np.zeros((6, 6))]])
    c = la.lstsq(KKT, np.concatenate([Q.T @ f, np.zeros(6)]))[0][:Q.shape[1]]
    return Q @ c


def errs(prefix, u, ur, Kc, m3):
    e = ur - u
    return {f"{prefix}_relL2": float(np.sqrt((m3 * e ** 2).sum() / (m3 * u ** 2).sum())),
            f"{prefix}_relE": float(np.sqrt(max(e @ (Kc @ e), 0) / (u @ (Kc @ u))))}


def lumps(Y, rng, n):
    cs, rs = [], []
    while len(cs) < n:
        c, r = Y[rng.integers(len(Y))], rng.uniform(0.12, 0.2)
        if all(np.linalg.norm(c - c2) > r + r2 + 0.05 for c2, r2 in zip(cs, rs)):
            cs.append(c); rs.append(r)
    return lambda x: 1 + 4 * np.clip(sum(1 / (1 + np.exp(-(r - np.linalg.norm(x - c, axis=1)) / 0.02))
                                         for c, r in zip(cs, rs)), 0, 1)


# ---------------------------------------------------------------- per shape
def neo_basis(Y, m, Ks):
    """Scalar volume NEO (t4096) on CPU -> M-orthonormal -> scalar Rayleigh-Ritz (256)."""
    import torch
    from g2pt.training.pretrain_tet import PretrainTetTraining
    from g2pt.utils.ortho_operations import qr_orthogonalization
    global _NEO
    if "_NEO" not in globals():
        _NEO = PretrainTetTraining.load_from_checkpoint(CKPT, weights_only=False).model.to("cuda").eval()
    with torch.no_grad():
        x = torch.tensor(Y, dtype=torch.float32, device="cuda")[None]
        mass = torch.tensor(m, dtype=torch.float32, device="cuda").view(1, -1, 1)
        S = qr_orthogonalization(_NEO(x, x, mass), mass)[0].double().cpu().numpy()
    A, B = S.T @ (Ks @ S), (S * m[:, None]).T @ S
    _, V = la.eigh(0.5 * (A + A.T), 0.5 * (B + B.T))
    return S @ V


class Op:
    """Operator rebuilt from the cache (fields used here: tets, grads, vol, m, K, info)."""

    def __init__(self, z):
        self.tets, self.grads, self.vol, self.m = z["tets"], z["grads"], z["vol"], z["m"]
        self.info = dict(sliver_frac=float(z["sliver"]))
        T, G, n = self.tets, self.grads, len(self.m)
        Ke = self.vol[:, None, None] * np.einsum("tki,tkj->tij", G, G)
        self.K = sp.coo_matrix((Ke.ravel(), (np.repeat(T, 4, 1).ravel(), np.tile(T, (1, 4)).ravel())), shape=(n, n)).tocsr()


def cache_path(family, idx):
    return os.path.join(CACHE, f"{family}_{idx}.npz")


def prep_task(args):
    family, idx = args
    t0 = time.time()
    Y, op, sdf, rng = build_shape(family, idx)
    ls, Ps = sla.eigsh(op.K.tocsc(), k=256, M=sp.diags(op.m).tocsc(), sigma=-1e-6)
    o = np.argsort(ls)
    np.savez(cache_path(family, idx), Y=Y, tets=op.tets, grads=op.grads, vol=op.vol, m=op.m,
             sliver=op.info["sliver_frac"], Ps=Ps[:, o], t_scalar=time.time() - t0)
    return family, idx, time.time() - t0


def neo_step(tasks):
    for family, idx in tasks:
        p = cache_path(family, idx)
        z = dict(np.load(p))
        t0 = time.time()
        z["Pn"] = neo_basis(z["Y"], z["m"], Op(z).K.tocsc())
        z["t_neo"] = time.time() - t0
        np.savez(p, **z)
        print(family, idx, f"neo {z['t_neo']:.2f}s", flush=True)


def task(args):
    family, idx, nus = args
    t0 = time.time()
    z = np.load(cache_path(family, idx))
    Y, op = z["Y"], Op(z)
    rng = np.random.default_rng([SEED, FAMILIES.index(family), idx, 1])
    m = op.m
    m3 = np.repeat(m, 3)
    Ps, Pn, t_scalar, t_neo = z["Ps"], z["Pn"], float(z["t_scalar"]), float(z["t_neo"])
    E_het = lumps(Y, rng, 1 + idx % 2)(Y[op.tets].mean(1))
    loads = pinch_load(Y, op.tets, rng)
    A12 = affine_fields(Y)
    raw = {"arpack128x3": replicate(Ps[:, :128]), "arpack256x3": replicate(Ps), "neo256x3": replicate(Pn)}
    rows = []
    for nu in nus:
        K = elastic_stiffness(op, np.ones(len(op.vol)), nu)
        Kh = elastic_stiffness(op, E_het, nu)
        te = time.time()
        le, Ue = sla.eigsh(K.tocsc(), k=6 + 256, M=sp.diags(m3).tocsc(), sigma=-1e-6)
        o = np.argsort(le); le, Ue = le[o], Ue[:, o]
        t_el = time.time() - te
        kr = k_ref(le, 6, K_MODES)
        n_rigid_ok = int((le[:6] < 1e-8 * le[6]).sum())
        u_full = {(ld, mat): full_solve(Kc, f, Y, m3) for ld, f in loads.items() for mat, Kc in (("hom", K), ("lumps", Kh))}
        bases = {k: ritz(m_orthonormal(np.hstack([V, A12]), m3), K)[1] for k, V in raw.items()}
        bases["elastic256"] = m_orthonormal(np.hstack([Ue[:, :6 + 256], A12]), m3)
        for bname, Q in bases.items():
            r = dict(family=family, idx=idx, nu=nu, basis=bname, dim=Q.shape[1], K_ref=kr - 6,
                     span_rigid=span_loss(Q, Ue[:, :6], m3), span_modes=span_loss(Q, Ue[:, 6:kr], m3),
                     span_modes32=span_loss(Q, Ue[:, 6:k_ref(le, 6, 32)], m3),
                     lam7=float(le[6]), lam7_over_lam12=float(le[6] / le[11]), rigid_zero=n_rigid_ok)
            for ld, f in loads.items():
                for mat, Kc in (("hom", K), ("lumps", Kh)):
                    r.update(errs(f"static_{ld}_{mat}", u_full[(ld, mat)], reduced_solve(Q, Kc, f, Y, m3), Kc, m3))
                Qsc = m_orthonormal(np.hstack([Q, u_full[(ld, "hom")][:, None]]), m3)          # + static correction
                r.update(errs(f"sc_{ld}_lumps", u_full[(ld, "lumps")], reduced_solve(Qsc, Kh, f, Y, m3), Kh, m3))
            rows.append(r)
        rows[-1]["t_elastic_eigs_s"] = t_el
    for r in rows:
        r.update(t_scalar_eigs_s=t_scalar, t_neo_s=t_neo, t_shape_s=time.time() - t0, sliver=op.info["sliver_frac"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "neo", "run"])
    ap.add_argument("--nu", default="0.3")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--per-family", type=int, default=2)
    a = ap.parse_args()
    nus = [float(v) for v in a.nu.split(",")]
    os.makedirs(CACHE, exist_ok=True)
    shapes = [(f, i) for f in FAMILIES for i in range(a.per_family)]
    if a.step == "prep":
        with Pool(a.procs) as pool:
            for f, i, dt in pool.imap_unordered(prep_task, shapes):
                print(f, i, f"prep {dt:.0f}s", flush=True)
        return
    if a.step == "neo":
        neo_step(shapes)
        return
    tasks = [(f, i, nus) for f, i in shapes]
    t0 = time.time()
    rows = []
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(task, tasks):
            rows += r
            print(pd.DataFrame(r)[["family", "idx", "nu", "basis", "dim", "span_modes", "static_pinch_hom_relL2",
                                   "static_smooth_hom_relL2", "sc_pinch_lumps_relL2"]].to_string(header=False,
                  float_format=lambda v: f"{v:.2e}"), flush=True)
    tag = "nu" + "_".join(f"{v:g}" for v in nus)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"e1_{tag}.csv"), index=False)
    cols = ["span_rigid", "span_modes32", "span_modes"] + [c for c in df.columns if c.startswith(("static_", "sc_")) and c.endswith("relL2")]
    summ = df.groupby(["nu", "basis"])[cols].agg(["mean", "max"])
    open(os.path.join(OUT, f"e1_{tag}_summary.txt"), "w").write(summ.to_string(float_format=lambda v: f"{v:.2e}")
                                                                + f"\n\n{len(tasks)} shapes in {time.time() - t0:.0f}s\n")
    print(summ.to_string(float_format=lambda v: f"{v:.2e}"))


if __name__ == "__main__":
    main()

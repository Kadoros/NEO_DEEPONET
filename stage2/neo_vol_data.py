"""NEO volume pilot data (decision 3) -> stage2/data_neo/{train,test_in,test_out}/

Families (procedural solids, random rotation):
  train / test_in : sh_ball (spherical-harmonic perturbed ball), superquadric solid, solid torus (r/R 0.25-0.55),
                    open cup (concave), offset shell (closed cavity)
  test_out        : c_tube (bent tube sweeping 200-320 deg with round caps; concave) -- never in training
Per shape and sample: points = quasi-uniform (jittered lattice inside + FPS on the surface) or random
(iid inside + area-uniform on the surface), surface count matched to the interior spacing, then normalised
the way NEO's normalize_pc does it (centre = mean, max |coord| = 1).  Operator = Delaunay-P1 with SDF
carving + boundary-sliver removal (volop.delaunay_p1, validated in REPORT_opcheck.md) built directly in the
normalised coordinates; targets = its 96 lowest eigenvectors (M-orthonormal), mass = its lumped mass.
Files follow the NEO pretrain_tet layout (*_mesh.hdf5 + *_samples_evecs.hdf5).

    python stage2/neo_vol_data.py --estimate        # timing on a few shapes only
    python stage2/neo_vol_data.py                   # full generation
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
for var, sub in [("MPLCONFIGDIR", "mpl"), ("XDG_CACHE_HOME", ""), ("TMPDIR", "tmp")]:
    os.environ[var] = os.path.join(ROOT, ".cache", sub)
    os.makedirs(os.environ[var], exist_ok=True)
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "stage1"))

from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import shapes as S1                 # stage-1 radius functions (sh_sphere_radius, superquadric_radius)
from volop import delaunay_p1

OUT = os.path.join(ROOT, "data_neo")
SEED = 20260926
K = 96
TRAIN_FAMILIES = ["sh_ball", "superquadric", "torus", "cup", "shell"]
TEST_FAMILY = "c_tube"
SPLIT = {"train": 500, "test_in": 20, "test_out": 60}     # shapes per family (train/test_in) or total (test_out)
PER_SHAPE = [("quasi", 2048), ("quasi", 2048), ("random", 2048), ("random", 2048)]
TEST_SAMPLES = [("quasi", 2048), ("random", 2048), ("quasi", 8192), ("random", 8192)]


# ------------------------------------------------------------------ shapes --
def make_sdf(family, rng):
    """Returns sdf(x) (> 0 inside, canonical frame) and a bounding radius."""
    if family in ("sh_ball", "superquadric"):
        radius, _ = (S1.sh_sphere_radius if family == "sh_ball" else S1.superquadric_radius)(rng)

        def sdf(x):
            r = np.linalg.norm(x, axis=1)
            d = x / np.maximum(r, 1e-12)[:, None]
            return radius(d) - r
        return sdf, 1.9
    if family == "torus":
        R, r = 1.0, rng.uniform(0.25, 0.55)
        return (lambda x: r - np.sqrt((np.linalg.norm(x[:, :2], axis=1) - R) ** 2 + x[:, 2] ** 2)), R + r
    if family == "cup":
        H, a, depth = rng.uniform(0.8, 1.4), rng.uniform(0.45, 0.8), rng.uniform(0.4, 0.8)
        zb = H / 2 - depth * H
        zc, hh = (zb + H / 2 + 0.3) / 2, (H / 2 + 0.3 - zb) / 2

        def sdf(x):
            rho = np.linalg.norm(x[:, :2], axis=1)
            outer = np.minimum(1 - rho, H / 2 - np.abs(x[:, 2]))
            cav = np.minimum(a - rho, hh - np.abs(x[:, 2] - zc))
            return np.minimum(outer, -cav)
        return sdf, np.sqrt(1 + (H / 2) ** 2)
    if family == "shell":
        ax = rng.uniform(0.8, 1.2, 3)
        ri, off = rng.uniform(0.3, 0.5), rng.normal(size=3) * 0.08

        def sdf(x):
            r = np.linalg.norm(x, axis=1)
            d = x / np.maximum(r, 1e-12)[:, None]
            r_out = 1 / np.sqrt(((d / ax) ** 2).sum(1))
            return np.minimum(r_out - r, np.linalg.norm(x - off, axis=1) - ri)
        return sdf, 1.25
    if family == "c_tube":
        Rc, r, th = 1.0, rng.uniform(0.2, 0.4), np.deg2rad(rng.uniform(200, 320))
        e0, e1 = np.array([Rc, 0, 0]), np.array([Rc * np.cos(th), Rc * np.sin(th), 0])

        def sdf(x):
            phi = np.mod(np.arctan2(x[:, 1], x[:, 0]), 2 * np.pi)
            on_arc = phi <= th
            d_arc = np.sqrt((np.linalg.norm(x[:, :2], axis=1) - Rc) ** 2 + x[:, 2] ** 2)
            d_end = np.minimum(np.linalg.norm(x - e0, axis=1), np.linalg.norm(x - e1, axis=1))
            return r - np.where(on_arc, d_arc, d_end)
        return sdf, Rc + r
    raise ValueError(family)


def rotated(sdf, R):
    return lambda y: sdf(y @ R)                    # y = p R^T  <=>  p = y R


def grad(sdf, x, eps=1e-4):
    g = np.zeros_like(x)
    for i in range(3):
        dx = np.zeros(3); dx[i] = eps
        g[:, i] = (sdf(x + dx) - sdf(x - dx)) / (2 * eps)
    return g


def geometry(sdf, bound, rng):
    """Tight box, volume, area, and surface candidates (band points projected onto the zero set)."""
    X = rng.uniform(-bound, bound, (200_000, 3))
    ins = X[sdf(X) > 0]
    lo, hi = ins.min(0) - 0.05, ins.max(0) + 0.05
    box = np.prod(hi - lo)
    X = rng.uniform(lo, hi, (400_000, 3))
    s = sdf(X)
    V = box * (s > 0).mean()
    eps = 0.02
    band = X[np.abs(s) < eps]
    g = grad(sdf, band)
    gn = np.linalg.norm(g, axis=1)
    A = box * gn.sum() / len(X) / (2 * eps)
    C = band.copy()
    for _ in range(4):                               # Newton projection onto sdf = 0
        gc = grad(sdf, C)
        C = C - (sdf(C) / np.maximum((gc ** 2).sum(1), 1e-12))[:, None] * gc
    ok = np.abs(sdf(C)) < 1e-6
    return dict(lo=lo, hi=hi, V=V, A=A, cand=C[ok], cand_w=gn[ok])


def fps(P, n, rng):
    idx = np.empty(n, np.int64)
    idx[0] = rng.integers(len(P))
    d = ((P - P[idx[0]]) ** 2).sum(1)
    for i in range(1, n):
        idx[i] = np.argmax(d)
        d = np.minimum(d, ((P - P[idx[i]]) ** 2).sum(1))
    return P[idx]


def sample_points(sdf, geo, N, kind, rng):
    V, A = geo["V"], geo["A"]
    h = (V / N) ** (1 / 3)
    for _ in range(3):
        n_s = int(round(A / h ** 2))
        h = (V / (N - n_s)) ** (1 / 3)
    n_i = N - n_s
    lo, hi = geo["lo"], geo["hi"]
    if kind == "random":
        pts = []
        while sum(len(p) for p in pts) < n_i:
            X = rng.uniform(lo, hi, (4 * n_i, 3))
            pts.append(X[sdf(X) > 0])
        inner = np.concatenate(pts)[:n_i]
        w = geo["cand_w"] / geo["cand_w"].sum()      # band density ∝ 1/|grad sdf| -> reweight to area-uniform
        surf = geo["cand"][rng.choice(len(w), n_s, replace=len(w) < n_s, p=w)]
    else:
        hh = 0.93 * h
        while True:                                  # thin parts lose many lattice points to the 0.35h margin
            axes = [np.arange(l + hh / 2, u, hh) for l, u in zip(lo, hi)]
            X = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
            X = X + rng.uniform(-0.35 * hh, 0.35 * hh, X.shape)
            X = X[sdf(X) > 0.35 * h]
            if len(X) >= n_i:
                break
            hh *= 0.9
        inner = fps(X, n_i, rng)
        surf = fps(geo["cand"], n_s, rng)
    P = np.concatenate([inner, surf])
    return P, np.r_[np.zeros(len(inner), bool), np.ones(len(surf), bool)], h


def build_sample(sdf, geo, N, kind, rng):
    for _ in range(5):
        P, is_s, h = sample_points(sdf, geo, N, kind, rng)
        if len(P) != N:
            continue
        c = P.mean(0)
        s = 1.0 / np.abs(P - c).max()
        Y = (P - c) * s
        sdf_n = lambda y: s * sdf(y / s + c)
        op = delaunay_p1(Y, is_s, sdf_n, h * s)
        if op.info["isolated"] == 0:
            lam, vec = op.eigs(K + 1)
            return (Y.astype(np.float32), vec[:, :K].astype(np.float32), op.m.astype(np.float32), lam, op.info,
                    op.K, is_s)
    raise RuntimeError("could not build a valid sample")


def shape_task(args):
    split, family, idx, samples = args
    rng = np.random.default_rng([SEED, ["train", "test_in", "test_out"].index(split),
                                 (TRAIN_FAMILIES + [TEST_FAMILY]).index(family), idx])
    t0 = time.perf_counter()
    sdf0, bound = make_sdf(family, rng)
    R = Rotation.random(random_state=rng.integers(2 ** 31)).as_matrix()
    sdf = rotated(sdf0, R)
    geo = geometry(sdf, bound * 1.05, rng)
    t_geo = time.perf_counter() - t0
    out = []
    for kind, N in samples:
        t1 = time.perf_counter()
        Y, vec, m, lam, info, Kop, is_s = build_sample(sdf, geo, N, kind, rng)
        out.append(dict(kind=kind, N=N, points=Y, evecs=vec, mass=m, lam=lam[:K], info=info,
                        t=time.perf_counter() - t1))
        if split != "train":                          # evaluation needs the operator (Rayleigh-Ritz) and eigenvalues
            d = os.path.join(OUT, split, "ops")
            os.makedirs(d, exist_ok=True)
            Kc = Kop.tocsr()
            np.savez(os.path.join(d, f"{family}_{idx:04d}_{kind}_{N}.npz"), K_data=Kc.data, K_indices=Kc.indices,
                     K_indptr=Kc.indptr, m=m.astype(np.float64), lam=lam, is_surface=is_s, points=Y)
    return dict(split=split, family=family, idx=idx, V=geo["V"], A=geo["A"], t_geo=t_geo, samples=out)


def write_h5(results, path_prefix, key):
    """NEO pretrain_tet layout; one file pair per group of equal-size samples."""
    n = len(results)
    per = len(results[0]["samples"])
    N = results[0]["samples"][0]["N"]
    samples = np.stack([[s["points"] for s in r["samples"]] for r in results])
    evecs = np.stack([[s["evecs"] for s in r["samples"]] for r in results])
    mass = np.stack([[s["mass"] for s in r["samples"]] for r in results])
    vl = h5py.special_dtype(vlen=np.dtype("float32"))
    vi = h5py.special_dtype(vlen=np.dtype("int32"))
    with h5py.File(f"{path_prefix}_mesh.hdf5", "w") as hf:
        hf.create_dataset("obj_paths", data=np.array([f"{r['family']}_{r['idx']:04d}".encode() for r in results]))
        vd = hf.create_dataset("verts", shape=(n,), dtype=vl)
        fd = hf.create_dataset("faces", shape=(n,), dtype=vi)
        for i, r in enumerate(results):
            vd[i] = r["samples"][0]["points"].ravel()
            fd[i] = np.zeros(3, np.int32)
        hf.create_dataset("vert_shapes", data=np.array([[N, 3]] * n, np.int32))
        hf.create_dataset("face_shapes", data=np.array([[1, 3]] * n, np.int32))
    with h5py.File(f"{path_prefix}_samples_evecs.hdf5", "w") as hf:
        hf.create_dataset("samples", data=samples, chunks=(1, 1, N, 3), compression="lzf")
        hf.create_dataset("evecs", data=evecs, chunks=(1, 1, N, min(64, K)), compression="lzf")
        hf.create_dataset("mass", data=mass, chunks=(1, 1, N), compression="lzf")
        hf.attrs["k"] = K
        hf.attrs["per_mesh_count"] = per
        hf.attrs["n_samples"] = N


def tasks_for(split):
    if split == "test_out":
        return [(split, TEST_FAMILY, i, TEST_SAMPLES) for i in range(SPLIT[split])]
    samples = PER_SHAPE if split == "train" else TEST_SAMPLES
    return [(split, f, i, samples) for f in TRAIN_FAMILIES for i in range(SPLIT[split])]


def estimate():
    t = []
    for split, fam, i, smp in [("train", f, 0, PER_SHAPE) for f in TRAIN_FAMILIES] + \
                              [("test_out", TEST_FAMILY, 0, TEST_SAMPLES), ("test_in", "cup", 0, TEST_SAMPLES)]:
        r = shape_task((split, fam, i, smp))
        per = [(s["kind"], s["N"], round(s["t"], 2), round(s["info"]["carved_frac"], 3), round(s["info"]["sliver_frac"], 4))
               for s in r["samples"]]
        print(f"{split:8s} {fam:12s} V={r['V']:.3f} A={r['A']:.3f} geo {r['t_geo']:.2f}s  samples {per}", flush=True)
        t.append(dict(split=split, family=fam, t_geo=r["t_geo"], t_samples=sum(s["t"] for s in r["samples"])))
    return pd.DataFrame(t)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for split in ("train", "test_in", "test_out"):
        d = os.path.join(OUT, split)
        os.makedirs(d, exist_ok=True)
        tasks = tasks_for(split)
        t0 = time.time()
        with Pool(20) as p:
            res = p.map(shape_task, tasks, chunksize=2)
        if split == "train":
            for b in range(0, len(res), 256):                     # file pairs of 256 shapes
                write_h5(res[b:b + 256], os.path.join(d, f"vol_{b // 256:03d}"), "train")
        else:
            for N in sorted({s["N"] for s in res[0]["samples"]}):
                for kind in ("quasi", "random"):
                    sub = [dict(r, samples=[s for s in r["samples"] if s["N"] == N and s["kind"] == kind]) for r in res]
                    dd = os.path.join(d, f"{kind}_{N}")
                    os.makedirs(dd, exist_ok=True)
                    write_h5(sub, os.path.join(dd, "vol_000"), split)
        for r in res:
            for s in r["samples"]:
                rows.append(dict(split=split, family=r["family"], idx=r["idx"], kind=s["kind"], N=s["N"],
                                 V=r["V"], A=r["A"], t_s=s["t"], **s["info"]))
        print(f"{split}: {len(res)} shapes in {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "index.csv"), index=False)


if __name__ == "__main__":
    print(estimate().to_string()) if "--estimate" in sys.argv else main()

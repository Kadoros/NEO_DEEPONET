"""C step 1: real shapes -> closed tet meshes (fTetWild) -> volume point clouds + Delaunay-P1 operators -> real_shapes/

Shapes: Stanford Bunny, Armadillo (Open3D example data), Dragon (pyvista example data), cow (libigl tutorial data)
        + 10 Thingi10K models (c_select_thingi.py, seeded, genus-stratified).
Per shape
  1. surface -> centred on the bounding-box centre, scaled to max |coord| = 1
  2. fTetWild (wildmeshing.Tetrahedralizer, edge_length_r = EDGE_R of the bbox diagonal, epsilon 1e-3); open input
     surfaces (Bunny, Dragon) are closed by fTetWild's winding-number inside test
  3. SDF = - igl.signed_distance(x, boundary of the tet mesh, winding number)   (positive inside, as in stage 2)
  4. volume point clouds exactly as the stage-2 pipeline (neo_vol_data.geometry / sample_points, Delaunay-P1 with
     SDF carving + sliver removal), normalised like NEO (centre = mean, max |coord| = 1):
     kind in {quasi, random} x N in {8192, 16384}
  5. scalar ARPACK eigenpairs (256) of each operator  -> cache npz (same fields as stage3/e1/cache)
Outputs: real_shapes/tet/<shape>.npz (tet mesh + boundary), real_shapes/cache/<shape>_<kind>_<N>.npz

    python stage3/c_volumize.py [--procs 4]          (CPU; fTetWild uses TET_THREADS threads per shape)
"""
import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import e1_basis as E                      # sets caches, imports stage-2 modules (D0, delaunay_p1)

import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as sla  # noqa: E402

RS = os.path.join(E.ROOT, "real_shapes")
RAW = os.path.join(RS, "raw")
EDGE_R = 0.02
TET_THREADS = 4
SAMPLES = [("quasi", 8192), ("random", 8192), ("quasi", 16384), ("random", 16384)]
SEED = 20260928


def shape_list():
    out = {"bunny": ("ply", os.path.join(RAW, "open3d/extract/BunnyMesh/BunnyMesh.ply")),
           "armadillo": ("ply", os.path.join(RAW, "open3d/extract/ArmadilloMesh/ArmadilloMesh.ply")),
           "dragon": ("ply", os.path.join(RAW, "pyvista/dragon.ply")),
           "cow": ("off", os.path.join(RAW, "libigl/cow.off"))}
    import pandas as pd
    sel = pd.read_csv(os.path.join(RAW, "thingi10k", "selected.csv"))
    for _, r in sel.iterrows():
        out[f"thingi{r.file_id}"] = ("npz", os.path.join(RAW, "thingi10k", f"{r.file_id}.npz"))
    return out


def load_surface(kind, path):
    if kind == "npz":
        z = np.load(path)
        return z["vertices"].astype(np.float64), z["facets"].astype(np.int64)
    import igl
    if kind == "off":
        V, F = igl.read_triangle_mesh(path)[:2]
        return np.asarray(V, np.float64), np.asarray(F, np.int64)
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(path)
    return np.asarray(m.vertices, np.float64), np.asarray(m.triangles, np.int64)


def tetwild(name, kind, path):
    import wildmeshing as wm
    out = os.path.join(RS, "tet", f"{name}.npz")
    if os.path.exists(out):
        return out
    V, F = load_surface(kind, path)
    c = 0.5 * (V.min(0) + V.max(0))
    V = (V - c) / np.abs(V - c).max()
    t0 = time.time()
    tw = wm.Tetrahedralizer(stop_quality=10, epsilon=1e-3, edge_length_r=EDGE_R, max_threads=TET_THREADS)
    tw.set_mesh(V, F)
    tw.tetrahedralize()
    Vt, Tt = tw.get_tet_mesh()[:2]                                     # (V, T, markers)
    Vt, Tt = np.asarray(Vt, np.float64), np.asarray(Tt, np.int64)
    used, Tt = np.unique(Tt, return_inverse=True)
    Vt, Tt = Vt[used], Tt.reshape(-1, 4)
    Fb, _ = E.boundary_faces(Tt, Vt)
    np.savez(out, V=Vt, T=Tt, Fb=Fb, n_input_faces=len(F), t_tetwild=time.time() - t0)
    return out


def make_sdf(tetfile):
    import igl
    z = np.load(tetfile)
    V, Fb = z["V"], z["Fb"].astype(np.int64)
    st = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_WINDING_NUMBER

    def sdf(x):
        return -igl.signed_distance(np.ascontiguousarray(x, np.float64), V, Fb, sign_type=st)[0]
    return sdf


def build(name, sdf, kind, N, rng):
    """As stage 2 (up to 5 resamplings until no isolated node); if thin features keep isolating a few nodes, the
    last draw is kept and c_cleanup.py drops the isolated nodes (recorded)."""
    geo = E.D0.geometry(sdf, 1.05, rng)
    best = None
    for _ in range(5):
        P, is_s, h = E.D0.sample_points(sdf, geo, N, kind, rng)
        c = P.mean(0)
        s = 1.0 / np.abs(P - c).max()
        Y = (P - c) * s
        sdf_n = lambda y, s=s, c=c: s * sdf(y / s + c)
        op = E.delaunay_p1(Y, is_s, sdf_n, h * s)
        if len(Y) == N and op.info["isolated"] == 0:
            return Y, op, geo
        if len(Y) == N and (best is None or op.info["isolated"] < best[1].info["isolated"]):
            best = (Y, op)
    if best is None:
        raise RuntimeError("sampling failed")
    return best[0], best[1], geo


def task(args):
    try:
        return _task(args)
    except Exception as e:                                             # one bad shape must not stop the others
        return args[0], -1, -1, [f"FAILED: {e!r}"]


def _task(args):
    name, kind_file, path = args
    t0 = time.time()
    tetfile = tetwild(name, kind_file, path)
    t_tet = time.time() - t0
    sdf = make_sdf(tetfile)
    rows = []
    for j, (kind, N) in enumerate(SAMPLES):
        out = os.path.join(RS, "cache", f"{name}_{kind}_{N}.npz")
        if os.path.exists(out):
            continue
        t1 = time.time()
        rng = np.random.default_rng([SEED, sum(map(ord, name)), j])
        Y, op, geo = build(name, sdf, kind, N, rng)
        t_op = time.time() - t1
        if op.info["isolated"] > 0:                                   # zero-mass rows: eigenpairs after c_cleanup
            ls, Ps = np.zeros(256), np.zeros((len(Y), 256))
        else:
            ls, Ps = sla.eigsh(op.K.tocsc(), k=256, M=sp.diags(op.m).tocsc(), sigma=-1e-6)
        o = np.argsort(ls)
        np.savez(out, Y=Y, tets=op.tets, grads=op.grads, vol=op.vol, m=op.m, sliver=op.info["sliver_frac"],
                 carved=op.info["carved_frac"], isolated=op.info["isolated"], Ps=Ps[:, o], lam_s=ls[o], t_scalar=time.time() - t1 - t_op, t_op=t_op,
                 V=geo["V"], A=geo["A"])
        rows.append(f"{name} {kind} {N}: op {t_op:.0f}s eigs {time.time() - t1 - t_op:.0f}s sliver {op.info['sliver_frac']:.3f}")
    return name, t_tet, time.time() - t0, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    for d in ("tet", "cache"):
        os.makedirs(os.path.join(RS, d), exist_ok=True)
    shapes = shape_list()
    if a.only:
        shapes = {k: v for k, v in shapes.items() if k in a.only.split(",")}
    tasks = [(n, k, p) for n, (k, p) in shapes.items()]
    with Pool(a.procs) as pool:
        for name, t_tet, t_all, rows in pool.imap_unordered(task, tasks):
            print(f"{name}: tetwild {t_tet:.0f}s, total {t_all:.0f}s", flush=True)
            for r in rows:
                print("   ", r, flush=True)


if __name__ == "__main__":
    main()

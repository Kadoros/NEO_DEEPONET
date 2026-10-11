"""Cross-check of the pilot targets (Delaunay-P1 eigenpairs) against fine tet FEM on a few test shapes
-> data_neo/crosscheck.csv

For shapes idx 0,1 of every family in test_in and of c_tube in test_out, each test sample is replayed with the same random stream to recover its raw points, rotation and normalisation.
A fine tet mesh of the same solid is built (gmsh OCC: torus, cup, shell, c_tube; tetgen from a dense star
surface: sh_ball, superquadric), and its P1 eigenpairs are computed ONCE per shape in the canonical frame.
Per sample the mesh is mapped into the sample's normalised frame x -> s (R x - c): eigenvector functions are
unchanged and eigenvalues scale by 1/s^2; the eigenvectors are interpolated to the points (opcheck_span_fem.interp).  Reported: span loss of the first K_ref complete FEM
groups (<= 32) inside the Delaunay-P1 span of the same size, and the eigenvalue relative error.

    python stage2/neo_vol_crosscheck.py [--data neo|neo2]     (env neo, CPU)
      neo  -> pilot data (data_neo, neo_vol_data);  neo2 -> main data (data_neo2, neo_vol_data2, 7 families)
"""
import argparse
import os
import sys
import time

import opcheck_span_fem as SF            # project-local HOME/cache setup via opcheck on import
import neo_vol_data as D0
import neo_vol_data2 as D2

from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.spatial.transform import Rotation

import shapes as S1
from volop import delaunay_p1, p1_elements

H_FINE = 0.04                             # pilot (data_neo, tests up to N=8192)
N_FINE = 80_000                           # main (data_neo2, tests N=8192/16384): h from volume, ~N_FINE nodes
                                          # (200k nodes: >20 min and ~10 GB per shape with SuperLU shift-invert -> too costly)
D = D0                                    # data module; switched by --data (set before the Pool forks)


def params(family, rng):
    """Replays make_sdf's draws (same order) and returns the parameters."""
    if family == "sh_ball":
        radius, _ = S1.sh_sphere_radius(rng)
        return dict(radius=radius)
    if family == "superquadric":
        radius, _ = S1.superquadric_radius(rng)
        return dict(radius=radius)
    if family == "torus":
        return dict(r=rng.uniform(0.25, 0.55))
    if family == "cup":
        H, a, depth = rng.uniform(0.8, 1.4), rng.uniform(0.45, 0.8), rng.uniform(0.4, 0.8)
        return dict(H=H, a=a, zb=H / 2 - depth * H)
    if family == "shell":
        ax = rng.uniform(0.8, 1.2, 3)
        ri, off = rng.uniform(0.3, 0.5), rng.normal(size=3) * 0.08
        return dict(ax=ax, ri=ri, off=off)
    if family == "c_tube":
        return dict(r=rng.uniform(0.2, 0.4), th=np.deg2rad(rng.uniform(200, 320)))


def fine_mesh(family, p, h=H_FINE):
    """Fine tet mesh (canonical frame): returns vertices, tets."""
    spheroid = family == "elongated" and p["shape"] == "spheroid"      # gmsh gives no tets for a dilated sphere
    if family in ("sh_ball", "superquadric") or spheroid:
        import tetgen
        import trimesh
        ico = trimesh.creation.icosphere(subdivisions=5)
        d = np.asarray(ico.vertices)
        V = d * np.array([p["a"], p["b"], p["b"]]) if spheroid else d * p["radius"](d)[:, None]
        tg = tetgen.TetGen(V, np.asarray(ico.faces))
        # the maxvolume= keyword is ignored by this tetgen wrapper; the switch string applies the constraint
        P, T = tg.tetrahedralize(switches=f"pq1.5a{h ** 3 / 6:.3e}Q")[:2]
        return np.asarray(P), np.asarray(T)
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    occ = gmsh.model.occ
    if family == "torus":
        occ.addTorus(0, 0, 0, 1.0, p["r"])
    elif family == "cup":
        o = occ.addCylinder(0, 0, -p["H"] / 2, 0, 0, p["H"], 1.0)
        c = occ.addCylinder(0, 0, p["zb"], 0, 0, p["H"] / 2 + 0.3 - p["zb"], p["a"])
        occ.cut([(3, o)], [(3, c)])
    elif family == "shell":
        o = occ.addSphere(0, 0, 0, 1.0)
        occ.dilate([(3, o)], 0, 0, 0, *p["ax"])
        c = occ.addSphere(*p["off"], p["ri"])
        occ.cut([(3, o)], [(3, c)])
    elif family == "elongated" and p["shape"] == "capsule":
        c = occ.addCylinder(-p["hl"], 0, 0, 2 * p["hl"], 0, 0, p["r"])
        s0, s1 = occ.addSphere(-p["hl"], 0, 0, p["r"]), occ.addSphere(p["hl"], 0, 0, p["r"])
        occ.fuse([(3, c)], [(3, s0), (3, s1)])
    elif family == "multihole":
        o = occ.addBox(-1, -0.6, -0.3, 2, 1.2, 0.6)
        holes = [(3, occ.addCylinder(c[0], c[1], -0.5, 0, 0, 1.0, r)) for c, r in zip(p["centres"], p["radii"])]
        occ.cut([(3, o)], holes)
    elif family == "c_tube":
        t = occ.addTorus(0, 0, 0, 1.0, p["r"], angle=p["th"])
        s0 = occ.addSphere(1.0, 0, 0, p["r"])
        s1 = occ.addSphere(np.cos(p["th"]), np.sin(p["th"]), 0, p["r"])
        occ.fuse([(3, t)], [(3, s0), (3, s1)])
    occ.synchronize()
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
    return coords.reshape(-1, 3)[used], T.reshape(-1, 4)


def replay(split, family, idx, samples):
    """Same random stream as neo_vol_data.shape_task; returns per-sample (raw points, c, s, normalised points)."""
    seed = [D.SEED, ["train", "test_in", "test_out"].index(split), (D.TRAIN_FAMILIES + [D.TEST_FAMILY]).index(family), idx]
    rng = np.random.default_rng(seed)
    make_sdf = getattr(D, "make_sdf")
    sdf0, bound = make_sdf(family, rng)
    p = getattr(sdf0, "p", None) or params(family, np.random.default_rng(seed))
    R = Rotation.random(random_state=rng.integers(2 ** 31)).as_matrix()
    sdf = D.rotated(sdf0, R)
    geo = D.geometry(sdf, bound * 1.05, rng)
    out = []
    for kind, N in samples:                                   # mirrors build_sample's retry loop exactly
        for _ in range(5):
            P, is_s, h = D.sample_points(sdf, geo, N, kind, rng)
            if len(P) != N:
                continue
            c = P.mean(0)
            s = 1.0 / np.abs(P - c).max()
            Y = (P - c) * s
            op = delaunay_p1(Y, is_s, lambda y, s=s, c=c: s * sdf(y / s + c), h * s)
            if op.info["isolated"] == 0:
                out.append((kind, N, c, s, Y, op))
                break
    return p, R, geo["V"], out


def task(args):
    try:
        return _task(args)
    except Exception as e:                                    # record, do not abort the whole run
        split, family, idx = args
        return [dict(split=split, family=family, idx=idx, error=repr(e)[:200])]


def fine_eigs(Pc, T, k=40):
    """P1 eigenpairs of the fine mesh in the canonical frame (computed once per shape)."""
    Tc, G, vol = p1_elements(Pc, T)
    n = len(Pc)
    Kf = sp.coo_matrix(((vol[:, None, None] * np.einsum("tki,tkj->tij", G, G)).ravel(),
                        (np.repeat(Tc, 4, 1).ravel(), np.tile(Tc, (1, 4)).ravel())), shape=(n, n)).tocsr()
    mf = np.zeros(n)
    np.add.at(mf, Tc.ravel(), np.repeat(vol / 4, 4))
    lf, vf = sla.eigsh(Kf.tocsc(), k=k, M=sp.diags(mf).tocsc(), sigma=-1e-6)
    o = np.argsort(lf)
    return Tc, lf[o], vf[:, o]


def _task(args):
    split, family, idx = args
    p, R, V, smp = replay(split, family, idx, D.TEST_SAMPLES)
    t0 = time.perf_counter()
    h = H_FINE if D is D0 else (1.5 * V / N_FINE) ** (1 / 3)      # empirical: nodes ~ 1.5 V / h^3
    Pc, T = fine_mesh(family, p, h)
    Tc, lf0, vf = fine_eigs(Pc, T)
    t_fine = time.perf_counter() - t0
    rows = []
    for kind, N, c, s, Y, op in smp:
        stored = np.load(os.path.join(D.OUT, split, "ops", f"{family}_{idx:04d}_{kind}_{N}.npz"))
        assert np.allclose(stored["points"], Y, atol=1e-5), "replay mismatch"
        Pf = ((Pc @ R.T) - c) * s                             # canonical -> rotated -> normalised frame
        lf = lf0 / s ** 2                                     # eigenvalues scale with 1/length^2; vectors unchanged
        kr = SF.k_ref_from_eigs(lf)
        I, _ = SF.interp(Pf, Tc, Y)
        lam, Phi = op.eigs(kr + 8)
        rows.append(dict(split=split, family=family, idx=idx, kind=kind, N=N, n_fine=len(Pc), K_ref=kr, t_fine_s=t_fine,
                         span_loss=SF.span(Phi[:, :kr], I @ vf[:, :kr], op.m),
                         eig_relerr=float(np.abs(lam[1:kr] / lf[1:kr] - 1).mean()),
                         eig_bias=float((lam[1:kr] / lf[1:kr] - 1).mean())))
    return rows


def main():
    global D
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["neo", "neo2"], default="neo")
    D = D2 if ap.parse_args().data == "neo2" else D0
    tasks = [("test_in", f, i) for f in D.TRAIN_FAMILIES for i in (0, 1)] + [("test_out", "c_tube", i) for i in (0, 1)]
    t0 = time.time()
    with Pool(12) as pool:
        rows = sum(pool.map(task, tasks, chunksize=1), [])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(D.OUT, "crosscheck.csv"), index=False)
    if "error" in df:
        print("FAILED:", df[df.error.notna()][["split", "family", "idx", "error"]].to_string())
        df = df[df.error.isna()] if df.error.notna().any() else df
    print(df.groupby(["family", "kind", "N"])[["span_loss", "eig_relerr", "eig_bias", "n_fine"]].mean()
          .to_string(float_format=lambda v: f"{v:.3g}"))
    print(f"{len(df)} comparisons in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

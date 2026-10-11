"""NEO volume MAIN training data (branch A) -> stage2/data_neo2/

Same pipeline as the pilot (neo_vol_data.py: sampling, normalisation, Delaunay-P1 targets, NEO h5 layout),
with two added training families and a configurable target resolution:
  elongated  straight capsules (radius 0.2-0.35, half-length 0.6-1.2) or prolate spheroids (aspect 3-6)
  multihole  slab [-1,1]x[-0.6,0.6]x[-0.3,0.3] with 2-3 non-overlapping cylindrical through-holes (genus 2-3)
Test-only family stays c_tube (bent tube with round caps; never in training).

    python stage2/neo_vol_data2.py --estimate            # timing (N = 2048 and 4096)
    python stage2/neo_vol_data2.py --N 4096,2048 [--per-family 600] [--procs 16]
      train targets for every listed N from the SAME shapes -> data_neo2/train_<N>/ (2 quasi + 2 random per N)
"""
import argparse
import os
import sys
import time

import neo_vol_data as D                     # sets cache dirs; reuses geometry / sampling / targets / h5 writer

from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

SEED = 20260927
TRAIN_FAMILIES = D.TRAIN_FAMILIES + ["elongated", "multihole"]
TEST_FAMILY = "c_tube"
OUT = os.path.join(D.ROOT, "data_neo2")
TEST_SAMPLES = [("quasi", 8192), ("random", 8192), ("quasi", 16384), ("random", 16384)]   # evaluation at N >= 8192
rotated, geometry, sample_points, K = D.rotated, D.geometry, D.sample_points, D.K        # for the cross-check replay


def make_sdf(family, rng):
    if family == "elongated":
        if rng.uniform() < 0.5:                                    # capsule along x
            r, hl = rng.uniform(0.2, 0.35), rng.uniform(0.6, 1.2)

            def sdf(x):
                q = x.copy()
                q[:, 0] = np.clip(q[:, 0], -hl, hl)
                return r - np.linalg.norm(x - q * np.array([1, 0, 0]), axis=1)
            sdf.p = dict(shape="capsule", r=r, hl=hl)                # parameters for the fine-mesh cross-check
            return sdf, hl + r
        a = 1.0
        b = a / rng.uniform(3, 6)                                  # prolate spheroid (a, b, b)

        def sdf(x):
            rr = np.linalg.norm(x, axis=1)
            d = x / np.maximum(rr, 1e-12)[:, None]
            return 1 / np.sqrt((d[:, 0] / a) ** 2 + (d[:, 1] / b) ** 2 + (d[:, 2] / b) ** 2) - rr
        sdf.p = dict(shape="spheroid", a=a, b=b)
        return sdf, a
    if family == "multihole":
        n_holes = rng.integers(2, 4)
        centres, radii = [], []
        while len(centres) < n_holes:
            c = np.array([rng.uniform(-0.75, 0.75), rng.uniform(-0.35, 0.35)])
            r = rng.uniform(0.15, 0.25)
            inside = (abs(c[0]) + r < 0.9) and (abs(c[1]) + r < 0.5)
            if inside and all(np.linalg.norm(c - c2) > r + r2 + 0.1 for c2, r2 in zip(centres, radii)):
                centres.append(c); radii.append(r)

        def sdf(x):
            box = np.minimum.reduce([1 - np.abs(x[:, 0]), 0.6 - np.abs(x[:, 1]), 0.3 - np.abs(x[:, 2])])
            holes = [np.linalg.norm(x[:, :2] - c, axis=1) - r for c, r in zip(centres, radii)]
            return np.minimum(box, np.minimum.reduce(holes))
        sdf.p = dict(centres=centres, radii=radii)
        return sdf, 1.2
    return D.make_sdf(family, rng)


def shape_task(args):
    split, family, idx, samples = args
    rng = np.random.default_rng([SEED, ["train", "test_in", "test_out"].index(split),
                                 (TRAIN_FAMILIES + [TEST_FAMILY]).index(family), idx])
    t0 = time.perf_counter()
    sdf0, bound = make_sdf(family, rng)
    R = Rotation.random(random_state=rng.integers(2 ** 31)).as_matrix()
    sdf = D.rotated(sdf0, R)
    geo = D.geometry(sdf, bound * 1.05, rng)
    t_geo = time.perf_counter() - t0
    out = []
    for kind, N in samples:
        t1 = time.perf_counter()
        Y, vec, m, lam, info, Kop, is_s = D.build_sample(sdf, geo, N, kind, rng)
        out.append(dict(kind=kind, N=N, points=Y, evecs=vec, mass=m, lam=lam[:D.K], info=info,
                        t=time.perf_counter() - t1))
        if split != "train":
            d = os.path.join(OUT, split, "ops")
            os.makedirs(d, exist_ok=True)
            Kc = Kop.tocsr()
            np.savez(os.path.join(d, f"{family}_{idx:04d}_{kind}_{N}.npz"), K_data=Kc.data, K_indices=Kc.indices,
                     K_indptr=Kc.indptr, m=m.astype(np.float64), lam=lam, is_surface=is_s, points=Y)
    return dict(split=split, family=family, idx=idx, V=geo["V"], A=geo["A"], t_geo=t_geo, samples=out)


def estimate():
    rows = []
    for fam in TRAIN_FAMILIES + [TEST_FAMILY]:
        for N in (2048, 4096):
            r = shape_task(("train", fam, 0, [("quasi", N), ("random", N)]))
            rows.append(dict(family=fam, N=N, t_geo=r["t_geo"], t_per_sample=np.mean([s["t"] for s in r["samples"]]),
                             sliver=np.mean([s["info"]["sliver_frac"] for s in r["samples"]]),
                             carved=np.mean([s["info"]["carved_frac"] for s in r["samples"]])))
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main(Ns, per_family, procs, n_test_in=20, n_test_out=60):
    os.makedirs(OUT, exist_ok=True)
    per_shape = [(kind, N) for N in Ns for kind in ("quasi", "quasi", "random", "random")]
    plan = {"train": [("train", f, i, per_shape) for f in TRAIN_FAMILIES for i in range(per_family)],
            "test_in": [("test_in", f, i, TEST_SAMPLES) for f in TRAIN_FAMILIES for i in range(n_test_in)],
            "test_out": [("test_out", TEST_FAMILY, i, TEST_SAMPLES) for i in range(n_test_out)]}
    rows = []
    for split, tasks in plan.items():
        d = os.path.join(OUT, split)
        t0 = time.time()
        res = []
        with Pool(procs) as p:
            if split == "train":                                   # chunks of 256 shapes: bounded RAM
                for b in range(0, len(tasks), 256):
                    chunk = p.map(shape_task, tasks[b:b + 256], chunksize=2)
                    for N in Ns:
                        dn = os.path.join(OUT, f"train_{N}")
                        os.makedirs(dn, exist_ok=True)
                        D.write_h5([dict(r, samples=[s for s in r["samples"] if s["N"] == N]) for r in chunk],
                                   os.path.join(dn, f"vol_{b // 256:03d}"), "train")
                    res += [dict(r, samples=[dict(s, points=None, evecs=None, mass=None) for s in r["samples"]]) for r in chunk]
                    print(f"  train chunk {b // 256} done ({time.time() - t0:.0f}s)", flush=True)
            else:
                res = p.map(shape_task, tasks, chunksize=2)
        if split != "train":
            for n in sorted({s["N"] for s in res[0]["samples"]}):
                for kind in ("quasi", "random"):
                    sub = [dict(r, samples=[s for s in r["samples"] if s["N"] == n and s["kind"] == kind]) for r in res]
                    dd = os.path.join(d, f"{kind}_{n}")          # d = OUT/<split>
                    os.makedirs(dd, exist_ok=True)
                    D.write_h5(sub, os.path.join(dd, "vol_000"), split)
        for r in res:
            for s in r["samples"]:
                rows.append(dict(split=split, family=r["family"], idx=r["idx"], kind=s["kind"], N=s["N"], t_s=s["t"], **s["info"]))
        print(f"{split}: {len(res)} shapes in {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "index.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--N", default="4096,2048")
    ap.add_argument("--per-family", type=int, default=600)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    print(estimate().to_string()) if a.estimate else main([int(n) for n in a.N.split(",")], a.per_family, a.procs)

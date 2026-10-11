"""E1 diagnostic: do "local rotation" fields fix the replicated basis on bending-dominated shapes?

Hypothesis (from e1_basis results): on slender / thin-walled shapes (c_tube, torus, cup) the static response is
dominated by soft bending modes, which are nearly strain-free local rotations; the replicated scalar basis only
approximates them with residual strain, so the reduced model is too stiff (reduced-model bending locking).
Test: augment neo256x3 (+ 12 affine) with R_{k,c}(x) = phi_k(x) (e_c x (x - x_bar)) for the first K_rot scalar NEO
modes (3 K_rot fields), then the same span / static tests as e1_basis (smooth + pinch loads, homogeneous + lumps).

    python stage3/e1_rot.py [--nu 0.3,0.49] [--procs 8]      (CPU; uses e1/cache from e1_basis prep/neo)
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as sla

import e1_basis as E

K_ROTS = [0, 16, 32, 64]


def rotation_fields(Y, Phi, k):
    x = Y - Y.mean(0)
    F = []
    for j in range(k):
        for c in range(3):
            e = np.zeros(3); e[c] = 1
            F.append((Phi[:, j:j + 1] * np.cross(e, x)).ravel())
    return np.stack(F, 1) if F else np.zeros((3 * len(Y), 0))


def task(args):
    family, idx, nus = args
    t0 = time.time()
    z = np.load(E.cache_path(family, idx))
    Y, op = z["Y"], E.Op(z)
    rng = np.random.default_rng([E.SEED, E.FAMILIES.index(family), idx, 1])     # same draws as e1_basis.task
    m3 = np.repeat(op.m, 3)
    Pn = z["Pn"]
    E_het = E.lumps(Y, rng, 1 + idx % 2)(Y[op.tets].mean(1))
    loads = E.pinch_load(Y, op.tets, rng)
    base = np.hstack([E.replicate(Pn), E.affine_fields(Y)])
    rows = []
    for nu in nus:
        K = E.elastic_stiffness(op, np.ones(len(op.vol)), nu)
        Kh = E.elastic_stiffness(op, E_het, nu)
        le, Ue = sla.eigsh(K.tocsc(), k=6 + 100, M=sp.diags(m3).tocsc(), sigma=-1e-6)
        o = np.argsort(le); le, Ue = le[o], Ue[:, o]
        kr = E.k_ref(le, 6, E.K_MODES)
        u_full = {(ld, mat): E.full_solve(Kc, f, Y, m3) for ld, f in loads.items() for mat, Kc in (("hom", K), ("lumps", Kh))}
        for kr_ in K_ROTS:
            Q = E.ritz(E.m_orthonormal(np.hstack([base, rotation_fields(Y, Pn, kr_)]), m3), K)[1]
            r = dict(family=family, idx=idx, nu=nu, k_rot=kr_, dim=Q.shape[1],
                     span_mode7=E.span_loss(Q, Ue[:, 6:7], m3), span_modes10=E.span_loss(Q, Ue[:, 6:16], m3),
                     span_modes=E.span_loss(Q, Ue[:, 6:kr], m3))
            for ld, f in loads.items():
                for mat, Kc in (("hom", K), ("lumps", Kh)):
                    r.update(E.errs(f"static_{ld}_{mat}", u_full[(ld, mat)], E.reduced_solve(Q, Kc, f, Y, m3), Kc, m3))
            rows.append(r)
    for r in rows:
        r["t_s"] = time.time() - t0
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", default="0.3,0.49")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    nus = [float(v) for v in a.nu.split(",")]
    tasks = [(f, i, nus) for f in E.FAMILIES for i in range(2)]
    rows = []
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(task, tasks):
            rows += r
            print(r[0]["family"], r[0]["idx"], f"{r[0]['t_s']:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(E.OUT, "e1_rot.csv"), index=False)
    for col in ("static_smooth_hom_relL2", "static_smooth_lumps_relL2", "static_pinch_hom_relL2", "span_mode7", "span_modes"):
        print(f"\n== {col}")
        print(df.pivot_table(index=["nu", "family"], columns="k_rot", values=col, aggfunc="mean")
              .to_string(float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()

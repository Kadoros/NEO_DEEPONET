"""Stage 3 / E1 re-evaluation with the corrected (locking-free) reference -> stage3/e1/e1_v2_*.csv

Reference operator: selective nodal averaging (ANP, elastic_ops.anp_stiffness), validated on the Lame sphere.
Bases (all (M x I3)-orthonormalised, Rayleigh-Ritz with the homogeneous ANP stiffness of that nu):
  neo            scalar NEO 256 x 3 + 12 affine                                     (E1 design as first proposed)
  neo_rot        neo + 192 rotation fields phi_k (e_c x (x - x_bar)), k < 64         (decision 1/6, part 1)
  neo_rot_sc     neo_rot + the static-correction vector of the load (homogeneous-E full solution)  (decision 1/6)
  neo_rot_inv{1,2}_sc  (c): the neo_rot block refined by 1 or 2 steps of block inverse iteration
                 V <- (K + s M)^{-1} M V (one sparse LU of the shifted ANP K, s = 1e-3 lambda_7), then + static correction
  elastic_sc     exact ANP elastic modes (6 + 256) + 12 affine + static correction   (reference of the same idea)
A4 shape indicators (per shape, nu-independent): Korn constant C_K = 1 / theta_7 with theta the generalised
eigenvalues of (strain energy int eps:eps, gradient energy int grad u:grad u) on the rigid-mode complement;
thickness index = d_max / V^(1/3); lambda_7 of the ANP operator per nu.
Loads: as E1 (free body, equilibrated two-spot pressure: pinch = sharp discs, smooth = Gaussian; rigid modes pinned
by M-orthogonality).
Criterion (decision): model error / lump signal <= 0.2, surface and interior separately, per depth.
  lump      one stiff sphere E = 5 (edge width 0.02), radius r = min(0.15, 0.6 d_max); centre depth below the
            surface: shallow = r + 0.03, deep = d_max, mid = halfway (d_max = deepest interior point)
  signal    u_lump - u_hom           (both full-order ANP solutions)
  error     u_lump_reduced - u_lump  (Galerkin in the basis)
  norms     surface: nodal surface-area weights; interior: lumped mass on non-surface nodes
Also reported: relative L2 of the reduced solution (hom and lump), span loss of the first 96 exact ANP modes,
and the ANP spectrum check (near-zero modes at nu = 0.49 vs 0.3).

    python stage3/e1_v2.py [--nu 0.3,0.45,0.49] [--procs 8]          (CPU; uses e1/cache)
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.spatial import cKDTree

import e1_basis as E
import elastic_ops as EO
from e1_rot import rotation_fields

K_ROT = 64
E_HI = 5.0
DEPTHS = ("shallow", "mid", "deep")


def surface_weights(Y, T):
    F, nrm = E.boundary_faces(T, Y)
    area = np.linalg.norm(nrm, axis=1)
    w = np.zeros(len(Y))
    np.add.at(w, F.ravel(), np.repeat(area / 3, 3))
    return w


def depth_lumps(Y, surf_nodes):
    d = cKDTree(Y[surf_nodes]).query(Y)[0]
    dmax = d.max()
    r = min(0.15, 0.6 * dmax)
    targets = {"shallow": r + 0.03, "deep": dmax}
    targets["mid"] = 0.5 * (targets["shallow"] + targets["deep"])
    out = {}
    for name in DEPTHS:
        i = int(np.argmin(np.abs(d - targets[name])))
        out[name] = (Y[i], r, float(d[i]))
    return out, float(dmax)


def lump_field(x, c, r, w=0.02):
    return 1 + (E_HI - 1) / (1 + np.exp(-(r - np.linalg.norm(x - c, axis=1)) / w))


def refine(Q, lu, K, m3, iters, Y):
    """Block inverse iteration V <- (K + s M)^{-1} M V on the M-complement of the rigid modes (free body: without
    this deflation the rigid components are amplified by 1/s and swamp the orthonormalisation); after each step the
    12 affine fields are added back and Rayleigh-Ritz is applied.  Returns {step: basis}, the per-step time."""
    R = E.rigid_modes(Y)
    RtMR = R.T @ (m3[:, None] * R)
    defl = lambda V: V - R @ np.linalg.solve(RtMR, R.T @ (m3[:, None] * V))
    V = E.m_orthonormal(defl(Q), m3)
    out, ts = {}, []
    for it in range(1, iters + 1):
        t = time.time()
        V = E.m_orthonormal(defl(lu.solve(m3[:, None] * V)), m3)
        out[it] = E.ritz(E.m_orthonormal(np.hstack([V, E.affine_fields(Y)]), m3), K)[1]
        ts.append(time.time() - t)
    return out, ts


def korn_constant(op, m3, Y):
    """C_K = 1 / theta_7:  int eps:eps >= theta int grad u : grad u on the complement of the 6 rigid modes."""
    B = EO.b_matrices(op.grads)
    De = np.diag([1, 1, 1, 0.5, 0.5, 0.5])                            # eps:eps with engineering shear
    Keps = EO.assemble(op.tets, op.vol[:, None, None] * np.einsum("tia,ij,tjb->tab", B, De, B), len(m3))
    G = op.grads
    Ks = op.vol[:, None, None] * np.einsum("tki,tkj->tij", G, G)       # scalar (4 x 4) -> replicate per component
    n = len(op.m)
    rows, cols, vals = [], [], []
    for c in range(3):
        rows.append((3 * op.tets[:, :, None] + c).repeat(4, 2).ravel())
        cols.append((3 * op.tets[:, None, :] + c).repeat(4, 1).ravel())
        vals.append(Ks.ravel())
    Kg = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * n, 3 * n)).tocsr()
    th, X = sla.eigsh(Keps.tocsc(), k=14, M=(Kg + 1e-8 * sp.diags(m3)).tocsc(), sigma=-1e-3)
    o = np.argsort(th); th, X = th[o], X[:, o]
    # exclude eigenvectors lying in the rigid-mode span (solver noise can put a 7th ~1e-6 value there)
    R = E.rigid_modes(Y)
    MR = m3[:, None] * R
    res = X - R @ np.linalg.solve(R.T @ MR, MR.T @ X)
    frac = np.sqrt((m3[:, None] * res ** 2).sum(0) / (m3[:, None] * X ** 2).sum(0))
    nonrigid = th[frac > 1e-2]
    return float(1 / nonrigid[0]), th


def wnorm(v, w3):
    return float(np.sqrt((w3 * v ** 2).sum()))


def timed(fn, *a):
    t = time.time()
    r = fn(*a)
    return r, time.time() - t


def task(args):
    family, idx, nus = args
    t0 = time.time()
    z = np.load(E.cache_path(family, idx))
    Y, op = z["Y"], E.Op(z)
    rng = np.random.default_rng([E.SEED, E.FAMILIES.index(family), idx, 1])
    E.lumps(Y, rng, 1 + idx % 2)                                  # keep the e1_basis draw order for the loads
    loads = E.pinch_load(Y, op.tets, rng)
    m3 = np.repeat(op.m, 3)
    ws = surface_weights(Y, op.tets)
    surf = ws > 0
    w3_s = np.repeat(ws, 3)
    w3_i = np.repeat(np.where(surf, 0, op.m), 3)
    lumpset, dmax = depth_lumps(Y, np.where(surf)[0])
    cen = Y[op.tets].mean(1)
    S = EO.divergence_rows(op)
    ones = np.ones(len(op.vol))
    raw_neo, t_rep = timed(lambda: np.hstack([E.replicate(z["Pn"]), E.affine_fields(Y)]))
    rotf, t_rotf = timed(rotation_fields, Y, z["Pn"], K_ROT)
    raw_rot = np.hstack([raw_neo, rotf])
    rows = []
    C_K, _ = korn_constant(op, m3, Y)
    thick = dmax / float(op.vol.sum()) ** (1 / 3)
    for nu in nus:
        K, t_asm = timed(EO.anp_stiffness, op, ones, nu, S)
        Q_neo = E.ritz(E.m_orthonormal(raw_neo, m3), K)[1]
        Q_rot, t_rr_rot = timed(lambda: E.ritz(E.m_orthonormal(raw_rot, m3), K)[1])
        lam7 = float(sla.eigsh(K.tocsc(), k=7, M=sp.diags(m3).tocsc(), sigma=-1e-6)[0].max())   # shift scale only
        lu, t_lu = timed(lambda: sla.splu((K + 1e-3 * lam7 * sp.diags(m3)).tocsc()))
        Q_inv, t_it = refine(Q_rot, lu, K, m3, 2, Y)                 # (c): with rotation fields
        Q_inv_nr, t_it_nr = refine(Q_neo, lu, K, m3, 2, Y)           # (c) without rotation fields
        dim_c = Q_inv[2].shape[1]
        (le, Ue), t_eig = timed(lambda: sla.eigsh(K.tocsc(), k=dim_c - 12, M=sp.diags(m3).tocsc(), sigma=-1e-6))
        o = np.argsort(le); le, Ue = le[o], Ue[:, o]
        kr = E.k_ref(le, 6, E.K_MODES)
        Q_el = E.m_orthonormal(np.hstack([Ue[:, :6 + 256], E.affine_fields(Y)]), m3)       # 262 exact modes
        Q_elm = E.m_orthonormal(np.hstack([Ue, E.affine_fields(Y)]), m3)                   # dimension-matched to (c)
        span = {b: E.span_loss(Q, Ue[:, 6:kr], m3) for b, Q in (
            ("neo", Q_neo), ("neo_rot", Q_rot), ("neo_rot_inv1", Q_inv[1]), ("neo_rot_inv2", Q_inv[2]),
            ("neo_inv2", Q_inv_nr[2]), ("elastic", Q_el), ("elastic_matched", Q_elm))}
        Kl = {dn: EO.anp_stiffness(op, lump_field(cen, c, r), nu, S) for dn, (c, r, _) in lumpset.items()}
        cost = dict(t_neo_s=float(z["t_neo"]), t_replicate_s=t_rep, t_rotfields_s=t_rotf, t_orth_rr_s=t_rr_rot,
                    t_lu_s=t_lu, t_iter1_s=t_it[0], t_iter2_s=t_it[1], t_elastic_eigs_s=t_eig, t_assemble_K_s=t_asm)
        for ld, f in loads.items():
            u_hom, t_full = timed(E.full_solve, K, f, Y, m3)
            sc = lambda Q: E.m_orthonormal(np.hstack([Q, u_hom[:, None]]), m3)
            bases = {"neo": Q_neo, "neo_rot": Q_rot, "neo_rot_sc": sc(Q_rot), "neo_rot_inv1_sc": sc(Q_inv[1]),
                     "neo_rot_inv2_sc": sc(Q_inv[2]), "neo_inv2_sc": sc(Q_inv_nr[2]),
                     "elastic_sc": sc(Q_el), "elastic_matched_sc": sc(Q_elm)}
            u_lump = {}
            for dn in DEPTHS:
                u_lump[dn], t_full_lump = timed(E.full_solve, Kl[dn], f, Y, m3)
            for bn, Q in bases.items():
                uh_r = E.reduced_solve(Q, K, f, Y, m3)
                for dn in DEPTHS:
                    c, r, dep = lumpset[dn]
                    (Kn, ur), t_red = timed(lambda: (None, E.reduced_solve(Q, Kl[dn], f, Y, m3)))
                    sig, err = u_lump[dn] - u_hom, ur - u_lump[dn]
                    rows.append(dict(
                        family=family, idx=idx, nu=nu, load=ld, basis=bn, dim=Q.shape[1], depth=dn, depth_val=dep,
                        r=r, dmax=dmax, span_modes=span[bn[:-3] if bn.endswith("_sc") else bn],
                        korn=C_K, thick=thick, lam7=float(le[6]), n_zero=int((le < 1e-8 * le[6]).sum()),
                        lam8=float(le[7]), lam12=float(le[11]),
                        ratio_surf=wnorm(err, w3_s) / wnorm(sig, w3_s), ratio_int=wnorm(err, w3_i) / wnorm(sig, w3_i),
                        signal_surf=wnorm(sig, w3_s) / wnorm(u_lump[dn], w3_s),
                        signal_int=wnorm(sig, w3_i) / wnorm(u_lump[dn], w3_i),
                        relL2_lump=wnorm(err, m3) / wnorm(u_lump[dn], m3),
                        relL2_hom=wnorm(uh_r - u_hom, m3) / wnorm(u_hom, m3),
                        t_full_solve_s=t_full, t_reduced_solve_s=t_red, **cost))
    for rw in rows:
        rw["t_shape_s"] = time.time() - t0
    return rows


def guarded(args):
    """Skip a (shape, nu) whose part file appeared meanwhile (another instance finished it)."""
    t, path = args
    if os.path.exists(path):
        return None
    return task(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", default="0.3,0.45,0.49")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--shapes", default="", help="e.g. c_tube:0,torus:0 (default: all 8 families x 2)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--reverse", action="store_true", help="work through the task list from the end (2nd instance)")
    a = ap.parse_args()
    nus = [float(v) for v in a.nu.split(",")]
    shapes = ([(x.split(":")[0], int(x.split(":")[1])) for x in a.shapes.split(",")] if a.shapes
              else [(f, i) for f in E.FAMILIES for i in range(2)])
    tag = a.tag or "all"
    parts = os.path.join(E.OUT, f"e1_v2_parts_{tag}")                    # one csv per finished (shape, nu) -> resumable
    os.makedirs(parts, exist_ok=True)
    part = lambda f, i, nu: os.path.join(parts, f"{f}_{i}_nu{nu:g}.csv")
    tasks = [(f, i, [nu]) for f, i in shapes for nu in nus]              # one task per (shape, nu)
    todo = [t for t in tasks if not os.path.exists(part(t[0], t[1], t[2][0]))]
    todo = todo[::-1] if a.reverse else todo
    print(f"{len(tasks) - len(todo)} tasks already done, {len(todo)} to run", flush=True)
    t0 = time.time()
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(guarded, [(t, part(t[0], t[1], t[2][0])) for t in todo]):
            if r is None:
                continue
            pd.DataFrame(r).to_csv(part(r[0]["family"], r[0]["idx"], r[0]["nu"]), index=False)
            print(r[0]["family"], r[0]["idx"], r[0]["nu"], f"{r[0]['t_shape_s']:.0f}s", flush=True)
    df = pd.concat([pd.read_csv(part(f, i, nu[0])) for f, i, nu in tasks])
    df.to_csv(os.path.join(E.OUT, f"e1_v2_{tag}.csv"), index=False)
    fmt = lambda v: f"{v:.3g}"
    lines = [f"{len(tasks)} shapes, {time.time() - t0:.0f}s"]
    for col in ("ratio_surf", "ratio_int"):
        lines.append(f"\n== {col} (max over shapes, loads) ==")
        lines.append(df.pivot_table(index=["nu", "basis"], columns="depth", values=col, aggfunc="max")[list(DEPTHS)].to_string(float_format=fmt))
        lines.append(f"-- {col} (mean) --")
        lines.append(df.pivot_table(index=["nu", "basis"], columns="depth", values=col, aggfunc="mean")[list(DEPTHS)].to_string(float_format=fmt))
    lines.append("\n== pass rate ratio_surf<=0.2 & ratio_int<=0.2 ==")
    df["pass"] = (df.ratio_surf <= 0.2) & (df.ratio_int <= 0.2)
    lines.append(df.pivot_table(index=["nu", "basis"], columns="depth", values="pass", aggfunc="mean")[list(DEPTHS)].to_string(float_format=fmt))
    lines.append("\n== relL2 (mean) ==")
    lines.append(df.groupby(["nu", "basis"])[["relL2_hom", "relL2_lump", "span_modes"]].mean().to_string(float_format=fmt))
    lines.append("\n== lump signal size (mean) ==")
    lines.append(df[df.basis == "neo"].pivot_table(index="nu", columns="depth", values=["signal_surf", "signal_int"]).to_string(float_format=fmt))
    lines.append("\n== cost per shape and nu (median, seconds; 8 jobs in parallel, 2 threads each) ==")
    c = df.drop_duplicates(["family", "idx", "nu", "load"])
    cc = ["t_neo_s", "t_replicate_s", "t_rotfields_s", "t_orth_rr_s", "t_lu_s", "t_iter1_s", "t_iter2_s",
          "t_elastic_eigs_s", "t_assemble_K_s", "t_full_solve_s"]
    lines.append(c[cc].median().to_string(float_format=fmt))
    red = df[df.basis == "neo_rot_inv2_sc"].groupby(["family", "idx", "nu"]).t_reduced_solve_s.median()
    cs = c.groupby(["family", "idx", "nu"])[cc].median()
    cs["t_basis_c"] = cs[["t_neo_s", "t_replicate_s", "t_rotfields_s", "t_orth_rr_s", "t_lu_s", "t_iter1_s", "t_iter2_s"]].sum(1)
    cs["t_basis_c_plus_sc"] = cs.t_basis_c + cs.t_full_solve_s        # the static-correction vector costs one full solve
    cs["t_reduced"] = red
    cs["gain_per_solve"] = cs.t_full_solve_s - cs.t_reduced
    cs["breakeven_new_coef"] = np.ceil(cs.t_basis_c_plus_sc / cs.gain_per_solve.clip(lower=1e-9))
    lines.append("-- break-even: number of solves with new coefficients E(x) (same load) after which (c) is cheaper --")
    lines.append(cs[["t_basis_c", "t_full_solve_s", "t_reduced", "breakeven_new_coef"]].to_string(float_format=fmt))
    lines.append("\n== ANP spectrum per shape (n_zero = rigid zeros, should be 6) ==")
    lines.append(df.drop_duplicates(["family", "idx", "nu"]).pivot_table(index=["family", "idx"], columns="nu",
                 values=["n_zero", "lam7", "lam12"]).to_string(float_format=fmt))
    lines.append("\n== shape indicators ==")
    lines.append(df.drop_duplicates(["family", "idx"])[["family", "idx", "korn", "thick", "dmax"]].to_string(float_format=fmt))
    txt = "\n".join(lines)
    open(os.path.join(E.OUT, f"e1_v2_{tag}_summary.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()

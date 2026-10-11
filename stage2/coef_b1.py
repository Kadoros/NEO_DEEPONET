"""Branch B-1: hidden conductivity field on ONE shape, recovered by direct optimisation through a
differentiable Galerkin model (no neural network) -> stage2/coef/b1/

Problem   u_t = div(kappa(x) grad u) in a volume, Neumann.  kappa = 1 + (K_HI - 1) * sum of smooth ball
          indicators ("stiff lumps", radius 0.15-0.25, random positions); the model is never given kappa.
Truth     Delaunay-P1 operator on N = 8192 quasi-uniform points (volop), per-tet kappa_e = kappa(centroid);
          u(t) = expm_multiply(-t M^{-1} K_kappa, u0) for 8 random Gaussian ICs at 6 times; the observation is
          the full interior field at all points and times.
Model     Y = first M_MODES eigenvectors of the homogeneous operator (ARPACK, M-orthonormal).
          A_hat(theta) = Y^T K(kappa_hat) Y = sum_e kappa_e vol_e (grad Y_e)^T (grad Y_e)   (one weighted Gram)
          kappa_hat = exp(theta) per point, kappa_e = mean over the tet's vertices
          c(t) = expm(-t A_hat) c0,  c0 = Y^T M u0;  data term on the projected observations c_obs = Y^T M u_obs
          loss = sum ||c(t) - c_obs(t)||^2 / sum ||c_obs||^2 + REG * R(theta)
          R = theta^T K1 theta / sum(m)                      (--regtype h1, default)
          R = sum_e vol_e sqrt(|grad theta_e|^2 + eps^2) / sum(m)   (--regtype tv; keeps sharp lump edges)
Reports   kappa_hat vs kappa (mass-weighted relative L2), lump localisation (threshold + connected components on
          the Delaunay edges, matched to true centres), slice figures.

    CUDA_VISIBLE_DEVICES=<gpu> python stage2/coef_b1.py [--seed S] [--reg R]
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
for var, sub in [("MPLCONFIGDIR", "mpl"), ("XDG_CACHE_HOME", ""), ("TMPDIR", "tmp"), ("CUDA_CACHE_PATH", "nv")]:
    os.environ[var] = os.path.join(ROOT, ".cache", sub)
    os.makedirs(os.environ[var], exist_ok=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as sla  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import neo_vol_data as D  # noqa: E402
from volop import EDGES, delaunay_p1  # noqa: E402

OUT = os.path.join(ROOT, "coef", "b1")
N = 8192
M_MODES = 128
K_HI = 5.0
N_IC = 8
TV_EPS = 0.05
TAUS = np.geomspace(0.01, 1.0, 6)              # t = tau / lambda_1 (homogeneous)


def smooth_ball(x, c, r, w=0.02):
    return 1 / (1 + np.exp(-(r - np.linalg.norm(x - c, axis=1)) / w))


def build_problem(seed, n_blobs, family="sh_ball", key=(2026, 9)):
    """key/family defaults = B-1; B-2 passes its own key so its shapes never coincide with B-1's."""
    rng = np.random.default_rng([*key, seed])
    sdf0, bound = D.make_sdf(family, rng)
    R = Rotation.random(random_state=rng.integers(2 ** 31)).as_matrix()
    sdf = D.rotated(sdf0, R)
    geo = D.geometry(sdf, bound * 1.05, rng)
    for _ in range(5):
        P, is_s, h = D.sample_points(sdf, geo, N, "quasi", rng)
        c = P.mean(0)
        s = 1.0 / np.abs(P - c).max()
        Y = (P - c) * s
        sdf_n = lambda y: s * sdf(y / s + c)
        op = delaunay_p1(Y, is_s, sdf_n, h * s)
        if op.info["isolated"] == 0:
            break
    # stiff lumps: centres inside with clearance, radii 0.15-0.25, not overlapping
    centres, radii = [], []
    tries = 0
    while len(centres) < n_blobs:
        tries += 1
        if tries > 20000:                    # thin shape: no room for the lumps (never hit by B-1's shapes)
            raise RuntimeError(f"cannot place {n_blobs} lumps")
        cand = Y[rng.integers(len(Y))]
        r = rng.uniform(0.15, 0.25)
        if sdf_n(cand[None])[0] > r + 0.05 and all(np.linalg.norm(cand - c2) > r + r2 + 0.05
                                                   for c2, r2 in zip(centres, radii)):
            centres.append(cand); radii.append(r)
    kappa = lambda x: 1 + (K_HI - 1) * np.clip(sum(smooth_ball(x, c2, r2) for c2, r2 in zip(centres, radii)), 0, 1)
    return dict(Y=Y, op=op, sdf=sdf_n, centres=np.array(centres), radii=np.array(radii), kappa=kappa, rng=rng)


def wquantile(x, w, q):
    o = np.argsort(x)
    return float(np.interp(q, np.cumsum(w[o]) / w.sum(), x[o]))


def robust_threshold(kappa_hat, m):
    """Half-way between the background (mass-weighted median) and the robust peak (mass-weighted 99th percentile);
    uses no knowledge of the true kappa (a single-point spike cannot move it, unlike the max)."""
    bg, pk = wquantile(kappa_hat, m, 0.5), wquantile(kappa_hat, m, 0.99)
    return bg + 0.5 * (pk - bg)


def lump_localisation(Y, op, kappa_hat, centres, radii, thr):
    """Connected components (Delaunay edges) of {kappa_hat > thr}; match component centroids to true centres.
    matched = Hungarian pair with centroid inside the true lump radius; false = components left unmatched."""
    T = op.tets
    rows = np.concatenate([T[:, a] for a, b in EDGES])
    cols = np.concatenate([T[:, b] for a, b in EDGES])
    hot = kappa_hat > thr
    keep = hot[rows] & hot[cols]
    G = sp.coo_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(len(Y), len(Y)))
    ncomp, lab = connected_components(G, directed=False)
    comps = [np.where((lab == k) & hot)[0] for k in range(ncomp)]
    comps = [cmp for cmp in comps if len(cmp) >= 5]                   # ignore specks
    if not comps:
        return dict(n_components=0, n_matched=0, n_false=0, errors=[np.nan] * len(centres))
    w = op.m
    cents = np.array([(Y[cmp] * w[cmp, None]).sum(0) / w[cmp].sum() for cmp in comps])
    Dm = np.linalg.norm(centres[:, None] - cents[None], axis=2)
    ri, ci = linear_sum_assignment(Dm)
    err = [np.nan] * len(centres)
    for a, b in zip(ri, ci):
        err[a] = float(Dm[a, b])
    n_matched = int(sum(Dm[a, b] < radii[a] for a, b in zip(ri, ci)))
    return dict(n_components=len(comps), n_matched=n_matched, n_false=len(comps) - n_matched, errors=err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blobs", type=int, default=2)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--modes", type=int, default=M_MODES)
    ap.add_argument("--regtype", choices=["h1", "tv"], default="h1")
    ap.add_argument("--rescore", action="store_true", help="recompute localisation from a saved fields.npz only")
    a = ap.parse_args()
    assert a.rescore or os.environ.get("CUDA_VISIBLE_DEVICES"), "set CUDA_VISIBLE_DEVICES"
    tag = f"seed{a.seed}_blobs{a.blobs}_reg{a.reg:g}_noise{a.noise:g}"
    tag += (f"_m{a.modes}" if a.modes != M_MODES else "") + ("_tv" if a.regtype == "tv" else "")
    out = os.path.join(OUT, tag)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    pb = build_problem(a.seed, a.blobs)
    Y, op, rng = pb["Y"], pb["op"], pb["rng"]
    if a.rescore:
        f = np.load(os.path.join(out, "fields.npz"))
        assert np.allclose(f["Y"], Y)
        thr = robust_threshold(f["kappa_hat"], op.m)
        loc = lump_localisation(Y, op, f["kappa_hat"], pb["centres"], pb["radii"], thr)
        r = json.load(open(os.path.join(out, "result.json")))
        r.pop("n_found", None)
        r.update(threshold=thr, **loc)
        json.dump(r, open(os.path.join(out, "result.json"), "w"), indent=2)
        print(tag, loc)
        return
    m = op.m
    cen = Y[op.tets].mean(1)
    kappa_e = pb["kappa"](cen)
    K1 = op.K.tocsc()
    Kk = op.assemble(kappa_e).tocsc()
    lam, Phi = sla.eigsh(K1, k=a.modes, M=sp.diags(m).tocsc(), sigma=-1e-6)
    o = np.argsort(lam); lam, Phi = lam[o], Phi[:, o]
    t = TAUS / lam[1]
    # truth: full heterogeneous operator, exact in time
    U0 = []
    for _ in range(N_IC):
        cidx = rng.integers(len(Y))
        U0.append(np.exp(-((Y - Y[cidx]) ** 2).sum(1) / (2 * 0.25 ** 2)))
    U0 = np.stack(U0, 1)                                                     # (N, I)
    A = (sp.diags(1 / m) @ Kk).tocsr()
    Uobs = np.stack([sla.expm_multiply(-tj * A, U0) for tj in t])            # (T, N, I)
    if a.noise > 0:
        Uobs = Uobs + a.noise * np.abs(Uobs).max() * rng.normal(size=Uobs.shape)
    t_truth = time.time() - t0

    dev = "cuda"
    f64 = torch.float64
    GY = np.einsum("tki,tim->tkm", op.grads, Phi[op.tets])                 # (T, 3, m) gradients of Y per tet
    GY_t = torch.tensor(GY, dtype=f64, device=dev)
    vol_t = torch.tensor(op.vol, dtype=f64, device=dev)
    tets_t = torch.tensor(op.tets, device=dev)
    grads_t = torch.tensor(np.transpose(op.grads, (0, 2, 1)), dtype=f64, device=dev)   # (T, 4, 3)
    c0 = torch.tensor(Phi.T @ (m[:, None] * U0), dtype=f64, device=dev)       # (m, I)
    cobs = torch.tensor(np.einsum("nm,n,tni->tmi", Phi, m, Uobs), dtype=f64, device=dev)
    tt = torch.tensor(t, dtype=f64, device=dev)
    K1s = K1.tocoo()
    K1_t = torch.sparse_coo_tensor(np.vstack([K1s.row, K1s.col]), K1s.data, K1s.shape, dtype=f64, device=dev)
    msum = float(m.sum())

    def forward(kap_e):
        Ahat = torch.einsum("t,tkm,tkn->mn", kap_e * vol_t, GY_t, GY_t)
        Ahat = 0.5 * (Ahat + Ahat.T)
        E = torch.linalg.matrix_exp(-tt[:, None, None] * Ahat[None])          # (T, m, m)
        return E @ c0[None]                                                  # (T, m, I)

    def data_loss(kap_e):
        return ((forward(kap_e) - cobs) ** 2).sum() / (cobs ** 2).sum()

    with torch.no_grad():
        oracle = data_loss(torch.tensor(kappa_e, dtype=f64, device=dev)).item()
        const1 = data_loss(torch.ones(len(op.vol), dtype=f64, device=dev)).item()

    theta = torch.zeros(len(Y), dtype=f64, device=dev, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    hist = []
    for it in range(a.iters + 1):
        kap_e = torch.exp(theta)[tets_t].mean(1)
        dl = data_loss(kap_e)
        if a.regtype == "h1":
            reg = a.reg * (theta @ torch.sparse.mm(K1_t, theta[:, None])[:, 0]) / msum
        else:
            g = torch.einsum("tki,tk->ti", grads_t, theta[tets_t])
            reg = a.reg * (vol_t * torch.sqrt((g ** 2).sum(1) + TV_EPS ** 2)).sum() / msum
        loss = dl + reg
        if it % 100 == 0:
            hist.append(dict(it=it, data=dl.item(), reg=reg.item()))
        if it == a.iters:
            break
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    kap_hat = np.exp(theta.detach().cpu().numpy())                          # per point
    kap_true = pb["kappa"](Y)
    rel_l2 = float(np.sqrt((m * (kap_hat - kap_true) ** 2).sum() / (m * kap_true ** 2).sum()))
    rel_l2_contrast = float(np.sqrt((m * (kap_hat - kap_true) ** 2).sum() / (m * (kap_true - 1) ** 2).sum()))
    thr = robust_threshold(kap_hat, m)
    loc = lump_localisation(Y, op, kap_hat, pb["centres"], pb["radii"], thr)
    res = dict(tag=tag, n_points=len(Y), n_tets=len(op.vol), modes=a.modes, regtype=a.regtype, K_HI=K_HI, taus=TAUS.tolist(),
               centres=pb["centres"].tolist(), radii=pb["radii"].tolist(), oracle_data_loss=oracle,
               const1_data_loss=const1, final=hist[-1], rel_l2=rel_l2, rel_l2_contrast=rel_l2_contrast,
               kappa_hat_max=float(kap_hat.max()), kappa_hat_in_lumps=float(np.average(kap_hat[kap_true > 3], weights=m[kap_true > 3])),
               kappa_hat_background=float(np.average(kap_hat[kap_true < 1.1], weights=m[kap_true < 1.1])),
               threshold=float(thr), **loc, t_truth_s=t_truth, t_total_s=time.time() - t0)
    json.dump(dict(res, history=hist), open(os.path.join(out, "result.json"), "w"), indent=2)
    np.savez(os.path.join(out, "fields.npz"), Y=Y, kappa_true=kap_true, kappa_hat=kap_hat, m=m)

    # slices through each lump centre (plane normal = z), points within +-h of the plane
    h = np.median(np.linalg.norm(Y[op.tets[:, 0]] - Y[op.tets[:, 1]], axis=1))
    nb = len(pb["centres"])
    fig, axes = plt.subplots(nb, 2, figsize=(9, 4.2 * nb), constrained_layout=True, squeeze=False)
    vmax = max(K_HI, kap_hat.max())
    for i, cc in enumerate(pb["centres"]):
        sel = np.abs(Y[:, 2] - cc[2]) < h
        for j, (f, name) in enumerate([(kap_true, "true κ"), (kap_hat, "estimated κ̂")]):
            ax = axes[i, j]
            sc = ax.scatter(Y[sel, 0], Y[sel, 1], c=f[sel], s=14, cmap="viridis", vmin=1, vmax=vmax)
            for c2, r2 in zip(pb["centres"], pb["radii"]):
                if abs(c2[2] - cc[2]) < r2:
                    rr = np.sqrt(r2 ** 2 - (c2[2] - cc[2]) ** 2)
                    ax.add_patch(plt.Circle(c2[:2], rr, fill=False, color="#e34948", lw=1.2))
            ax.set_aspect("equal"); ax.set_title(f"slice z = {cc[2]:.2f} · {name}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=axes[i, :].tolist(), shrink=0.8)
    fig.suptitle(f"B-1 {tag}: rel L2 {rel_l2:.3f}, lumps matched {loc['n_matched']}/{len(pb['centres'])} (+{loc['n_false']} false), centre errors "
                 f"{[round(e, 3) if e == e else None for e in loc['errors']]}  (red circles: true lumps)", fontsize=10)
    fig.savefig(os.path.join(out, "slices.png"), dpi=130)
    plt.close(fig)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()

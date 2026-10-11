"""Gate experiment data (narrow training, wide testing) -> stage3/gate/data/<split>/<name>.npz

Physics  volume heat u_t = Laplace u, Neumann, point-cloud Delaunay-P1 operator (K, lumped M), kappa = 1;
         truth = exact discrete solution exp(-t M^{-1} K) u0 (shift-invert block Krylov, c_heat.fine_heat).
IC       Gaussian bump exp(-|x - x0|^2 / (2 sigma^2)), x0 = a random cloud point, sigma ~ U(0.15, 0.35)
         (so both models see the same (x0, sigma) description; Point-DeepONet gets them as global conditions).
Times    tau = t lambda_1 in {0.01, 0.03, 0.1, 0.3, 1, 3}.
Splits
  train      stage-2 B-2 shapes (sh_ball, superquadric; coef/b2_data/train, 160), 2 ICs each
  val        B-2 test shapes 0-4 of each family (10), 4 ICs         (model selection only)
  test_in    B-2 test shapes 5-9 of each family (10), 4 ICs         (training families, unseen shapes)
  test_ood   torus, cup, c_tube, multihole, 5 shapes each (stage-3 E1 generator), 4 ICs
  test_real  the 13 real shapes of C (cleaned quasi-8192 clouds), 4 ICs
Per shape stored: x, m, d (distance to the nearest boundary node, used as the SDF input of Point-DeepONet),
  NEO raw features F_hat (fp16) and z (stage-1 normalisation), NEO Rayleigh-Ritz basis Phi (256) and Ritz values,
  coef = Phi^T M u0, r = u0 - Phi coef, r_p = (M + t K)^{-1} M r, u0, U (truth), IC parameters, t.

    CUDA_VISIBLE_DEVICES=<gpu> python stage3/gate_data.py [--procs 10]
"""
import argparse
import glob
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.spatial import cKDTree

import e1_basis as E

OUT = os.path.join(E.ROOT, "gate", "data")
B2 = os.path.join(E.S2, "coef", "b2_data")
RS = os.path.join(E.ROOT, "real_shapes")
TAUS = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
OOD_FAMILIES = ["torus", "cup", "c_tube", "multihole"]
SEED = 20260930


def stiffness(tets, grads, vol, n):
    Kl = vol[:, None, None] * np.einsum("tki,tkj->tij", grads, grads)
    return sp.coo_matrix((Kl.ravel(), (np.repeat(tets, 4, 1).ravel(), np.tile(tets, (1, 4)).ravel())), shape=(n, n)).tocsr()


def sources():
    """(split, name, loader) list; loader returns Y, tets, grads, vol, m."""
    out = []
    tr = sorted(glob.glob(os.path.join(B2, "train", "*.npz")))
    out += [("train", os.path.basename(p)[:-4], ("b2", p)) for p in tr]
    te = sorted(glob.glob(os.path.join(B2, "test", "*.npz")))
    for p in te:
        fam, idx = os.path.basename(p)[:-4].rsplit("_", 1)
        out.append(("val" if int(idx) < 5 else "test_in", os.path.basename(p)[:-4], ("b2", p)))
    for fam in OOD_FAMILIES:
        for i in range(5):
            out.append(("test_ood", f"{fam}_{i}", ("e1", (fam, i))))
    for p in sorted(glob.glob(os.path.join(RS, "cache", "*_quasi_8192.npz"))):
        out.append(("test_real", os.path.basename(p)[:-len("_quasi_8192.npz")], ("real", p)))
    return out


def load(src):
    kind, p = src
    if kind == "e1":
        fam, i = p
        if not os.path.exists(E.cache_path(fam, i)):
            E.prep_task((fam, i))                                        # stage-3 E1 generator (seeded)
        p = E.cache_path(fam, i)
    z = np.load(p)
    Y = z["Y"].astype(np.float64)
    return Y, z["tets"].astype(np.int64), z["grads"].astype(np.float64), z["vol"].astype(np.float64), z["m"].astype(np.float64)


def neo_features(items):
    """GPU pass: raw NEO features (stage-1 normalisation) and the M-orthonormal NEO output per shape."""
    import torch
    from g2pt.training.pretrain_tet import PretrainTetTraining
    from g2pt.utils.ortho_operations import qr_orthogonalization
    model = PretrainTetTraining.load_from_checkpoint(E.CKPT, weights_only=False).model.to("cuda").eval()
    res = {}
    for key, Y, m in items:
        with torch.no_grad():
            x = torch.tensor(Y, dtype=torch.float32, device="cuda")[None]
            mass = torch.tensor(m, dtype=torch.float32, device="cuda").view(1, -1, 1)
            F = model(x, x, mass)
            S = qr_orthogonalization(F, mass)[0].double().cpu().numpy()
            F = F[0].double().cpu().numpy()
        sF = np.sqrt((m[:, None] * F ** 2).sum() / (m.sum() * F.shape[1]))
        Fh = F / sF
        res[key] = (Fh.astype(np.float16), (m[:, None] * Fh).sum(0) / m.sum(), S)
    return res


def cpu_task(args):
    split, name, src, Fh, zf, S = args
    import c_heat as CH
    t0 = time.time()
    Y, tets, grads, vol, m = load(src)
    K = stiffness(tets, grads, vol, len(Y)).tocsc()
    A, B = S.T @ (K @ S), (S * m[:, None]).T @ S
    lam, W = la.eigh(0.5 * (A + A.T), 0.5 * (B + B.T))
    Phi = S @ W                                                        # NEO Rayleigh-Ritz basis (raw, 256)
    lam1 = float(np.sort(sla.eigsh(K, k=3, M=sp.diags(m).tocsc(), sigma=-1e-6)[0])[1])
    t = TAUS / lam1
    n_ic = 2 if split == "train" else 4
    rng = np.random.default_rng([SEED, sum(map(ord, name)), ["train", "val", "test_in", "test_ood", "test_real"].index(split)])
    x0 = Y[rng.choice(len(Y), n_ic, replace=False)]
    sig = rng.uniform(0.15, 0.35, n_ic)
    u0 = np.stack([np.exp(-((Y - c) ** 2).sum(1) / (2 * s ** 2)) for c, s in zip(x0, sig)])        # (I, N)
    U, _ = CH.fine_heat(K, m, u0.T, t, lam1)                                                       # (T, N, I)
    U = U.transpose(0, 2, 1)                                                                        # (T, I, N)
    coef = Phi.T @ (m[:, None] * u0.T)                                                              # (K, I)
    r = u0 - (Phi @ coef).T                                                                         # (I, N)
    Msp = sp.diags(m)
    rp = np.stack([sla.splu((Msp + tj * K).tocsc()).solve((m[:, None] * r.T)).T for tj in t])      # (T, I, N)
    ws = np.zeros(len(Y))
    F_, nrm = E.boundary_faces(tets, Y)
    ws[np.unique(F_)] = 1
    d = cKDTree(Y[ws > 0]).query(Y)[0]
    np.savez(os.path.join(OUT, split, f"{name}.npz"), x=Y.astype(np.float32), m=m.astype(np.float32), d=d.astype(np.float32),
             F=Fh, z=zf.astype(np.float32), Phi=Phi.astype(np.float32), lam=lam.astype(np.float32), coef=coef.astype(np.float32),
             u0=u0.astype(np.float32), U=U.astype(np.float32), r=r.astype(np.float32), rp=rp.astype(np.float32),
             t=t.astype(np.float32), x0=x0.astype(np.float32), sigma=sig.astype(np.float32), lam1=lam1)
    return f"{split}/{name} {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    for s in ("train", "val", "test_in", "test_ood", "test_real"):
        os.makedirs(os.path.join(OUT, s), exist_ok=True)
    src = [x for x in sources() if not os.path.exists(os.path.join(OUT, x[0], f"{x[1]}.npz"))]
    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(a.procs) as pool:                                      # build missing E1 shapes on CPU first
        need = [x[2] for x in src if x[2][0] == "e1" and not os.path.exists(E.cache_path(*x[2][1]))]
        pool.map(load, need)
        print(f"E1 shapes built: {len(need)} ({time.time() - t0:.0f}s)", flush=True)
        geo = pool.map(load, [x[2] for x in src])
        feats = neo_features([(i, g[0], g[4]) for i, g in enumerate(geo)])
        print(f"NEO features: {len(feats)} ({time.time() - t0:.0f}s)", flush=True)
        jobs = [(x[0], x[1], x[2]) + feats[i] for i, x in enumerate(src)]
        for msg in pool.imap_unordered(cpu_task, jobs):
            print(msg, flush=True)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

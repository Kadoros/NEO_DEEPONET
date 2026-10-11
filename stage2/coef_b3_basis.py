"""Branch B-3: alternative Galerkin bases for the B-2 shapes -> coef/b3_basis/<split>/<name>_<basis>.npz

  arpack256  256 lowest eigenvectors of the homogeneous Delaunay-P1 operator (ARPACK shift-invert), M-orthonormal
  neo256     volume NEO (main/t4096, best.ckpt) 256-dim output on the shape's points and lumped mass
             -> M-orthonormalised (repo qr_orthogonalization) -> Rayleigh-Ritz with the homogeneous stiffness
             (Ritz vectors sorted by Ritz value; M-orthonormal)
Also stored: span loss of the first 96 / 128 ARPACK eigenvectors inside the NEO span (basis quality check).

    python stage2/coef_b3_basis.py arpack --procs 12            (CPU)
    CUDA_VISIBLE_DEVICES=<gpu> python stage2/coef_b3_basis.py neo
"""
import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
for var, sub in [("MPLCONFIGDIR", "mpl"), ("XDG_CACHE_HOME", ""), ("TMPDIR", "tmp"), ("TRITON_CACHE_DIR", "triton"),
                 ("TORCH_HOME", "torch"), ("CUDA_CACHE_PATH", "nv")]:
    os.environ[var] = os.path.join(ROOT, ".cache", sub)
    os.makedirs(os.environ[var], exist_ok=True)

if len(sys.argv) > 1 and sys.argv[1] == "neo":
    import open3d  # noqa: F401  -- must precede h5py/torch/lightning in this env

from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import scipy.linalg as la  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as sla  # noqa: E402

DATA = os.path.join(ROOT, "coef", "b2_data")
OUT = os.path.join(ROOT, "coef", "b3_basis")
SPLITS = ["train", "test", "test_ood"]
CKPT = os.path.join(ROOT, "main", "t4096", "best.ckpt")


def stiffness(z):
    T, G, vol = z["tets"].astype(np.int64), z["grads"].astype(np.float64), z["vol"]
    n = len(z["Y"])
    Ke = vol[:, None, None] * np.einsum("tki,tkj->tij", G, G)
    return sp.coo_matrix((Ke.ravel(), (np.repeat(T, 4, 1).ravel(), np.tile(T, (1, 4)).ravel())), shape=(n, n)).tocsr()


def out_path(p, basis):
    split = os.path.basename(os.path.dirname(p))
    os.makedirs(os.path.join(OUT, split), exist_ok=True)
    return os.path.join(OUT, split, os.path.basename(p)[:-4] + f"_{basis}.npz")


def arpack_task(p):
    t0 = time.time()
    z = np.load(p)
    K, m = stiffness(z), z["m"]
    lam, Phi = sla.eigsh(K.tocsc(), k=256, M=sp.diags(m).tocsc(), sigma=-1e-6)
    o = np.argsort(lam)
    return p, lam[o], Phi[:, o].astype(np.float32), time.time() - t0


def run_arpack(procs):
    paths = sorted(sum([glob.glob(os.path.join(DATA, s, "*.npz")) for s in SPLITS], []))
    with Pool(procs) as pool:
        for p, lam, Phi, dt in pool.imap_unordered(arpack_task, paths):
            np.savez(out_path(p, "arpack256"), lam=lam, Phi=Phi)
            print(f"{p[len(DATA) + 1:]} {dt:.0f}s", flush=True)


def run_neo():
    import torch
    from g2pt.training.pretrain_tet import PretrainTetTraining
    from g2pt.utils.ortho_operations import qr_orthogonalization
    model = PretrainTetTraining.load_from_checkpoint(CKPT, weights_only=False).model.to("cuda").eval()
    paths = sorted(sum([glob.glob(os.path.join(DATA, s, "*.npz")) for s in SPLITS], []))
    for p in paths:
        t0 = time.time()
        z = np.load(p)
        m = z["m"]
        with torch.no_grad():
            x = torch.tensor(z["Y"], dtype=torch.float32, device="cuda")[None]
            mass = torch.tensor(m, dtype=torch.float32, device="cuda").view(1, -1, 1)
            S = qr_orthogonalization(model(x, x, mass), mass)[0].double().cpu().numpy()
        K = stiffness(z)
        A, B = S.T @ (K @ S), (S * m[:, None]).T @ S
        mu, V = la.eigh(0.5 * (A + A.T), 0.5 * (B + B.T))
        Q = S @ V                                                            # M-orthonormal Ritz vectors
        P128 = z["Phi"].astype(np.float64)                                   # ARPACK 128 from the B-2 data
        span = {k: float(1 - ((Q.T @ (m[:, None] * P128[:, :k])) ** 2).sum() / k) for k in (96, 128)}
        np.savez(out_path(p, "neo256"), lam=mu, Phi=Q.astype(np.float32), span96=span[96], span128=span[128])
        print(f"{p[len(DATA) + 1:]} span96 {span[96]:.2e} span128 {span[128]:.2e} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "arpack":
        run_arpack(int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 12)
    else:
        run_neo()

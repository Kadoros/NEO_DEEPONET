"""Evaluate the NEO volume pilot runs -> stage2/pilot/eval.csv, pilot/tables.md

Models: each pilot run (best.ckpt) + the surface Ours-Large checkpoint used zero-shot on volumes.
Test sets: data_neo/{test_in,test_out}/{quasi,random}_{2048,8192} (test_out = c_tube, never seen in training).
Per sample (points, lumped mass m, operator K from ops/*.npz; target eigenvectors from the h5):
  span loss    1 - ||Q^T M U_96||_F^2 / 96, Q = the model's 256-dim span made M-orthonormal
               (Rayleigh-Ritz eigenvectors), U_96 = Delaunay-P1 eigenvectors (targets)
  RR eig error Rayleigh-Ritz of the same K in the model span; mean |lambda_RR / lambda - 1| over i=1..95
               and over i=1..10

    CUDA_VISIBLE_DEVICES=<gpu> python stage2/pilot_eval.py
"""
import glob
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
for var, sub in [("MPLCONFIGDIR", "mpl"), ("XDG_CACHE_HOME", ""), ("TMPDIR", "tmp"), ("TRITON_CACHE_DIR", "triton"),
                 ("TORCH_HOME", "torch"), ("CUDA_CACHE_PATH", "nv")]:
    os.environ[var] = os.path.join(ROOT, ".cache", sub)
    os.makedirs(os.environ[var], exist_ok=True)

import open3d  # noqa: F401,E402  -- must precede h5py/torch/lightning in this env
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.linalg as la  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

from g2pt.training.pretrain import PretrainTraining  # noqa: E402
from g2pt.training.pretrain_tet import PretrainTetTraining  # noqa: E402
from g2pt.utils.ortho_operations import qr_orthogonalization  # noqa: E402

DATA = os.path.join(ROOT, "data_neo")
PILOT = os.path.join(ROOT, "pilot")
CKPT_SURF = os.path.join(os.path.dirname(ROOT), "neo_check", "ckpts", "Ours-Large.ckpt")
K = 96


def models():
    out = {"surface Large (zero-shot)":
           PretrainTraining.load_from_checkpoint(CKPT_SURF, weights_only=False, strict=False).model}
    for run in sorted(os.listdir(PILOT)):
        ck = os.path.join(PILOT, run, "best.ckpt")
        if os.path.exists(ck):
            out[run] = PretrainTetTraining.load_from_checkpoint(ck, weights_only=False).model
    return {k: m.to("cuda").eval() for k, m in out.items()}


def samples(split, sub):
    hf = h5py.File(os.path.join(DATA, split, sub, "vol_000_samples_evecs.hdf5"), "r")
    names = [n.decode() for n in h5py.File(os.path.join(DATA, split, sub, "vol_000_mesh.hdf5"), "r")["obj_paths"][:]]
    kind, N = sub.split("_")
    for i, name in enumerate(names):
        op = np.load(os.path.join(DATA, split, "ops", f"{name}_{kind}_{N}.npz"))
        Kop = sp.csr_matrix((op["K_data"], op["K_indices"], op["K_indptr"]), shape=(int(N), int(N)))
        pts = hf["samples"][i, 0]
        assert np.allclose(pts, op["points"]), name
        yield name, pts, hf["evecs"][i, 0].astype(np.float64), op["m"], Kop, op["lam"]


@torch.no_grad()
def evaluate(model, pts, U, m, Kop, lam):
    x = torch.tensor(pts, dtype=torch.float32, device="cuda")[None]
    mass = torch.tensor(m, dtype=torch.float32, device="cuda").view(1, -1, 1)
    S = qr_orthogonalization(model(x, x, mass), mass)[0].double().cpu().numpy()
    A, B = S.T @ (Kop @ S), (S * m[:, None]).T @ S
    mu, Y = la.eigh(0.5 * (A + A.T), 0.5 * (B + B.T))
    Q = S @ Y                                                        # M-orthonormal Ritz vectors (256)
    span = 1 - ((Q.T @ (m[:, None] * U)) ** 2).sum() / K
    rel = mu[1:K] / lam[1:K] - 1
    return dict(span_loss=float(span), eig_relerr=float(np.abs(rel).mean()), eig_relerr_1to10=float(np.abs(rel[:10]).mean()),
                eig_bias=float(rel.mean()))


def main():
    M = models()
    rows = []
    for split in ("test_in", "test_out"):
        for sub in sorted(os.listdir(os.path.join(DATA, split))):
            if sub == "ops":
                continue
            for name, pts, U, m, Kop, lam in samples(split, sub):
                for mname, model in M.items():
                    r = evaluate(model, pts, U, m, Kop, lam)
                    rows.append(dict(model=mname, split=split, set=sub, family=name.rsplit("_", 1)[0], name=name, **r))
            print(f"{split}/{sub} done", flush=True)
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(PILOT, "eval.csv"), index=False)
    lines = []
    for metric in ("span_loss", "eig_relerr", "eig_relerr_1to10"):
        t = E.pivot_table(index="model", columns=["split", "set"], values=metric, aggfunc="mean")
        lines.append(f"## {metric} (mean)\n\n{t.to_string(float_format=lambda v: f'{v:.3g}')}\n")
    t = E[E.split == "test_in"].pivot_table(index="model", columns="family", values="span_loss", aggfunc="mean")
    lines.append(f"## span_loss by family (test_in, all sets)\n\n{t.to_string(float_format=lambda v: f'{v:.3g}')}\n")
    open(os.path.join(PILOT, "tables.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

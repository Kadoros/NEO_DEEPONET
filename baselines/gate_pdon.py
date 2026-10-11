"""Gate experiment, Point-DeepONet baseline (pdon env) -> stage3/gate/runs/pdon_N<N>_s<seed>/{result.json, test.csv}

The authors' code is used unchanged (point-deeponet/5.Point_DeepONet/main.py: DeepONetCartesianProd, define_model,
TripleCartesianProd, train_model with their defaults: 40000 iterations, batch 16, lr 1e-3, hidden 100, AdamW,
SiLU/sine encodings, MinMax scaling to [-1, 1], tanh output, their training loop incl. detect_anomaly).
Only the data loader is ours (replaces load_and_preprocess_data):
  one record = (shape, IC, tau); branch condition (5 dims, the slot of their 'mlc' = mass, load, 3 direction cosines)
  = [log10 tau, sigma, x0 (3)]; PointNet input = cloud xyz; trunk input = xyz + d (distance to the boundary, the
  slot of their SDF); output = u replicated into their 4 output channels (ux, uy, uz, vm) - channel 0 is evaluated.
  Their per-direction output clipping and hard-coded VTK evaluation are not used.  Scalers are fitted on the training
  records exactly as in their loader.  Seeds: our seed replaces their fixed 2024 (needed for 2 seeds).
Test: every test record predicted at the cloud's own size (PointNet max-pool + pointwise trunk accept any N);
relative L2 (lumped mass) on u, same metric as gate_ours.py.

    python baselines/gate_pdon.py --N 50 --seed 0 --gpu <physical id>      (env pdon)
"""
import argparse
import glob
import importlib.util
import json
import os
import random
import sys
import time
from functools import partial
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(ROOT, "point-deeponet", "5.Point_DeepONet")
DATA = os.path.join(os.path.dirname(ROOT), "stage3", "gate", "data")
RUNS = os.path.join(os.path.dirname(ROOT), "stage3", "gate", "runs")
if "--v2" in sys.argv:                                                # gate v2 (task B): other data / run folders
    DATA = DATA.replace(os.sep + "data", os.sep + "data_v2")
    RUNS = RUNS.replace(os.sep + "runs", os.sep + "runs_v2")
if "--gpu" in sys.argv:                                               # pin the GPU before torch touches CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, ".cache", "mpl"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

spec = importlib.util.spec_from_file_location("pdon_main", os.path.join(REPO, "main.py"))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)                                            # authors' module, unchanged


def records(split, names=None):
    """(name, cond (R,5), xyz (N,3), xyzd (N,4), u (R,N), m (N)) per shape, R = I x T records."""
    out = []
    for p in sorted(glob.glob(os.path.join(DATA, split, "*.npz"))):
        name = os.path.basename(p)[:-4]
        if names is not None and name not in names:
            continue
        z = np.load(p)
        t, sig, x0, U = z["t"] / 1.0, z["sigma"], z["x0"], z["U"]       # U (T, I, N)
        tau = t * float(z["lam1"])
        cond, u = [], []
        for i in range(len(sig)):
            for j in range(len(t)):
                cond.append([np.log10(tau[j]), sig[i], *x0[i]])
                u.append(U[j, i])
        xyz = z["x"].astype(np.float32)
        out.append((name, np.array(cond, np.float32), xyz, np.c_[xyz, z["d"]].astype(np.float32), np.array(u, np.float32), z["m"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--iterations", type=int, default=40000)          # authors' default
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--v2", action="store_true")                 # their get_device() sets CUDA_VISIBLE_DEVICES
    a = ap.parse_args()
    out = os.path.join(RUNS, f"pdon_N{a.N}_s{a.seed}")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    device = P.get_device(a.gpu)                                      # authors' helper: sets CUDA_VISIBLE_DEVICES = gpu
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(DATA, "train", "*.npz")))
    pick = sorted(np.random.default_rng([a.seed, a.N]).choice(len(names), a.N // 2, replace=False))   # same subset as ours
    tr = records("train", {names[i] for i in pick})
    # monitoring set for their training loop: 4 val shapes x first IC x 6 times = 24 records. Their loop pushes the
    # whole test set through the network at once (240 records x 8192 points does not fit a 24 GB GPU); it is only
    # logged (deepxde picks nothing from it), so its size does not change training.
    va = [(n, c[:6], xyz, xyzd, u[:6], m) for n, c, xyz, xyzd, u, m in records("val")[:4]]
    cat = lambda recs, k: np.concatenate([np.repeat(r[k][None], len(r[1]), 0) if k in (2, 3) else r[k] for r in recs])
    bc, pn, tk, u = cat(tr, 1), cat(tr, 2), cat(tr, 3), cat(tr, 4)
    y = np.repeat(u[..., None], 4, -1)                                 # their 4 output channels
    sc = partial(MinMaxScaler, feature_range=(-1, 1))                  # as in their loader
    bs, ps, ts, os_ = sc().fit(bc), sc().fit(pn.reshape(-1, 3)), sc().fit(tk.reshape(-1, 4)), sc().fit(y.reshape(-1, 4))
    f = lambda recs: (bs.transform(cat(recs, 1)).astype(np.float32),
                      ps.transform(cat(recs, 2).reshape(-1, 3)).reshape(-1, cat(recs, 2).shape[1], 3).astype(np.float32),
                      ts.transform(cat(recs, 3).reshape(-1, 4)).reshape(-1, cat(recs, 3).shape[1], 4).astype(np.float32))
    ytr = os_.transform(y.reshape(-1, 4)).reshape(y.shape).astype(np.float32)
    uva = cat(va, 4)
    yva = os_.transform(np.repeat(uva[..., None], 4, -1).reshape(-1, 4)).reshape(*uva.shape, 4).astype(np.float32)
    data = P.TripleCartesianProd(f(tr), ytr, f(va), yva)
    args = SimpleNamespace(N_pt=pn.shape[1], branch_hidden_dim=100, trunk_hidden_dim=100, fc_hidden_dim=100,
                           trunk_encoding_hidden_dim=100, learning_rate=1e-3, N_iterations=a.iterations, batch_size=16,
                           output_components="xyzs")                                  # their default (4 channels)
    model, scheduler, optimizer = P.define_model(args, device, data, os_)
    t0 = time.time()
    P.train_model(model, scheduler, optimizer, args, os_, out, device)
    t_train = time.time() - t0
    n_par = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    rows = []
    for split in ("test_in", "test_ood", "test_real"):
        recs = records(split)
        ti = time.time()
        for rec in recs:
            X = f([rec])
            pred = model.predict(X)
            pred = os_.inverse_transform(np.asarray(pred).reshape(-1, 4)).reshape(pred.shape)[..., 0]  # (R, N)
            m = rec[5][None]
            e = np.sqrt((m * (pred - rec[4]) ** 2).sum(1) / (m * rec[4] ** 2).sum(1))
            T = len(np.unique(rec[1][:, 0]))
            for k, ek in enumerate(e):
                rows.append(dict(split=split, shape=rec[0], tau_idx=k % T, ic=k // T, rel=float(ek)))
        t_inf = (time.time() - ti) / max(len(recs), 1)
        for r in rows:
            r.setdefault("t_infer_per_shape_s", t_inf)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "test.csv"), index=False)
    summ = df.groupby("split").rel.mean().to_dict()
    json.dump(dict(model="pdon", N=a.N, seed=a.seed, n_train_shapes=len(tr), params=n_par, t_train_s=t_train,
                   iterations=a.iterations, mean_rel=summ), open(os.path.join(out, "result.json"), "w"), indent=1)
    print(summ, f"train {t_train:.0f}s params {n_par}")


if __name__ == "__main__":
    main()

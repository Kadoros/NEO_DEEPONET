"""Branch B-2: a coefficient network trained across many shapes -> stage2/coef/b2/<run>/

For each shape the network sees only the observations (points, initial fields, the observed interior fields at
6 times) and predicts theta(x), kappa_hat = exp(theta).  It is trained WITHOUT the true kappa: the loss is the
same differentiable Galerkin data term as B-1 (A_hat = Y^T K(kappa_hat) Y, c(t) = expm(-t A_hat) c0, projected
misfit to the observations) plus the smoothness term of B-1.  True kappa is used only for evaluation.

Network (per point, N = 8192):
  input  x (3), u0 (8 ICs), u(t_j) (6 x 8), each IC normalised by max u0
  encoder MLP -> 128; L blocks of [local kNN aggregation (k = 16, max over neighbours of MLP(h_j - h_i, x_j - x_i))
  + global slice attention (Transolver-style, 32 slices)]; head MLP -> theta
Evaluation on unseen test shapes: relative L2, lump localisation (robust threshold, as B-1), data misfit,
and the same metrics after a short B-1-style refinement started from the network output.

    CUDA_VISIBLE_DEVICES=<gpu> python stage2/coef_b2.py --run base [--epochs 150] [--reg 1e-4] [--regtype h1]
        [--basis arpack128|arpack256|neo256]      (B-3: Galerkin basis Y; files from coef_b3_basis.py)
"""
import argparse
import glob
import json
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
for var, sub in [("MPLCONFIGDIR", "mpl"), ("XDG_CACHE_HOME", ""), ("TMPDIR", "tmp"), ("CUDA_CACHE_PATH", "nv")]:
    os.environ[var] = os.path.join(ROOT, ".cache", sub)
    os.makedirs(os.environ[var], exist_ok=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import coef_b1 as B1  # noqa: E402

DATA = os.path.join(ROOT, "coef", "b2_data")
OUT = os.path.join(ROOT, "coef", "b2")
DEV = "cuda"
F64 = torch.float64
KNN = 16
TV_EPS = B1.TV_EPS


class Shape:
    """One shape on the GPU: geometry, basis, observations, and the fixed parts of the Galerkin model."""

    def __init__(self, path, basis="arpack128"):
        z = np.load(path)
        self.name = os.path.basename(path)[:-4]
        self.Y = z["Y"].astype(np.float64)
        self.tets_np, self.m_np = z["tets"].astype(np.int64), z["m"]
        self.kappa, self.centres, self.radii = z["kappa"].astype(np.float64), z["centres"], z["radii"]
        self.kappa_e_true = z["kappa_e"].astype(np.float64)                      # evaluation / oracle only
        T = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt, device=DEV)
        self.tets = torch.tensor(self.tets_np, device=DEV)
        self.vol = T(z["vol"], F64)
        if basis == "arpack128":
            Phi = T(z["Phi"], F64)
        else:                                                                    # B-3 alternative basis
            split = os.path.basename(os.path.dirname(path))
            Phi = T(np.load(os.path.join(ROOT, "coef", "b3_basis", split, f"{self.name}_{basis}.npz"))["Phi"], F64)
        grads = T(z["grads"], F64)                                               # (T, 3, 4)
        self.Phi, self.grads = Phi, grads                                        # grad Y per tet built per call (memory)
        self.grads_t = grads.transpose(1, 2).contiguous()                       # (T, 4, 3)
        m = T(self.m_np, F64)
        self.msum = float(m.sum())
        U0, Uobs = T(z["U0"], F64), T(z["Uobs"], F64)
        self.c0 = Phi.T @ (m[:, None] * U0)                                      # (M, I)
        self.cobs = torch.einsum("nm,n,tni->tmi", Phi, m, Uobs)                 # (T, M, I)
        self.t = T(z["t"], F64)
        # network input
        sc = U0.max(0).values
        feats = [T(self.Y), (U0 / sc).float(), (Uobs / sc).permute(1, 0, 2).reshape(len(self.Y), -1).float()]
        self.x = torch.cat(feats, 1)                                             # (N, 3 + 8 + 48)
        # kNN (Euclidean, includes self) for local aggregation
        P = T(self.Y)
        self.knn = torch.cat([torch.cdist(P[i:i + 2048], P).topk(KNN, largest=False).indices
                              for i in range(0, len(P), 2048)])
        # H1 regulariser matrix (homogeneous stiffness) as a sparse tensor
        rows = self.tets[:, :, None].expand(-1, 4, 4).reshape(-1)
        cols = self.tets[:, None, :].expand(-1, 4, 4).reshape(-1)
        vals = (self.vol[:, None, None] * torch.einsum("tki,tkj->tij", grads, grads)).reshape(-1)
        n = len(self.Y)
        self.K1 = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n)).coalesce()

    def data_loss(self, theta):
        kap_e = torch.exp(theta.double())[self.tets].mean(1)
        GY = torch.einsum("tki,tim->tkm", self.grads, self.Phi[self.tets])      # (T, 3, M)
        A = torch.einsum("t,tkm,tkn->mn", kap_e * self.vol, GY, GY)
        A = 0.5 * (A + A.T)
        c = torch.linalg.matrix_exp(-self.t[:, None, None] * A[None]) @ self.c0[None]
        return ((c - self.cobs) ** 2).sum() / (self.cobs ** 2).sum()

    def reg(self, theta, regtype):
        th = theta.double()
        if regtype == "h1":
            return (th @ torch.sparse.mm(self.K1, th[:, None])[:, 0]) / self.msum
        g = torch.einsum("tki,tk->ti", self.grads_t, th[self.tets])
        return (self.vol * torch.sqrt((g ** 2).sum(1) + TV_EPS ** 2)).sum() / self.msum


def mlp(*d):
    L = []
    for a, b in zip(d[:-1], d[1:]):
        L += [nn.Linear(a, b), nn.GELU()]
    return nn.Sequential(*L[:-1])


class Block(nn.Module):
    def __init__(self, c, slices=32, heads=4):
        super().__init__()
        self.local = mlp(c + 3, c, c)
        self.n1, self.n2, self.n3 = nn.LayerNorm(c), nn.LayerNorm(c), nn.LayerNorm(c)
        self.slice = nn.Linear(c, slices)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.ff = mlp(c, 2 * c, c)

    def forward(self, h, x, knn):
        hn = self.n1(h)
        e = torch.cat([hn[knn] - hn[:, None], x[knn] - x[:, None]], -1)          # (N, k, c + 3)
        h = h + self.local(e).max(1).values
        hn = self.n2(h)
        w = torch.softmax(self.slice(hn), -1)                                   # (N, S)
        z = (w.T @ hn) / (w.sum(0)[:, None] + 1e-6)                             # (S, c)
        z = self.attn(z[None], z[None], z[None])[0][0]
        h = h + w @ z
        return h + self.ff(self.n3(h))


class CoefNet(nn.Module):
    def __init__(self, cin, c=128, layers=4):
        super().__init__()
        self.enc = mlp(cin, c, c)
        self.blocks = nn.ModuleList([Block(c) for _ in range(layers)])
        self.head = mlp(c, c, 1)
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)   # start at kappa_hat = 1

    def forward(self, s):
        h = self.enc(s.x)
        xyz = s.x[:, :3]
        for b in self.blocks:
            h = b(h, xyz, s.knn)
        return self.head(h)[:, 0]


def evaluate(s, theta, a):
    kap_hat = np.exp(theta.detach().double().cpu().numpy())
    m = s.m_np
    rel = float(np.sqrt((m * (kap_hat - s.kappa) ** 2).sum() / (m * s.kappa ** 2).sum()))
    op = SimpleNamespace(tets=s.tets_np, m=m)
    loc = B1.lump_localisation(s.Y, op, kap_hat, s.centres, s.radii, B1.robust_threshold(kap_hat, m))
    with torch.no_grad():
        dl = s.data_loss(theta).item()
    return dict(rel_l2=rel, data=dl, n_lumps=len(s.centres), **{k: v for k, v in loc.items() if k != "errors"},
                centre_err=float(np.nanmean(loc["errors"])) if np.isfinite(loc["errors"]).any() else np.nan,
                in_lump=float(np.average(kap_hat[s.kappa > 3], weights=m[s.kappa > 3])),
                background=float(np.average(kap_hat[s.kappa < 1.1], weights=m[s.kappa < 1.1]))), kap_hat


def refine(s, theta0, a, iters):
    """B-1 direct optimisation started from theta0 (Adam, lr 0.02, cosine)."""
    theta = theta0.detach().double().clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=0.02)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for _ in range(iters):
        loss = s.data_loss(theta) + a.reg * s.reg(theta, a.regtype)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return theta.detach()


def slice_fig(s, kap_hat, kap_ref, path, title):
    Y = s.Y
    h = 0.03
    nb = len(s.centres)
    fig, ax = plt.subplots(nb, 3, figsize=(12, 4 * nb), squeeze=False)
    vmax = max(5.0, float(np.percentile(kap_hat, 99.5)))
    for i, c in enumerate(s.centres):
        sel = np.abs(Y[:, 2] - c[2]) < h
        for j, (f, lab) in enumerate([(s.kappa, "true κ"), (kap_hat, "network κ̂"), (kap_ref, "network + refine κ̂")]):
            a = ax[i, j]
            sc = a.scatter(Y[sel, 0], Y[sel, 1], c=f[sel], s=8, vmin=1, vmax=vmax)
            for c2, r2 in zip(s.centres, s.radii):
                d = abs(c2[2] - c[2])
                if d < r2:
                    a.add_patch(plt.Circle(c2[:2], np.sqrt(r2 ** 2 - d ** 2), fill=False, color="r"))
            a.set_title(f"z = {c[2]:.2f} · {lab}"); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        fig.colorbar(sc, ax=ax[i].tolist(), shrink=0.8)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=90); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--regtype", choices=["h1", "tv"], default="h1")
    ap.add_argument("--refine-iters", type=int, default=300)
    ap.add_argument("--basis", default="arpack128", choices=["arpack128", "arpack256", "neo256"])
    a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "set CUDA_VISIBLE_DEVICES"
    torch.manual_seed(0)
    out = os.path.join(OUT, a.run)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    train = [Shape(p, a.basis) for p in sorted(glob.glob(os.path.join(DATA, "train", "*.npz")))]
    test = [Shape(p, a.basis) for p in sorted(glob.glob(os.path.join(DATA, "test", "*.npz")))]
    print(f"loaded {len(train)} train / {len(test)} test shapes in {time.time() - t0:.0f}s", flush=True)
    net = CoefNet(train[0].x.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)
    steps = a.epochs * len(train)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    hist, rng = [], np.random.default_rng(0)
    for ep in range(a.epochs):
        tot = 0.0
        net.train()
        for i in rng.permutation(len(train)):
            s = train[i]
            theta = net(s)
            dl = s.data_loss(theta)
            loss = dl + a.reg * s.reg(theta, a.regtype)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sch.step()
            tot += dl.item()
        if ep % 10 == 0 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                te = np.mean([s.data_loss(net(s)).item() for s in test])
            hist.append(dict(epoch=ep, train_data=tot / len(train), test_data=te, t=time.time() - t0))
            print(hist[-1], flush=True)
    torch.save(net.state_dict(), os.path.join(out, "net.pt"))

    rows = []
    net.eval()
    for s in test:
        torch.cuda.synchronize(); ti = time.time()
        with torch.no_grad():
            th = net(s)
        torch.cuda.synchronize(); t_inf = time.time() - ti
        r_net, k_net = evaluate(s, th, a)
        ti = time.time()
        th_ref = refine(s, th, a, a.refine_iters)
        r_ref, k_ref = evaluate(s, th_ref, a)
        t_ref = time.time() - ti
        with torch.no_grad():
            oracle = s.data_loss(torch.log(torch.tensor(s.kappa, device=DEV))).item()
        rows.append(dict(name=s.name, oracle_data=oracle, t_infer_s=t_inf, t_refine_s=t_ref,
                         **{f"net_{k}": v for k, v in r_net.items()}, **{f"ref_{k}": v for k, v in r_ref.items()}))
        slice_fig(s, k_net, k_ref, os.path.join(out, f"slices_{s.name}.png"),
                  f"B-2 {a.run} {s.name}: network relL2 {r_net['rel_l2']:.3f}, matched {r_net['n_matched']}/{len(s.centres)} "
                  f"(+{r_net['n_false']}) | +refine({a.refine_iters}) relL2 {r_ref['rel_l2']:.3f}, matched "
                  f"{r_ref['n_matched']}/{len(s.centres)} (+{r_ref['n_false']})")
        print(rows[-1], flush=True)
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(out, "test.csv"), index=False)
    json.dump(dict(args=vars(a), history=hist, mean=E.drop(columns="name").mean().to_dict(),
                   t_total_s=time.time() - t0), open(os.path.join(out, "result.json"), "w"), indent=2)
    print(E.drop(columns="name").mean().to_string())


if __name__ == "__main__":
    main()

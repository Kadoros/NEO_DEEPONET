"""Branch B-4: two-stage lump recovery (contrast fix) -> coef/b4/<tag>/

Stage 1  B-2 network (coef/b2/<net>/net.pt) -> kappa_hat -> connected components above the robust threshold
         (same detection as B-2's evaluation, no truth used).
Stage 2  every detected component becomes a parametric lump
             kappa(x) = 1 + sum_i (k_i - 1) * sigmoid((r_i - |x - c_i|) / w_i)
         with only (centre c_i, radius r_i, value k_i, edge width w_i) as unknowns (background fixed at 1, as in the
         data), fitted by the SAME projected Galerkin data term (no smoothness regulariser), Adam.
         Initialisation: c = component centroid, r = radius of a ball with the component's volume,
         k = mean kappa_hat on the component, w = 0.03.
Split    each fitted lump is also tried as two lumps (placed +-0.5 r along the principal axis of its component,
         radii 0.75 r); the split is kept when the data term drops below SPLIT_GAIN x the unsplit value.
Reports  per true lump: matched / centre error / radius error / fitted value; kappa_hat inside true lumps
         (mass-weighted, as B-1/B-2), relative L2, false lumps; stage 1 vs stage 2; splits accepted.

    CUDA_VISIBLE_DEVICES=<gpu> python stage2/coef_b4.py [--split test|test_ood] [--basis arpack128] [--net base]
        [--shard k --nshards n]   (shapes[k::n] -> <tag>_shard<k>/;   --merge n: combine shards into <tag>/)
"""
import argparse
import glob
import json
import os
import time

import coef_b1 as B1
import coef_b2 as C

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components

from volop import EDGES

OUT = os.path.join(C.ROOT, "coef", "b4")
SPLIT_GAIN = 0.8
ITERS = 800
W_MIN, R_MIN = 0.005, 0.03


def components(s, kap_hat):
    """Same detection as B-2 evaluation: components (>= 5 points) of {kappa_hat > robust threshold}."""
    thr = B1.robust_threshold(kap_hat, s.m_np)
    T = s.tets_np
    rows = np.concatenate([T[:, a] for a, b in EDGES]); cols = np.concatenate([T[:, b] for a, b in EDGES])
    hot = kap_hat > thr
    keep = hot[rows] & hot[cols]
    G = sp.coo_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(len(s.Y), len(s.Y)))
    _, lab = connected_components(G, directed=False)
    comps = [np.where((lab == k) & hot)[0] for k in np.unique(lab[hot])]
    return [c for c in comps if len(c) >= 5]


class Lumps(torch.nn.Module):
    def __init__(self, c, r, k, w):
        super().__init__()
        t = lambda a: torch.nn.Parameter(torch.tensor(np.asarray(a, float), dtype=C.F64, device=C.DEV))
        self.c, self.lr_, self.lk, self.lw = t(c), t(np.log(r)), t(np.log(np.maximum(np.asarray(k) - 1, 0.05))), t(np.log(w))

    def params(self):
        return (self.c.detach().cpu().numpy(), (torch.exp(self.lr_) + R_MIN).detach().cpu().numpy(),
                (1 + torch.exp(self.lk)).detach().cpu().numpy(), (torch.exp(self.lw) + W_MIN).detach().cpu().numpy())

    def kappa(self, x):
        r, w = torch.exp(self.lr_) + R_MIN, torch.exp(self.lw) + W_MIN
        d = torch.linalg.norm(x[:, None] - self.c[None], dim=2)                 # (n, L)
        return 1 + (torch.exp(self.lk)[None] * torch.sigmoid((r[None] - d) / w[None])).sum(1)


ASM32 = False                      # --asm32: assemble A_hat in fp32 (matrix exponential and loss stay fp64)


def data_loss_e(s, kap_e):
    if ASM32:
        if not hasattr(s, "GY32"):                                      # kappa-independent: build once per shape
            s.GY32 = torch.einsum("tki,tim->tkm", s.grads.float(), s.Phi.float()[s.tets]).contiguous()
            s.vol32 = s.vol.float()
        A = torch.einsum("t,tkm,tkn->mn", kap_e.float() * s.vol32, s.GY32, s.GY32).double()
    else:
        GY = torch.einsum("tki,tim->tkm", s.grads, s.Phi[s.tets])
        A = torch.einsum("t,tkm,tkn->mn", kap_e * s.vol, GY, GY)
    A = 0.5 * (A + A.T)
    c = torch.linalg.matrix_exp(-s.t[:, None, None] * A[None]) @ s.c0[None]
    return ((c - s.cobs) ** 2).sum() / (s.cobs ** 2).sum()


def fit(s, cen_t, init):
    model = Lumps(*init)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
    for _ in range(ITERS):
        loss = data_loss_e(s, model.kappa(cen_t))
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        loss = data_loss_e(s, model.kappa(cen_t)).item()
    return model, loss


def score(s, kap_pts, centres_fit, radii_fit, values_fit):
    m = s.m_np
    rel = float(np.sqrt((m * (kap_pts - s.kappa) ** 2).sum() / (m * s.kappa ** 2).sum()))
    inl = s.kappa > 3
    res = dict(rel_l2=rel, in_lump=float(np.average(kap_pts[inl], weights=m[inl])), n_fit=len(centres_fit))
    per = [dict(centre_err=np.nan, radius_err=np.nan, value=np.nan) for _ in s.centres]
    n_matched = 0
    if len(centres_fit):
        Dm = np.linalg.norm(s.centres[:, None] - np.asarray(centres_fit)[None], axis=2)
        for a, b in zip(*linear_sum_assignment(Dm)):
            if Dm[a, b] < s.radii[a]:
                n_matched += 1
                per[a] = dict(centre_err=float(Dm[a, b]), radius_err=float(radii_fit[b] - s.radii[a]), value=float(values_fit[b]))
    res.update(n_matched=n_matched, n_false=len(centres_fit) - n_matched, per_lump=per)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--basis", default="arpack128")
    ap.add_argument("--net", default="base")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", type=int, default=0, help="1: combine all <tag>_shard* dirs into the unsharded tag")
    ap.add_argument("--asm32", action="store_true", help="fp32 assembly of the reduced matrix (precision check)")
    ap.add_argument("--names", default="", help="comma-separated case names to run (default: all in the split)")
    a = ap.parse_args()
    tag = f"{a.split}_{a.basis}_net-{a.net}" + ("_asm32" if a.asm32 else "") + ("_sel" if a.names else "")
    global ASM32
    ASM32 = a.asm32
    if a.merge:
        dirs = sorted(glob.glob(os.path.join(OUT, f"{tag}_shard*")))                # shards may use different n
        E = pd.concat([pd.read_csv(os.path.join(d, "results.csv")) for d in dirs]).sort_values("name")
        out = os.path.join(OUT, tag)
        os.makedirs(out, exist_ok=True)
        E.to_csv(os.path.join(out, "results.csv"), index=False)
        summarize(E, tag, out, sum(json.load(open(os.path.join(d, "summary.json")))["t_total_s"] for d in dirs))
        return
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "set CUDA_VISIBLE_DEVICES"
    out = os.path.join(OUT, tag + (f"_shard{a.shard}" if a.nshards > 1 else ""))
    os.makedirs(out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(C.DATA, a.split, "*.npz")))
    if a.names:
        paths = [p for p in paths if os.path.basename(p)[:-4] in a.names.split(",")]
    shapes = [C.Shape(p, a.basis) for p in paths][a.shard::a.nshards]
    net = C.CoefNet(shapes[0].x.shape[1]).to(C.DEV)
    net.load_state_dict(torch.load(os.path.join(C.OUT, a.net, "net.pt"), map_location=C.DEV))
    net.eval()
    rows, t0 = [], time.time()
    parts = os.path.join(out, "parts")                                    # one csv per finished case -> resumable
    os.makedirs(parts, exist_ok=True)
    for s in shapes:
        part = os.path.join(parts, f"{s.name}.csv")
        if os.path.exists(part):                                           # finished in an earlier (interrupted) run
            rows += pd.read_csv(part).to_dict("records")
            continue
        ts = time.time()
        with torch.no_grad():
            kap1 = np.exp(net(s).double().cpu().numpy())
        comps = components(s, kap1)
        cen = s.Y[s.tets_np].mean(1)
        cen_t = torch.tensor(cen, dtype=C.F64, device=C.DEV)
        Y_t = torch.tensor(s.Y, dtype=C.F64, device=C.DEV)
        m = s.m_np
        # stage-1 score (network field, component centroids as lump centres)
        c1 = [(s.Y[c] * m[c, None]).sum(0) / m[c].sum() for c in comps]
        st1 = score(s, kap1, c1, [np.nan] * len(c1), [np.nan] * len(c1))
        with torch.no_grad():
            oracle = data_loss_e(s, torch.tensor(s.kappa_e_true, dtype=C.F64, device=C.DEV)).item()
        if not comps:
            rows.append(dict(name=s.name, n_true=len(s.centres), n_comp=0, oracle=oracle,
                             **{f"s1_{k}": v for k, v in st1.items() if k != "per_lump"}))
            pd.DataFrame([rows[-1]]).to_csv(part, index=False)
            continue
        init = (np.array(c1), np.array([(3 * m[c].sum() / (4 * np.pi)) ** (1 / 3) for c in comps]),
                np.array([max(float(np.average(kap1[c], weights=m[c])), 1.3) for c in comps]), np.full(len(comps), 0.03))
        model, loss = fit(s, cen_t, init)
        loss0 = loss
        # split attempts, one per fitted lump (greedy, in order)
        n_split = 0
        axes = []
        for c in comps:
            P = s.Y[c] - s.Y[c].mean(0)
            axes.append(np.linalg.eigh(P.T @ (P * m[c, None]))[1][:, -1])
        i = 0
        while i < len(model.c):
            cc, rr, kk, ww = model.params()
            ax = axes[i] if i < len(axes) else np.eye(3)[0]
            new = (np.vstack([np.delete(cc, i, 0), cc[i] + 0.5 * rr[i] * ax, cc[i] - 0.5 * rr[i] * ax]),
                   np.concatenate([np.delete(rr, i), [0.75 * rr[i]] * 2]), np.concatenate([np.delete(kk, i), [kk[i]] * 2]),
                   np.concatenate([np.delete(ww, i), [ww[i]] * 2]))
            m2, l2 = fit(s, cen_t, new)
            if l2 < SPLIT_GAIN * loss:
                model, loss, n_split = m2, l2, n_split + 1
                axes = [a2 for j, a2 in enumerate(axes) if j != i] + [ax, ax]
                continue                                   # the lump at index i is new: examine it too
            i += 1
        with torch.no_grad():
            kap2 = model.kappa(Y_t).cpu().numpy()
        cc, rr, kk, ww = model.params()
        st2 = score(s, kap2, cc, rr, kk)
        row = dict(name=s.name, n_true=len(s.centres), n_comp=len(comps), n_split=n_split, oracle=oracle,
                   loss_fit=loss0, loss_final=loss, t_s=time.time() - ts,
                   **{f"s1_{k}": v for k, v in st1.items() if k != "per_lump"},
                   **{f"s2_{k}": v for k, v in st2.items() if k != "per_lump"},
                   values=json.dumps([round(float(v), 3) for v in kk]), widths=json.dumps([round(float(v), 4) for v in ww]),
                   per_lump=json.dumps(st2["per_lump"]))
        rows.append(row)
        pd.DataFrame([row]).to_csv(part, index=False)
        C.slice_fig(s, kap1, kap2, os.path.join(out, f"slices_{s.name}.png"),
                    f"B-4 {tag} {s.name}: stage1 (network) matched {st1['n_matched']}/{len(s.centres)} in-lump {st1['in_lump']:.2f} | "
                    f"stage2 (lumps) matched {st2['n_matched']}/{len(s.centres)} (+{st2['n_false']}), in-lump {st2['in_lump']:.2f}, "
                    f"values {[round(float(v), 2) for v in kk]}, splits {n_split}")
        print({k: row[k] for k in ("name", "n_true", "n_comp", "n_split", "s1_n_matched", "s2_n_matched", "s2_n_false",
                                    "s1_in_lump", "s2_in_lump", "s2_rel_l2", "values", "loss_final", "oracle")}, flush=True)
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(out, "results.csv"), index=False)
    summarize(E, tag, out, time.time() - t0)


def summarize(E, tag, out, t_total):
    vals = [p["value"] for pl in E.per_lump if isinstance(pl, str) for p in json.loads(pl) if p["value"] == p["value"]]
    summ = dict(tag=tag, shapes=len(E), lumps=int(E.n_true.sum()),
                s1_matched=int(E.s1_n_matched.sum()), s1_false=int(E.s1_n_false.sum()),
                s2_matched=int(E.s2_n_matched.sum()), s2_false=int(E.s2_n_false.sum()), splits=int(E.n_split.sum()),
                s1_in_lump=float(E.s1_in_lump.mean()), s2_in_lump=float(E.s2_in_lump.mean()),
                s1_rel_l2=float(E.s1_rel_l2.mean()), s2_rel_l2=float(E.s2_rel_l2.mean()),
                matched_value_mean=float(np.mean(vals)) if vals else np.nan,
                matched_value_median=float(np.median(vals)) if vals else np.nan, t_total_s=t_total)
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=1))

if __name__ == "__main__":
    main()

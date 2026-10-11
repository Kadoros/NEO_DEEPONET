"""Summarise NEO infer.py results.json files -> results/summary.csv, results/summary.md.

    conda run -n neo python neo_check/summarize.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse.linalg as sla
from robust_laplacian import point_cloud_laplacian

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

rows = []
for f in sorted(RES.glob("*_k*_*.json")):
    ckpt, k, shape = f.stem.split("_")
    r = json.loads(f.read_text())
    for prec, p in r["precisions"].items():
        e = np.asarray(p["scores"]["eval_relerr"])[1:]          # index 0 (λ=0) is set to 0 by infer.py
        s = np.asarray(p["scores"]["pointcloud_vs_network"])
        t = p["times"]
        rows.append(dict(ckpt=ckpt, k=int(k[1:]), shape=shape, precision=prec, n_points=r["n_points"],
                         span_loss=p["loss"],
                         eval_relerr_mean=e.mean(), eval_relerr_median=np.median(e),
                         eval_relerr_max=e.max(), eval_relerr_mean_1to10=e[:10].mean(),
                         cos_pc_net_mean=s.mean(), cos_pc_net_median=np.median(s),
                         t_neo_ms=1e3 * (t["forward"] + t["qr"] + t["network_gev"]),
                         t_arpack_ms=1e3 * t["pointcloud_gev"]))
df = pd.DataFrame(rows).sort_values(["ckpt", "k", "precision", "shape"])
df.to_csv(RES / "summary.csv", index=False)

# sphere: ARPACK point-cloud eigenvalues vs analytic l(l+1)/R^2
pts = np.load(HERE / "samples/sphere/input/points.npy").astype(np.float64)
R = np.linalg.norm(pts - pts.mean(0), axis=1).mean()
L, M = point_cloud_laplacian(pts)
lam = np.sort(sla.eigsh(L, k=96, M=M, sigma=-1e-8)[0])
ex = np.sort(np.concatenate([[l * (l + 1)] * (2 * l + 1) for l in range(12)]))[:96] / R ** 2
bias = lam[1:] / ex[1:] - 1

cols = ["ckpt", "k", "precision", "shape", "span_loss", "eval_relerr_mean", "eval_relerr_median",
        "eval_relerr_max", "eval_relerr_mean_1to10", "cos_pc_net_median", "t_neo_ms", "t_arpack_ms"]
with open(RES / "summary.md", "w") as fh:
    fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
    for _, r in df[cols].iterrows():
        fh.write("| " + " | ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in r) + " |\n")
    fh.write(f"\nsphere (N=2048, R={R:.4f}): ARPACK point-cloud λ vs l(l+1)/R², i=1..95: "
             f"mean {bias.mean():+.4f}, i=1..3 {bias[:3].mean():+.4f}, i=80..95 {bias[79:].mean():+.4f}\n")
print((RES / "summary.md").read_text())

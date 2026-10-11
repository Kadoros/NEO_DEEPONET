"""Tables + figures for the ε-path v2 experiment (results_eps/).

    python stage1/report_eps.py      (env neo, CPU)
"""
import common  # noqa: F401

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter

from common import DATA, FAMILIES, RESULTS

OUT = os.path.join(os.path.dirname(RESULTS), "results_eps")
ORDER = ["PC-discrete", "Oracle-mesh96", "G-neo", "(a) L-neo", "(b) L-neo+rp", "(c) L-neo+rp+eps",
         "(d) L-neo+eps(r)", "(d') L-neo+eps(r,rp-in)", "DON-base (converged)"]
STYLE = {
    "PC-discrete": dict(color="#0b0b0b", ls="-", lw=1.2, marker="s"),
    "Oracle-mesh96": dict(color="#8a8984", ls=(0, (1, 1)), lw=1.2, marker="."),
    "G-neo": dict(color="#eb6834", ls=":", lw=1.5, marker="o"),
    "(a) L-neo": dict(color="#eb6834", ls="--", lw=2, marker="o"),
    "(b) L-neo+rp": dict(color="#2a78d6", ls="--", lw=2, marker="o"),
    "(c) L-neo+rp+eps": dict(color="#2a78d6", ls="-", lw=2.2, marker="o"),
    "(d) L-neo+eps(r)": dict(color="#1baf7a", ls="-", lw=2, marker="o"),
    "(d') L-neo+eps(r,rp-in)": dict(color="#4a3aa7", ls="-", lw=2, marker="o"),
    "DON-base (converged)": dict(color="#008300", ls="-", lw=1.5, marker="o"),
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style_ax(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def md_table(df, fmt="{:.4f}"):
    df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * df.shape[1]]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    E = pd.read_csv(os.path.join(OUT, "errors_test.csv"))
    meta = json.load(open(os.path.join(OUT, "models.json")))
    eps_idx = pd.read_csv(os.path.join(DATA, "eps_index.csv")).set_index("sid")
    split = pd.read_csv(os.path.join(DATA, "split.csv"))
    test_ids = split[split.split == "test"].sid
    models = [m for m in ORDER if m in E.model.unique()]

    mean = E.pivot_table(index="model", columns="tau", values="err", aggfunc="mean").loc[models]
    mean.columns = [f"τ={c:.3g}" for c in mean.columns]
    small = [c for c in mean.columns if float(c[2:]) <= 0.031]
    mean["mean τ≤0.03"] = mean[small].mean(1)
    mean["mean all"] = E.groupby("model").err.mean().loc[models]
    ratio = mean.div(mean.loc["PC-discrete"], axis=1)
    med = E.pivot_table(index="model", columns="tau", values="err", aggfunc="median").loc[models]
    med.columns = [f"τ={c:.3g}" for c in med.columns]
    p90 = E.pivot_table(index="model", columns="tau", values="err", aggfunc=lambda v: np.percentile(v, 90)).loc[models]
    p90.columns = [f"τ={c:.3g}" for c in p90.columns]
    fam = E[E.tau <= 0.031].pivot_table(index="model", columns="family", values="err", aggfunc="mean").loc[models, FAMILIES]
    cost = pd.DataFrame(meta).T.loc[[m for m in models if m in meta], ["params", "train_s", "infer_ms"]].astype(float)
    pre = pd.DataFrame({"r_p (8 sparse LU solves) [s]": [eps_idx.loc[test_ids, "t_rp_s"].mean()],
                        "PC-discrete expm_multiply (8 τ) [s]": [eps_idx.loc[test_ids, "t_pcsol_s"].mean()],
                        "‖r‖_M/‖u0‖_M (mean)": [eps_idx.loc[test_ids, "r_rel"].mean()]},
                       index=["mean per test shape"])

    with open(os.path.join(OUT, "tables.md"), "w") as fh:
        fh.write("## Mean M-weighted relative L2 error (test: 60 shapes × 4 ICs)\n\n" + md_table(mean) + "\n")
        fh.write("\n## Ratio to PC-discrete (full point-cloud operator, exact in time)\n\n"
                 + md_table(ratio, "{:.3g}") + "\n")
        fh.write("\n## Median\n\n" + md_table(med) + "\n")
        fh.write("\n## 90th percentile\n\n" + md_table(p90) + "\n")
        fh.write("\n## Mean over τ ≤ 0.03, by family\n\n" + md_table(fam) + "\n")
        fh.write("\n## Cost (inference: one shape, 4 ICs × 8 τ, GPU; r_p / basis precomputed)\n\n"
                 + md_table(cost, "{:.4g}") + "\n")
        fh.write("\n## Precomputation per test shape (CPU)\n\n" + md_table(pre, "{:.4g}") + "\n")
    print(open(os.path.join(OUT, "tables.md")).read())

    # figure 1: all τ, log-log, all + per family
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), sharey=True, constrained_layout=True)
    for ax, (title, sub) in zip(axes, [("all test shapes", E)] + [(f, E[E.family == f]) for f in FAMILIES]):
        style_ax(ax)
        for m in models:
            a = sub[sub.model == m].groupby("tau").err.mean()
            ax.plot(a.index, a.values, ms=3.5, label=m, **STYLE[m])
        ax.axvspan(8e-4, 0.035, color="#cde2fb", alpha=0.35, lw=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xlabel("τ = t·λ₁", color=INK2)
    axes[0].set_ylabel("‖û − u‖_M / ‖u‖_M  (test mean)", color=INK2)
    axes[-1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("ε-path v2 — test error vs τ (shaded: τ ≤ 0.03)", color=INK, fontsize=11)
    fig.savefig(os.path.join(OUT, "fig_eps_error_vs_tau.png"), dpi=150)
    plt.close(fig)

    # figure 2: τ ≤ 0.03 zoom — per-sample distribution (box) per model and τ
    taus = sorted(E.tau.unique())[:4]
    show = [m for m in models if m not in ("DON-base (converged)",)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), sharey=True, constrained_layout=True)
    for ax, tau in zip(axes, taus):
        style_ax(ax)
        data = [E[(E.model == m) & (E.tau == tau)].err.values for m in show]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, m in zip(bp["boxes"], show):
            patch.set_facecolor(STYLE[m]["color"]); patch.set_alpha(0.55); patch.set_edgecolor(STYLE[m]["color"])
        for k in ("medians",):
            for line in bp[k]:
                line.set_color(INK)
        ax.set_yscale("log")
        ax.set_xticks(range(1, len(show) + 1))
        ax.set_xticklabels([m.split(" ")[0] if m.startswith("(") else m for m in show], rotation=45, fontsize=8)
        ax.set_title(f"τ = {tau:.3g}", color=INK, fontsize=10)
    axes[0].set_ylabel("per-sample relative error (test, 240 fields)", color=INK2)
    fig.suptitle("τ ≤ 0.03: per-sample error distribution (box = IQR, whiskers = 1.5·IQR)", color=INK, fontsize=11)
    fig.savefig(os.path.join(OUT, "fig_eps_small_tau.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

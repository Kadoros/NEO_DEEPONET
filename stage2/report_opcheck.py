"""Tables, pass/fail verdicts and figures for the volume-operator validation -> results_opcheck/

    python stage2/report_opcheck.py      (env neo)
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["MPLCONFIGDIR"] = os.path.join(ROOT, ".cache", "mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter

OUT = os.path.join(ROOT, "results_opcheck")
OPS = ["delaunay", "delaunay_nocarve", "knn16", "knn32"]
COL = {"delaunay": "#2a78d6", "delaunay_nocarve": "#eb6834", "knn16": "#1baf7a", "knn32": "#4a3aa7"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def md(df, fmt="{:.4g}"):
    df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * df.shape[1]]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    frames = [pd.read_csv(os.path.join(OUT, f)) for f in ("opcheck.csv", "opcheck_cup.csv")
              if os.path.exists(os.path.join(OUT, f))]
    d = pd.concat(frames)
    d["abs_low_bias"] = d.low_bias_cal.abs()
    g = d.groupby(["domain", "op", "sampling", "N"]).agg(
        mass_ratio=("mass_ratio", "mean"), low_bias=("low_bias_cal", "mean"), abs_low_bias=("abs_low_bias", "mean"),
        mean_abs_err=("abs_err_cal_1toK", "mean"), span_loss=("span_loss", "mean"),
        sliver_frac=("sliver_frac", "mean"), carved_frac=("carved_frac", "mean"), t_s=("t_s", "mean")).reset_index()
    g.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    # verdicts: (1) N = 8k quasi: |low bias| <= 1 %, mass in [0.98, 1.02], span <= 1e-3 (analytic domains)
    #           (2) random: |low bias|, mean |eig err| and span decrease monotonically with N
    rows = []
    for (dom, op), s in g.groupby(["domain", "op"]):
        q8 = s[(s.sampling == "quasi") & (s.N == 8000)].iloc[0]
        r = s[s.sampling == "random"].sort_values("N")
        mono = lambda col: bool(np.all(np.diff(r[col].values) < 0)) if r[col].notna().all() else None
        c1 = (abs(q8.low_bias) <= 0.01) and (0.98 <= q8.mass_ratio <= 1.02) and (
            np.isnan(q8.span_loss) or q8.span_loss <= 1e-3)
        rows.append(dict(domain=dom, op=op, q8_mass=q8.mass_ratio, q8_low_bias=q8.low_bias,
                         q8_span=q8.span_loss, crit1_N8k=c1, rnd_mono_bias=mono("abs_low_bias"),
                         rnd_mono_eigerr=mono("mean_abs_err"), rnd_mono_span=mono("span_loss"),
                         rnd32k_low_bias=r.low_bias.values[-1], rnd32k_span=r.span_loss.values[-1]))
    V = pd.DataFrame(rows).set_index(["domain", "op"])
    V.to_csv(os.path.join(OUT, "verdicts.csv"))

    with open(os.path.join(OUT, "tables.md"), "w") as fh:
        fh.write("## Verdicts (criterion 1: N=8k quasi-uniform; criterion 2: monotone decrease with N, random)\n\n")
        fh.write(md(V) + "\n\n")
        for dom in g.domain.unique():
            t = g[g.domain == dom].set_index(["op", "sampling", "N"])[
                ["mass_ratio", "low_bias", "mean_abs_err", "span_loss", "sliver_frac", "carved_frac", "t_s"]]
            fh.write(f"## {dom}\n\n" + md(t) + "\n\n")
    print(open(os.path.join(OUT, "tables.md")).read()[:6000])

    # figures: per domain, low-mode bias / mean eigen error / span loss vs N
    doms = list(g.domain.unique())
    fig, axes = plt.subplots(3, len(doms), figsize=(4.2 * len(doms), 10), constrained_layout=True, squeeze=False)
    for j, dom in enumerate(doms):
        for i, (col, lab) in enumerate([("abs_low_bias", "|low-mode bias| (i=1..10)"),
                                        ("mean_abs_err", "mean |eig. rel. err| (i < K_ref)"),
                                        ("span_loss", "span loss (first K_ref modes)")]):
            ax = axes[i, j]
            style(ax)
            for op in OPS:
                for smp, ls in (("quasi", "-"), ("random", "--")):
                    s = g[(g.domain == dom) & (g.op == op) & (g.sampling == smp)].sort_values("N")
                    if len(s) and s[col].notna().any():
                        ax.plot(s.N, s[col].clip(lower=1e-6), ls, marker="o", ms=3, color=COL[op], lw=1.6,
                                label=f"{op}, {smp}")
            if col == "span_loss":
                ax.axhline(1e-3, color=INK2, lw=0.8, ls=":")
            if col == "abs_low_bias":
                ax.axhline(1e-2, color=INK2, lw=0.8, ls=":")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xticks([2000, 4000, 8000, 16000, 32000]); ax.set_xticklabels(["2k", "4k", "8k", "16k", "32k"])
            if i == 0:
                ax.set_title(dom, color=INK, fontsize=11)
            if j == 0:
                ax.set_ylabel(lab, color=INK2, fontsize=9)
            if i == 2:
                ax.set_xlabel("N (points)", color=INK2)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Volume operators vs N — solid: quasi-uniform, dashed: random; dotted: pass thresholds "
                 "(1 % bias, span 1e-3)", color=INK, fontsize=11)
    fig.savefig(os.path.join(OUT, "fig_opcheck.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Part 3: tables + figures from results/ (run after train.py).

    python stage1/report.py      (env neo, CPU)
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

from common import DATA, FAMILIES, RESULTS, TAUS

ORDER = ["Oracle-mesh96", "G-exact", "L-exact", "L-exact+eps", "G-exact-uncal", "L-exact-uncal",
         "G-neo", "L-neo", "L-neo+eps", "DON-base"]
STYLE = {  # colour = basis, line style = model kind
    "Oracle-mesh96": dict(color="#8a8984", ls=(0, (1, 1)), lw=1.2),
    "G-exact": dict(color="#2a78d6", ls=":", lw=2), "L-exact": dict(color="#2a78d6", ls="--", lw=2),
    "L-exact+eps": dict(color="#2a78d6", ls="-", lw=2),
    "G-neo": dict(color="#eb6834", ls=":", lw=2), "L-neo": dict(color="#eb6834", ls="--", lw=2),
    "L-neo+eps": dict(color="#eb6834", ls="-", lw=2),
    "G-exact-uncal": dict(color="#4a3aa7", ls=":", lw=2), "L-exact-uncal": dict(color="#4a3aa7", ls="--", lw=2),
    "DON-base": dict(color="#008300", ls="-", lw=2),
}
T_COLORS = ["#b7d3f6", "#9ec5f4", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
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
    E = pd.read_csv(os.path.join(RESULTS, "errors_test.csv"))
    meta = json.load(open(os.path.join(RESULTS, "models.json")))
    index = pd.read_csv(os.path.join(DATA, "index.csv"))
    neo = pd.read_csv(os.path.join(DATA, "neo_index.csv"))
    split = pd.read_csv(os.path.join(DATA, "split.csv"))
    models = [m for m in ORDER if m in E.model.unique()]

    # ---- tables
    tab = E.pivot_table(index="model", columns="tau", values="err", aggfunc="mean").loc[models]
    tab.columns = [f"τ={c:.3g}" for c in tab.columns]
    tab["mean"] = E.groupby("model").err.mean().loc[models]
    ratio = tab.div(tab.loc["Oracle-mesh96"], axis=1)                      # model error / Oracle error
    fam = E.pivot_table(index="model", columns="family", values="err", aggfunc="mean").loc[models, FAMILIES]
    med = E.pivot_table(index="model", columns="tau", values="err", aggfunc="median").loc[models]
    med.columns = [f"τ={c:.3g}" for c in med.columns]
    cost = pd.DataFrame(meta).T.loc[models, ["params", "train_s", "infer_ms"]]
    test_ids = split[split.split == "test"].sid
    pre = pd.DataFrame({
        "pc Laplacian + ARPACK 96 [s]": [index.set_index("sid").loc[test_ids, "t_pc_basis_s"].mean()],
        "NEO input mass [s]": [neo.set_index("sid").loc[test_ids, "t_input_s"].mean()],
        "NEO forward [ms] (median)": [1e3 * neo.set_index("sid").loc[test_ids, "t_forward_s"].median()],
        "NEO Rayleigh–Ritz (incl. L build) [s]": [neo.set_index("sid").loc[test_ids, "t_rr_s"].mean()],
    }, index=["mean per test shape"])
    basis_q = pd.DataFrame({
        "pc mass / area": index.pc_mass_over_area.describe()[["mean", "min", "max"]],
        "pc λ bias i=1..10 (vs mesh)": index.pc_lam_bias_1to10.describe()[["mean", "min", "max"]],
        "pc λ bias i=80..95 (vs mesh)": index.pc_lam_bias_80to95.describe()[["mean", "min", "max"]],
        "NEO span loss vs pc (k=96)": neo.span_loss_vs_ex.describe()[["mean", "min", "max"]],
        "NEO λ rel.err vs pc i=1..95": neo.lam_relerr_1to95.describe()[["mean", "min", "max"]],
        "NEO λ rel.err vs mesh i=1..95": neo.lam_relerr_vs_mesh_1to95.describe()[["mean", "min", "max"]],
    }).T

    with open(os.path.join(RESULTS, "tables.md"), "w") as fh:
        fh.write("## Test error (mean M-weighted relative L2, 60 shapes × 4 ICs) by τ\n\n" + md_table(tab) + "\n")
        fh.write("\n## Ratio to Oracle (model mean error / Oracle-mesh96 mean error) by τ\n\n"
                 + md_table(ratio, "{:.3g}") + "\n")
        fh.write("\n## Median by τ\n\n" + md_table(med) + "\n")
        fh.write("\n## Mean by family\n\n" + md_table(fam) + "\n")
        fh.write("\n## Cost (inference = one shape, 4 ICs × 8 τ, GPU, given the precomputed basis)\n\n"
                 + md_table(cost.astype(float), "{:.4g}") + "\n")
        fh.write("\n## Basis precomputation (test shapes)\n\n" + md_table(pre, "{:.4g}") + "\n")
        fh.write("\n## Basis quality (all 600 shapes)\n\n" + md_table(basis_q, "{:+.4g}") + "\n")
    print(open(os.path.join(RESULTS, "tables.md")).read())

    # ---- figure: error vs τ (all + per family)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True, constrained_layout=True)
    for ax, (title, sub) in zip(axes, [("all test shapes", E)] + [(f, E[E.family == f]) for f in FAMILIES]):
        style_ax(ax)
        for m in models:
            a = sub[sub.model == m].groupby("tau").err.mean()
            ax.plot(a.index, a.values, marker="o", ms=3.5, label=m, **STYLE[m])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xlabel("τ = t·λ₁", color=INK2)
    axes[0].set_ylabel("‖û − u‖_M / ‖u‖_M  (test mean)", color=INK2)
    axes[-1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("Test error vs τ — colour: basis (blue exact / violet exact-uncalibrated / orange NEO / green DON), "
                 "line: G dotted, L dashed, L+ε solid", color=INK, fontsize=11)
    fig.savefig(os.path.join(RESULTS, "fig_error_vs_tau.png"), dpi=150)
    plt.close(fig)

    # ---- figure: learned g_θ vs exp(-λt)
    P = np.load(os.path.join(RESULTS, "g_theta_probe.npz"))
    fam_of = index.set_index("sid").family
    for suffix in ("", "+eps"):
        fig, axes = plt.subplots(2, 6, figsize=(22, 7.5), constrained_layout=True)
        for col, (basis_name, name) in enumerate([("exact", f"L-exact{suffix}"), ("NEO", f"L-neo{suffix}")]):
            g, lam, t = P[f"{name}|g"], P[f"{name}|lam"], P[f"{name}|t"]
            for j, sid in enumerate(P["probe_sids"]):
                ax_g, ax_r = axes[0, 3 * col + j], axes[1, 3 * col + j]
                style_ax(ax_g); style_ax(ax_r)
                s = lam[j][None, :] * t[j][:, None]                                    # (T,K)
                ss = np.geomspace(max(s[:, 1:].min(), 1e-5), s.max(), 200)
                ax_g.plot(ss, np.exp(-ss), color=INK, lw=1.2, label="exp(−λt)")
                for ti in range(len(TAUS)):
                    ax_g.scatter(s[ti, 1:], g[j, ti, 1:], s=6, color=T_COLORS[ti], label=f"τ={TAUS[ti]:.2g}")
                ax_g.set_xscale("log"); ax_g.set_ylim(-0.02, 1.02)
                ax_g.set_title(f"{name} · {fam_of[sid]} (sid {sid})", color=INK, fontsize=9)
                ax_g.set_xlabel("s = λ_k t", color=INK2)
                k = np.arange(1, lam.shape[1])
                with np.errstate(divide="ignore", invalid="ignore"):
                    rho = -np.log(np.clip(g[j, :, 1:], 1e-30, None)) / s[:, 1:]
                for ti in range(len(TAUS)):
                    keep = s[ti, 1:] < 30                                            # g < e^-30 carries no signal
                    ax_r.plot(k[keep], rho[ti, keep], color=T_COLORS[ti], lw=1.2)
                ax_r.plot(k, P["lam_mesh"][j, 1:] / lam[j, 1:], color=INK, ls="--", lw=1.2,
                          label="λ_mesh,k / λ_basis,k")
                ax_r.axhline(1, color=INK2, lw=0.8)
                ax_r.set_xlabel("mode k", color=INK2)
                ax_r.set_ylabel("effective rate ρ = −log g / (λ_k t)", color=INK2, fontsize=8)
        axes[0, 0].legend(fontsize=7, frameon=False, ncol=2)
        axes[1, 0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"Learned multiplier g_θ{' (models with ε)' if suffix else ''}: top — g vs s with exp(−s); "
                     "bottom — effective decay rate per mode (1 = plain Galerkin, dashed = mesh/basis eigenvalue "
                     "ratio)", color=INK, fontsize=11)
        fig.savefig(os.path.join(RESULTS, f"fig_g_theta{suffix.replace('+', '_')}.png"), dpi=130)
        plt.close(fig)

    # ---- figure: uncalibrated basis — does g_θ learn the missing factor c?
    name = "L-exact-uncal"
    g, lam, t, c = P[f"{name}|g"], P[f"{name}|lam"], P[f"{name}|t"], P["c_pc"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for j, sid in enumerate(P["probe_sids"]):
        ax_g, ax_r = axes[0, j], axes[1, j]
        style_ax(ax_g); style_ax(ax_r)
        s = lam[j][None, :] * t[j][:, None]
        ss = np.geomspace(max(s[:, 1:].min(), 1e-5), s.max(), 200)
        ax_g.plot(ss, np.exp(-ss), color=INK2, lw=1.2, ls=":", label="exp(−λt)  (λ: uncalibrated)")
        ax_g.plot(ss, np.exp(-c[j] * ss), color=INK, lw=1.2, label=f"exp(−cλt), c={c[j]:.4f}")
        for ti in range(len(TAUS)):
            ax_g.scatter(s[ti, 1:], g[j, ti, 1:], s=6, color=T_COLORS[ti], label=f"τ={TAUS[ti]:.2g}")
        ax_g.set_xscale("log"); ax_g.set_ylim(-0.02, 1.02)
        ax_g.set_title(f"{name} · {fam_of[sid]} (sid {sid})", color=INK, fontsize=9)
        ax_g.set_xlabel("s = λ_k t", color=INK2)
        k = np.arange(1, lam.shape[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = -np.log(np.clip(g[j, :, 1:], 1e-30, None)) / s[:, 1:]
        for ti in range(len(TAUS)):
            keep = s[ti, 1:] < 30
            ax_r.plot(k[keep], rho[ti, keep], color=T_COLORS[ti], lw=1.2)
        ax_r.axhline(1, color=INK2, lw=0.8, ls=":", label="1 (plain Galerkin, uncalibrated)")
        ax_r.axhline(c[j], color=INK, lw=1.2, label=f"c = {c[j]:.4f}")
        ax_r.plot(k, P["lam_mesh"][j, 1:] / lam[j, 1:], color=INK, ls="--", lw=1.0, label="λ_mesh,k / λ_uncal,k")
        ax_r.set_xlabel("mode k", color=INK2)
        ax_r.set_ylabel("effective rate ρ = −log g / (λ_k t)", color=INK2, fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=2)
    axes[1, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("L-exact-uncal: learned g_θ vs exp(−λt) and exp(−cλt) (λ = eigenvalues of the uncalibrated "
                 "operator L)", color=INK, fontsize=11)
    fig.savefig(os.path.join(RESULTS, "fig_g_theta_uncal.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()

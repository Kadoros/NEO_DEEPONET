"""A4: error vs shape indicator (Korn constant, thickness) -> stage3/e1/a4_<tag>.png

Panel 1  (points = shapes; filled nu = 0.3, open nu = 0.49) relative L2 error of the homogeneous static solution (smooth load)
         vs Korn constant C_K, for the
         replicated Laplacian basis (neo) and (a) neo + rotation fields.
Panel 2  model error / lump signal (interior, worst depth) vs C_K for (b), (c) and the exact-mode reference;
         the 0.2 pass line.
Panel 3  same as panel 1 against the thickness index d_max / V^(1/3) (thinner = smaller).

    python stage3/a4_plot.py --tag quick
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]                  # reference categorical slots 1-4 (validated)
MARKERS = ["o", "s", "^", "D"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax, xlabel, ylabel, logx=True, logy=True):
    ax.set_xscale("log" if logx else "linear"); ax.set_yscale("log" if logy else "linear")
    ax.set_xlabel(xlabel, color=INK2); ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2)


def series(ax, d, x, y, bases, labels):
    for k, (b, lab) in enumerate(zip(bases, labels)):
        for nu, ls in ((0.3, "-"), (0.49, "--")):
            s = d[(d.basis == b) & (d.nu == nu)].sort_values(x)
            if s.empty:
                continue
            ax.plot(s[x], s[y], ls="none", color=SERIES[k], marker=MARKERS[k], ms=9, mew=1.8,
                    mfc=SERIES[k] if nu == 0.3 else "none", markeredgecolor=SERIES[k] if nu != 0.3 else "white",
                    label=f"{lab}, ν={nu}")                             # discrete shapes: points only, no lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="quick")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(ROOT, "e1", f"e1_v2_{a.tag}.csv"))
    hom = df[df.load == "smooth"].groupby(["family", "idx", "nu", "basis"], as_index=False).agg(
        relL2_hom=("relL2_hom", "first"), korn=("korn", "first"), thick=("thick", "first"))
    worst = df.groupby(["family", "idx", "nu", "basis"], as_index=False).agg(
        ratio=("ratio_int", "max"), ratio_s=("ratio_surf", "max"), korn=("korn", "first"))
    fig, ax = plt.subplots(1, 3, figsize=(17, 6.6))
    series(ax[0], hom, "korn", "relL2_hom", ["neo", "neo_rot"], ["Laplacian ×3 + affine", "(a) + rotation fields"])
    style(ax[0], "Korn constant $C_K$ (larger = thinner / more flexible)", "static solution rel. L2 error")
    ax[0].set_title("Laplacian basis degrades with $C_K$", color=INK, loc="left")
    bases2 = [b for b in ["neo_rot_sc", "neo_rot_inv2_sc", "neo_inv2_sc", "elastic_matched_sc"] if b in set(worst.basis)]
    labels2 = {"neo_rot_sc": "(b) + static corr.", "neo_rot_inv2_sc": "(c) + 2 inv. iter",
               "neo_inv2_sc": "(c) without rotation fields", "elastic_matched_sc": "exact modes, same dim + static corr."}
    series(ax[1], worst, "korn", "ratio", bases2, [labels2[b] for b in bases2])
    ax[1].axhline(0.2, color=INK2, lw=1, ls=":")
    ax[1].text(ax[1].get_xlim()[0] * 1.05, 0.215, "pass ≤ 0.2", color=INK2, fontsize=9)
    style(ax[1], "Korn constant $C_K$", "model error / lump signal (interior, worst depth)")
    ax[1].set_title("Enriched bases vs the 0.2 criterion", color=INK, loc="left")
    series(ax[2], hom, "thick", "relL2_hom", ["neo", "neo_rot"], ["Laplacian ×3 + affine", "(a) + rotation fields"])
    style(ax[2], "thickness index $d_{max} / V^{1/3}$ (smaller = thinner)", "static solution rel. L2 error")
    ax[2].set_title("… and with thinness", color=INK, loc="left")
    for x in ax:
        x.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    top = hom[hom.basis == "neo"].groupby(["family", "idx"], as_index=False).agg(
        korn=("korn", "first"), thick=("thick", "first"), relL2_hom=("relL2_hom", "max"))
    for k, (_, r) in enumerate(top.iterrows()):                          # shape names above the highest point
        for axi, xc in ((ax[0], "korn"), (ax[2], "thick")):
            axi.annotate(f"{r.family}{r.idx}", (r[xc], r.relL2_hom), xytext=(0, 8 + 7 * (k % 2)),
                         textcoords="offset points", fontsize=7, color=INK2, ha="center")
    fig.tight_layout()
    out = os.path.join(ROOT, "e1", f"a4_{a.tag}.png")
    fig.savefig(out, dpi=110, facecolor="#fcfcfb")
    print(out)


if __name__ == "__main__":
    main()

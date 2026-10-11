"""A4 with the real shapes added -> stage3/real_shapes/a4_real.png

Panel 1  Laplacian x3 + affine basis: static-solution rel. L2 error (smooth load, nu = 0.3) vs Korn constant
Panel 2  basis (c): model error / lump signal (interior, worst over loads and depths, nu = 0.3) vs Korn constant
Panel 3  basis (c): same ratio vs the scalar NEO span loss of the shape (quasi 8192), real shapes only
Points: procedural shapes (A, 16) and real shapes (C); each point is one shape, real shapes are labelled.

    python stage3/a4_real_plot.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SYN, REAL = "#2a78d6", "#eb6834"                                      # reference categorical slots 1, 2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax, xl, yl):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xl, color=INK2); ax.set_ylabel(yl, color=INK2)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2)


def per_shape(df, nu=0.3):
    d = df[df.nu == nu]
    lap = d[(d.basis == "neo") & (d.load == "smooth") & (d.depth == "mid")].groupby("family").agg(
        korn=("korn", "first"), err=("relL2_hom", "first"))
    c = d[d.basis == "neo_rot_inv2_sc"].groupby("family").agg(ratio=("ratio_int", "max"), ratio_s=("ratio_surf", "max"))
    return lap.join(c)


def main():
    A = pd.read_csv(os.path.join(ROOT, "e1", "e1_v2_full.csv"))
    A["family"] = A.family + A.idx.astype(str)
    R = pd.read_csv(os.path.join(ROOT, "e1", "e1_v2_real.csv"))
    a, r = per_shape(A), per_shape(R)
    nb = pd.read_csv(os.path.join(ROOT, "real_shapes", "a_basis.csv"))
    nb = nb[(nb.kind == "quasi") & (nb.N == 8192)].set_index("shape")
    r = r.join(nb[["span96"]])
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))
    for axi, col, yl in ((ax[0], "err", "Laplacian×3 basis: static rel. L2 error (ν=0.3)"),
                         (ax[1], "ratio", "basis (c): model error / lump signal (worst, ν=0.3)")):
        axi.plot(a.korn, a[col], "o", ms=8, color=SYN, mec="white", mew=1.2, label="procedural shapes (A)")
        axi.plot(r.korn, r[col], "s", ms=9, color=REAL, mec="white", mew=1.2, label="real shapes (C)")
        for name, row in r.iterrows():
            axi.annotate(name.replace("thingi", "T"), (row.korn, row[col]), xytext=(0, 7), textcoords="offset points",
                         fontsize=7, color=INK2, ha="center")
        style(axi, "Korn constant $C_K$", yl)
        axi.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper left")
    ax[1].axhline(0.2, color=INK2, lw=1, ls=":")
    ax[0].set_title("Laplacian basis vs $C_K$ (real shapes added)", color=INK, loc="left")
    ax[1].set_title("Enriched basis (c) vs the 0.2 criterion", color=INK, loc="left")
    ax[2].plot(r.span96, r.ratio, "s", ms=9, color=REAL, mec="white", mew=1.2)
    for name, row in r.iterrows():
        ax[2].annotate(name.replace("thingi", "T"), (row.span96, row.ratio), xytext=(0, 7), textcoords="offset points",
                       fontsize=7, color=INK2, ha="center")
    ax[2].axhline(0.2, color=INK2, lw=1, ls=":")
    style(ax[2], "scalar NEO span loss, modes 1–96 (quasi 8192)", "basis (c): model error / lump signal")
    ax[2].set_title("Does a weaker NEO basis hurt (c)?", color=INK, loc="left")
    fig.tight_layout()
    out = os.path.join(ROOT, "real_shapes", "a4_real.png")
    fig.savefig(out, dpi=110, facecolor="#fcfcfb")
    print(out)


if __name__ == "__main__":
    main()

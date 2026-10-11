"""Figure of the four A-quick shapes (sh_ball, cup, torus, c_tube; idx 0) -> stage3/e1/quick_shapes.png

One row per shape:
  col 1-2  surface points (nodes with nonzero boundary-face area), two view angles
  col 3    planar section through the three lump centres (shallow / mid / deep, e1_v2.depth_lumps); tets cut by the
           plane are drawn as polygons, coloured by the lump they fall in (E > 3), grey otherwise
  col 4    |u| of the reference solution on the surface points: full-order ANP, nu = 0.3, homogeneous E = 1,
           "smooth" two-spot Gaussian pressure (same rng draw as e1_v2.task); load centres marked

    python stage3/quick_shapes_plot.py            (CPU, ~1 min per shape)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap

import e1_basis as E
import e1_v2 as V
import elastic_ops as EO

SHAPES = ["sh_ball", "cup", "torus", "c_tube"]
NU = 0.3
LUMP_C = {"shallow": "#2a78d6", "mid": "#eb6834", "deep": "#1baf7a"}   # validated categorical slots 1-3 (a4_plot)
INK, INK2, BG, GREY = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d4"
SEQ = LinearSegmentedColormap.from_list("seq", ["#dce9f7", "#7fb0e6", "#2a78d6", "#123f75"])   # one hue, light -> dark
VIEWS = [(20, 35), (-10, 145)]


def load_centres(Y, T, family, idx):
    """Replicates the rng draws of e1_v2.task up to pinch_load and returns the two spot centres."""
    rng = np.random.default_rng([E.SEED, E.FAMILIES.index(family), idx, 1])
    E.lumps(Y, rng, 1 + idx % 2)
    F, _ = E.boundary_faces(T, Y)
    cen = Y[F].mean(1)
    d = rng.normal(size=3); d /= np.linalg.norm(d)
    return np.stack([cen[np.argmax(cen @ d)], cen[np.argmin(cen @ d)]])


def section(Y, T, p0, n):
    """Cut every tet crossing the plane (p0, n) -> list of 2D polygons, their 3D centroids, the in-plane frame."""
    a = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0]); a /= np.linalg.norm(a)
    b = np.cross(n, a)
    s = (Y - p0) @ n
    st = s[T]
    cut = (st.min(1) < 0) & (st.max(1) > 0)
    polys, cents = [], []
    for t, sv in zip(T[cut], st[cut]):
        pts = []
        for i in range(4):
            for j in range(i + 1, 4):
                if sv[i] * sv[j] < 0:
                    w = sv[i] / (sv[i] - sv[j])
                    pts.append(Y[t[i]] + w * (Y[t[j]] - Y[t[i]]))
        pts = np.array(pts)
        c = pts.mean(0)
        q = np.c_[(pts - p0) @ a, (pts - p0) @ b]
        qc = q.mean(0)
        polys.append(q[np.argsort(np.arctan2(q[:, 1] - qc[1], q[:, 0] - qc[0]))])
        cents.append(c)
    return polys, np.array(cents), a, b


def clean3d(ax, Y):
    ax.set_box_aspect(np.ptp(Y, 0))
    ax.set_axis_off()
    ax.set_facecolor(BG)


def main():
    fig = plt.figure(figsize=(18, 17.5), facecolor=BG)
    for row, fam in enumerate(SHAPES):
        z = np.load(E.cache_path(fam, 0))
        Y, op = z["Y"], E.Op(z)
        m3 = np.repeat(op.m, 3)
        ws = V.surface_weights(Y, op.tets)
        surf = ws > 0
        lumpset, dmax = V.depth_lumps(Y, np.where(surf)[0])
        rng = np.random.default_rng([E.SEED, E.FAMILIES.index(fam), 0, 1])
        E.lumps(Y, rng, 1)
        f = E.pinch_load(Y, op.tets, rng)["smooth"]
        K = EO.anp_stiffness(op, np.ones(len(op.vol)), NU)
        u = E.full_solve(K, f, Y, m3).reshape(-1, 3)
        umag = np.linalg.norm(u, axis=1)
        pc = load_centres(Y, op.tets, fam, 0)
        P = Y[surf]
        print(fam, f"N={len(Y)} surface={surf.sum()} dmax={dmax:.3f} |u|max={umag.max():.3g}", flush=True)

        for k, (el, az) in enumerate(VIEWS):                           # col 1-2: surface points
            ax = fig.add_subplot(4, 4, 4 * row + k + 1, projection="3d")
            depth = P @ np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)),
                                  np.cos(np.radians(el)) * np.sin(np.radians(az)), np.sin(np.radians(el))])
            ax.scatter(*P.T, s=1.2, c=depth, cmap=LinearSegmentedColormap.from_list("g", ["#b9b8b3", "#3d3c39"]),
                       depthshade=False, linewidths=0)
            ax.view_init(el, az); clean3d(ax, Y)
            ax.set_title(f"{fam}  ·  surface points ({surf.sum()} of {len(Y)})" if k == 0 else f"view {k + 1}  (elev {el}°, azim {az}°)",
                         color=INK if k == 0 else INK2, loc="left", fontsize=11)

        ax = fig.add_subplot(4, 4, 4 * row + 3)                        # col 3: section through the lumps
        C = np.stack([lumpset[d][0] for d in V.DEPTHS])
        n = np.cross(C[1] - C[0], C[2] - C[0])
        if np.linalg.norm(n) < 1e-6:                                  # collinear centres -> any plane through them
            n = np.cross(C[2] - C[0], [0, 0, 1])
        n /= np.linalg.norm(n)
        p0 = C.mean(0)
        polys, cents, a, b = section(Y, op.tets, p0, n)
        col = np.array([GREY] * len(polys), dtype=object)
        for dn in V.DEPTHS:
            c, r, _ = lumpset[dn]
            col[V.lump_field(cents, c, r) > 3] = LUMP_C[dn]
        ax.add_collection(PolyCollection(polys, facecolors=list(col), edgecolors=list(col), linewidths=0.3))
        for dn in V.DEPTHS:
            c, r, dep = lumpset[dn]
            q = [(c - p0) @ a, (c - p0) @ b]
            ax.add_patch(plt.Circle(q, r, fill=False, color=INK, lw=0.8, ls="--"))
            ax.plot(*q, "+", color=INK, ms=6)
            ax.plot([], [], "s", color=LUMP_C[dn], ms=9, label=f"{dn}: depth {dep:.2f}")
        allq = np.concatenate(polys)
        ax.set_xlim(allq[:, 0].min() - 0.05, allq[:, 0].max() + 0.05)
        ax.set_ylim(allq[:, 1].min() - 0.05, allq[:, 1].max() + 0.05)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper left", bbox_to_anchor=(1.0, 1.0),
                  title=f"lump E = 5, r = {lumpset['deep'][1]:.3f}\n(one at a time)", title_fontsize=8)
        ax.set_title("section through the 3 lump centres", color=INK, loc="left", fontsize=11)

        ax = fig.add_subplot(4, 4, 4 * row + 4, projection="3d")      # col 4: |u|, reference solution
        el, az = VIEWS[0]
        sc = ax.scatter(*P.T, s=1.6, c=umag[surf], cmap=SEQ, vmin=0, depthshade=False, linewidths=0)
        ax.scatter(*pc.T, s=90, marker="v", color="#eb6834", edgecolors="white", linewidths=1, depthshade=False)
        ax.view_init(el, az); clean3d(ax, Y)
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label("|u|", color=INK2); cb.ax.tick_params(colors=INK2, labelsize=8); cb.outline.set_visible(False)
        ax.set_title(f"|u| reference (ANP, ν = {NU}, E = 1)\nsmooth two-spot load, ▼ = load centres", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    out = os.path.join(E.OUT, "quick_shapes.png")
    fig.savefig(out, dpi=110, facecolor=BG)
    print(out)


if __name__ == "__main__":
    main()

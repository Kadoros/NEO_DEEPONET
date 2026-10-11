"""Experiment driver: FEM references, point-cloud spectral solutions, errors,
eigenvalue validation, CSVs and figures."""
import os
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfem
from matplotlib.ticker import NullFormatter
from matplotlib.tri import Triangulation
from scipy.special import jnp_zeros

import fem_reference as fem
from laplacian import SHAPES, build_delaunay_laplacian, build_laplacian, sample_point_cloud
from spectral import eigendecompose, rel_error, spectral_coefficients, spectral_solution

# ---------------------------------------------------------------- settings --
SEED = 20260924
ALPHA = 1.0
TIMES = [0.0005, 0.002, 0.01, 0.05]
KS = [8, 16, 32, 64, 128, 256]
K_MAX = 256
NS = [1000, 2000, 4000, 8000]
SAMPLINGS = ["uniform", "nonuniform"]
N_TRIALS = 3                       # independent clouds per (shape, sampling, N)
SHAPE_NAMES = ["square", "disk", "square_hole"]
IC_NAMES = ["gaussian", "step"]
GAUSS_SIGMA = 0.1                  # u0 = exp(-|x-c|^2 / (2 sigma^2))
STEP_R = 0.15                      # u0 = 1 inside circle of radius STEP_R
H_COARSE, STEPS_COARSE = 0.01, 100   # FEM: mesh size, CN steps per output interval
H_FINE, STEPS_FINE = 0.005, 200      # refined FEM = reference
FIELD_CASE = dict(shape="square_hole", sampling="uniform", N=8000, trial=0)

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "cache")

# plot palette: ordinal blue ramp for time, categorical blue/orange for sampling
T_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
N_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SAMPLING_COLORS = {"uniform": "#2a78d6", "nonuniform": "#eb6834"}
LAP_COLORS = {"pointcloud": "#2a78d6", "delaunay": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SHAPE_TITLES = {"square": "unit square", "disk": "unit disk", "square_hole": "square with hole"}


# ------------------------------------------------------ initial conditions --
def ic_setup(shape_name):
    """Random (seeded) Gaussian centre and step-circle centre for a shape."""
    shape = SHAPES[shape_name]
    rng = np.random.default_rng([SEED, SHAPE_NAMES.index(shape_name), 7])
    xmin, xmax, ymin, ymax = shape.bbox

    def draw(margin):
        while True:
            c = np.array([[rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)]])
            if shape.inside(c, margin)[0]:
                return c[0]

    g_c = draw(GAUSS_SIGMA)
    s_c = draw(STEP_R + 0.03)      # step circle stays clear of the boundary
    return g_c, s_c


def ic_functions(g_c, s_c):
    gaussian = lambda x: np.exp(-((x - g_c) ** 2).sum(1) / (2 * GAUSS_SIGMA ** 2))
    step = lambda x: (((x - s_c) ** 2).sum(1) < STEP_R ** 2).astype(float)
    return [gaussian, step]


# ---------------------------------------------------------- FEM reference --
def _ref_path(shape_name):
    return os.path.join(CACHE, f"fem_{shape_name}.npz")


def fem_task(shape_name):
    path = _ref_path(shape_name)
    if os.path.exists(path):
        return path
    t0 = time.time()
    g_c, s_c = ic_setup(shape_name)
    ics = ic_functions(g_c, s_c)
    coarse = fem.compute_reference(shape_name, ics, s_c, STEP_R, TIMES, H_COARSE, STEPS_COARSE)
    fine = fem.compute_reference(shape_name, ics, s_c, STEP_R, TIMES, H_FINE, STEPS_FINE)
    conv = fem.self_convergence(coarse, fine)
    np.savez(path, p=fine["mesh"].p, t=fine["mesh"].t, U=fine["U"], conv=conv,
             n_coarse=coarse["mesh"].p.shape[1], n_fine=fine["mesh"].p.shape[1],
             g_c=g_c, s_c=s_c)
    print(f"  FEM {shape_name}: {coarse['mesh'].p.shape[1]} -> {fine['mesh'].p.shape[1]} nodes, "
          f"{time.time() - t0:.0f}s", flush=True)
    return path


# ------------------------------------------------------- exact eigenvalues --
def exact_eigenvalues(shape_name, k):
    if shape_name == "square":
        m = np.arange(40)
        lam = np.pi ** 2 * (m[:, None] ** 2 + m[None, :] ** 2).ravel()
    elif shape_name == "disk":
        lam = [0.0]
        for order in range(60):
            z = jnp_zeros(order, 30) ** 2
            lam += list(z) * (1 if order == 0 else 2)
        lam = np.array(lam)
    else:
        return None
    return np.sort(lam)[:k]


# ------------------------------------------------------- point-cloud runs --
_REFS, _INTERP = {}, {}


def _reference(shape_name):
    if shape_name not in _REFS:
        d = dict(np.load(_ref_path(shape_name)))
        mesh = skfem.MeshTri(d["p"], d["t"])
        _, _, M = fem.assemble(mesh)
        # ∫u dx is conserved under Neumann BCs: exact IC mass per IC
        d["ic_mass"] = np.asarray(M.sum(0)).ravel() @ d["U"][-1]
        _REFS[shape_name] = d
        _INTERP[shape_name] = fem.P1Interpolator(mesh)
    return _REFS[shape_name], _INTERP[shape_name]


def cloud_task(args):
    """One point cloud: sample, Laplacian(s), eigendecomposition, errors.

    'pointcloud' is the method under study; for uniform clouds the Delaunay
    cotan Laplacian on the same points is run as a diagnostic baseline."""
    shape_name, sampling, N, trial = args
    shape = SHAPES[shape_name]
    ref, interp = _reference(shape_name)
    rng = np.random.default_rng([SEED, SHAPE_NAMES.index(shape_name),
                                 SAMPLINGS.index(sampling), N, trial])
    pts, is_b = sample_point_cloud(shape, N, sampling, rng)
    u0 = np.stack([f(pts) for f in ic_functions(ref["g_c"], ref["s_c"])], 1)   # (N, n_ic)
    P = interp.matrix(pts)
    u_ref = np.stack([P @ ref["U"][i] for i in range(len(TIMES))])              # (nt, N, n_ic)
    lam_ex = exact_eigenvalues(shape_name, K_MAX)

    variants = {"pointcloud": lambda: build_laplacian(pts)}
    if sampling == "uniform":
        variants["delaunay"] = lambda: build_delaunay_laplacian(pts, shape)

    err_rows, eig_rows, cloud_rows = [], [], []
    for lap_name, build in variants.items():
        t0 = time.time()
        L, M = build()
        lam, Phi = eigendecompose(L, M, k=K_MAX)
        t_eig = time.time() - t0
        Md = M.diagonal()
        coeffs = spectral_coefficients(Phi, M, u0)

        key = dict(shape=shape_name, sampling=sampling, N=N, trial=trial, laplacian=lap_name)
        sols = {}
        for K in KS:
            u = spectral_solution(lam, Phi, coeffs, TIMES, K, ALPHA)            # (nt, N, n_ic)
            sols[K] = u
            for i, t in enumerate(TIMES):
                e = rel_error(u[i], u_ref[i], Md)
                for j, ic in enumerate(IC_NAMES):
                    err_rows.append({**key, "ic": ic, "K": K, "t": t, "rel_err": e[j]})

        if lam_ex is not None:
            for i in range(K_MAX):
                eig_rows.append({**key, "index": i, "lambda_pc": lam[i], "lambda_exact": lam_ex[i],
                                 "rel_err": abs(lam[i] - lam_ex[i]) / lam_ex[i] if i > 0 else np.nan})

        mass_err = (Md @ u0 - ref["ic_mass"]) / ref["ic_mass"]
        cloud_rows.append({**key, "n_boundary": int(is_b.sum()), "mass_sum": Md.sum(),
                           "mass_sum_over_area": Md.sum() / shape.area,
                           **{f"u0_mass_relerr_{ic}": mass_err[j] for j, ic in enumerate(IC_NAMES)},
                           "lambda_1": lam[1], "lambda_255": lam[-1], "seconds": t_eig})

        if lap_name == "pointcloud" and all(key[k] == v for k, v in FIELD_CASE.items()):
            np.savez(os.path.join(CACHE, "field_case.npz"), pts=pts, u_ref=u_ref, Md=Md,
                     **{f"u_K{K}": sols[K] for K in KS})
    return err_rows, eig_rows, cloud_rows


# --------------------------------------------------------------- plotting --
def _style(ax):
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def _logx(ax, ticks, base=10):
    ax.set_xscale("log", base=base); ax.set_yscale("log")
    ax.set_xticks(ticks); ax.set_xticklabels(ticks)
    ax.xaxis.set_minor_formatter(NullFormatter())


def _agg(df, by):
    return df.groupby(by)["rel_err"].agg(["mean", "min", "max"]).reset_index()


def _grid_fig(title):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True, constrained_layout=True)
    fig.suptitle(title, color=INK, fontsize=12)
    for r, ic in enumerate(IC_NAMES):
        for c, s in enumerate(SHAPE_NAMES):
            _style(axes[r, c])
            axes[r, c].set_title(f"{SHAPE_TITLES[s]} · {ic} IC", color=INK, fontsize=10)
    return fig, axes


def plot_K(err):
    N = max(NS)
    fig, axes = _grid_fig(f"Relative M-weighted L2 error vs number of modes K  "
                          f"(uniform sampling, N={N}, mean of {N_TRIALS} clouds, band = min–max)")
    d = err[(err.sampling == "uniform") & (err.N == N) & (err.laplacian == "pointcloud")]
    for r, ic in enumerate(IC_NAMES):
        for c, s in enumerate(SHAPE_NAMES):
            ax = axes[r, c]
            for t, col in zip(TIMES, T_COLORS):
                a = _agg(d[(d.ic == ic) & (d["shape"] == s) & (d.t == t)], "K")
                ax.plot(a.K, a["mean"], "-o", color=col, lw=2, ms=4, label=f"t = {t:g}")
                ax.fill_between(a.K, a["min"], a["max"], color=col, alpha=0.15, lw=0)
            _logx(ax, KS, base=2)
            if r == 1: ax.set_xlabel("K (modes)", color=INK2)
            if c == 0: ax.set_ylabel("‖u − u_ref‖_M / ‖u_ref‖_M", color=INK2)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.savefig(os.path.join(RESULTS, "fig1_K_vs_error.png"), dpi=150)
    plt.close(fig)


def plot_N(err):
    fig, axes = _grid_fig(f"Relative error vs N  (uniform sampling, K={K_MAX}, "
                          f"mean of {N_TRIALS} clouds; dotted = N^-1/2 guide)")
    d = err[(err.sampling == "uniform") & (err.K == K_MAX) & (err.laplacian == "pointcloud")]
    for r, ic in enumerate(IC_NAMES):
        for c, s in enumerate(SHAPE_NAMES):
            ax = axes[r, c]
            for t, col in zip(TIMES, T_COLORS):
                a = _agg(d[(d.ic == ic) & (d["shape"] == s) & (d.t == t)], "N")
                ax.plot(a.N, a["mean"], "-o", color=col, lw=2, ms=4, label=f"t = {t:g}")
                ax.fill_between(a.N, a["min"], a["max"], color=col, alpha=0.15, lw=0)
            a = _agg(d[(d.ic == ic) & (d["shape"] == s) & (d.t == TIMES[-1])], "N")
            guide = a["mean"].iloc[0] * (np.array(NS) / NS[0]) ** -0.5
            ax.plot(NS, guide, ":", color=INK2, lw=1)
            _logx(ax, NS)
            if r == 1: ax.set_xlabel("N (points)", color=INK2)
            if c == 0: ax.set_ylabel("relative error", color=INK2)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.savefig(os.path.join(RESULTS, "fig2_N_vs_error.png"), dpi=150)
    plt.close(fig)


def plot_sampling(err):
    ts = [TIMES[1], TIMES[2]]
    fig, axes = _grid_fig(f"Uniform vs non-uniform (5x density on x < x_mid) sampling, K={K_MAX}  "
                          f"(solid t={ts[0]:g}, dashed t={ts[1]:g})")
    d = err[(err.K == K_MAX) & (err.laplacian == "pointcloud")]
    for r, ic in enumerate(IC_NAMES):
        for c, s in enumerate(SHAPE_NAMES):
            ax = axes[r, c]
            for smp in SAMPLINGS:
                for t, ls in zip(ts, ["-", "--"]):
                    a = _agg(d[(d.ic == ic) & (d["shape"] == s) & (d.t == t) & (d.sampling == smp)], "N")
                    ax.plot(a.N, a["mean"], ls, marker="o", color=SAMPLING_COLORS[smp], lw=2, ms=4,
                            label=f"{smp}, t = {t:g}")
                    ax.fill_between(a.N, a["min"], a["max"], color=SAMPLING_COLORS[smp],
                                    alpha=0.12, lw=0)
            _logx(ax, NS)
            if r == 1: ax.set_xlabel("N (points)", color=INK2)
            if c == 0: ax.set_ylabel("relative error", color=INK2)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.savefig(os.path.join(RESULTS, "fig3_uniform_vs_nonuniform.png"), dpi=150)
    plt.close(fig)


def plot_eigen(eig):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, constrained_layout=True)
    fig.suptitle(f"Eigenvalue relative error |λ_pc − λ_exact| / λ_exact  (mean of {N_TRIALS} clouds)",
                 color=INK, fontsize=12)
    for ax, s, name in zip(axes, ["square", "disk"],
                           ["unit square: π²(m²+n²)", "unit disk: j'ₘₖ² (Bessel)"]):
        _style(ax)
        d = eig[(eig["shape"] == s) & (eig["index"] > 0) & (eig.laplacian == "pointcloud")]
        for N, col in zip(NS, N_COLORS):
            a = d[(d.N == N) & (d.sampling == "uniform")].groupby("index")["rel_err"].mean()
            ax.plot(a.index, a.values, color=col, lw=1.5, label=f"uniform N={N}")
        a = d[(d.N == max(NS)) & (d.sampling == "nonuniform")].groupby("index")["rel_err"].mean()
        ax.plot(a.index, a.values, "--", color=SAMPLING_COLORS["nonuniform"], lw=1.5,
                label=f"non-uniform N={max(NS)}")
        dd = eig[(eig["shape"] == s) & (eig["index"] > 0) & (eig.laplacian == "delaunay")]
        a = dd[dd.N == max(NS)].groupby("index")["rel_err"].mean()
        ax.plot(a.index, a.values, ":", color=LAP_COLORS["delaunay"], lw=1.5,
                label=f"Delaunay cotan (diagnostic), uniform N={max(NS)}")
        ax.set_yscale("log"); ax.set_xlabel("eigen index i", color=INK2)
        ax.set_title(name, color=INK, fontsize=10)
    axes[0].set_ylabel("relative error", color=INK2)
    axes[0].legend(fontsize=8, frameon=False)
    fig.savefig(os.path.join(RESULTS, "fig4_eigenvalue_error.png"), dpi=150)
    plt.close(fig)


def plot_field(K=64, t_idx=1):
    f = np.load(os.path.join(CACHE, "field_case.npz"))
    pts, u_ref, u = f["pts"], f["u_ref"][t_idx], f[f"u_K{K}"][t_idx]
    shape = SHAPES[FIELD_CASE["shape"]]
    tri = Triangulation(pts[:, 0], pts[:, 1])
    cen = pts[tri.triangles].mean(1)
    tri.set_mask(~shape.inside(cen))
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8), constrained_layout=True)
    fig.suptitle(f"{SHAPE_TITLES[FIELD_CASE['shape']]}, {FIELD_CASE['sampling']} N={FIELD_CASE['N']}, "
                 f"K={K}, t={TIMES[t_idx]:g}", color=INK, fontsize=12)
    for r, ic in enumerate(IC_NAMES):
        a, b = u_ref[:, r], u[:, r]
        vmin, vmax = min(a.min(), b.min()), max(a.max(), b.max())
        for c, (v, name, cmap) in enumerate([(a, "FEM reference", "Blues"),
                                             (b, "spectral", "Blues"),
                                             (b - a, "spectral − reference", "RdBu_r")]):
            ax = axes[r, c]
            if c < 2:
                im = ax.tripcolor(tri, v, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                m = np.abs(v).max()
                im = ax.tripcolor(tri, v, shading="gouraud", cmap=cmap, vmin=-m, vmax=m)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 2:
                name += f"  (rel. err {rel_error(b, a, f['Md']):.1e})"
            ax.set_title(f"{ic} IC · {name}", color=INK, fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(os.path.join(RESULTS, "fig5_field_comparison.png"), dpi=150)
    plt.close(fig)


def plot_laplacian_ablation(err):
    ts = [TIMES[1], TIMES[3]]
    fig, axes = _grid_fig(f"Diagnostic: robust point-cloud Laplacian vs Delaunay cotan Laplacian on the "
                          f"same uniform clouds, K={K_MAX}  (solid t={ts[0]:g}, dashed t={ts[1]:g})")
    d = err[(err.K == K_MAX) & (err.sampling == "uniform")]
    for r, ic in enumerate(IC_NAMES):
        for c, s in enumerate(SHAPE_NAMES):
            ax = axes[r, c]
            for lap in ["pointcloud", "delaunay"]:
                for t, ls in zip(ts, ["-", "--"]):
                    a = _agg(d[(d.ic == ic) & (d["shape"] == s) & (d.t == t) & (d.laplacian == lap)], "N")
                    ax.plot(a.N, a["mean"], ls, marker="o", color=LAP_COLORS[lap], lw=2, ms=4,
                            label=f"{lap}, t = {t:g}")
                    ax.fill_between(a.N, a["min"], a["max"], color=LAP_COLORS[lap], alpha=0.12, lw=0)
            _logx(ax, NS)
            if r == 1: ax.set_xlabel("N (points)", color=INK2)
            if c == 0: ax.set_ylabel("relative error", color=INK2)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.savefig(os.path.join(RESULTS, "fig6_laplacian_diagnostic.png"), dpi=150)
    plt.close(fig)


def make_figures(err=None, eig=None):
    err = pd.read_csv(os.path.join(RESULTS, "errors.csv")) if err is None else err
    eig = pd.read_csv(os.path.join(RESULTS, "eigenvalues.csv")) if eig is None else eig
    plot_K(err); plot_N(err); plot_sampling(err); plot_eigen(eig); plot_field()
    plot_laplacian_ablation(err)


# ------------------------------------------------------------------ tables --
def eigen_table(eig):
    idx = [1, 2, 5, 10, 20, 50, 100, 150, 200, 255]
    d = eig[(eig["shape"] == "square") & eig["index"].isin(idx)]
    tab = d.groupby(["laplacian", "sampling", "N", "index"])["rel_err"].mean().unstack("index")
    tab = tab.loc[["pointcloud", "delaunay"]]
    tab.columns = [f"i={i}" for i in tab.columns]
    lam = exact_eigenvalues("square", K_MAX)[idx] / np.pi ** 2
    header = "exact λ/π²: " + ", ".join(f"i={i}: {v:g}" for i, v in zip(idx, lam))
    return tab, header


# -------------------------------------------------------------------- main --
def main(workers=None):
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    workers = workers or min(24, os.cpu_count())

    t0 = time.time()
    print("[1/3] FEM references (coarse + refined) ...", flush=True)
    with Pool(len(SHAPE_NAMES)) as pool:
        pool.map(fem_task, SHAPE_NAMES)
    conv_rows = []
    for s in SHAPE_NAMES:
        d = np.load(_ref_path(s))
        for i, t in enumerate(TIMES):
            for j, ic in enumerate(IC_NAMES):
                conv_rows.append(dict(shape=s, ic=ic, t=t, rel_diff_coarse_vs_fine=d["conv"][i, j],
                                      nodes_coarse=int(d["n_coarse"]), nodes_fine=int(d["n_fine"]),
                                      h_coarse=H_COARSE, h_fine=H_FINE,
                                      cn_steps_per_interval_coarse=STEPS_COARSE,
                                      cn_steps_per_interval_fine=STEPS_FINE))
    conv = pd.DataFrame(conv_rows)
    conv.to_csv(os.path.join(RESULTS, "fem_convergence.csv"), index=False)
    print(conv.pivot_table(index=["shape", "ic"], columns="t",
                           values="rel_diff_coarse_vs_fine").to_string(float_format="%.2e"))

    print(f"[2/3] point clouds + eigendecomposition ({workers} workers) ...", flush=True)
    tasks = [(s, smp, N, tr) for N in sorted(NS, reverse=True) for s in SHAPE_NAMES
             for smp in SAMPLINGS for tr in range(N_TRIALS)]
    err_rows, eig_rows, cloud_rows = [], [], []
    with Pool(workers) as pool:
        for e, g, c in pool.imap_unordered(cloud_task, tasks):
            err_rows += e; eig_rows += g; cloud_rows += c
    keys = ["laplacian", "shape", "sampling", "N", "trial"]
    err = pd.DataFrame(err_rows).sort_values(keys + ["ic", "K", "t"])
    eig = pd.DataFrame(eig_rows).sort_values(keys + ["index"])
    clouds = pd.DataFrame(cloud_rows).sort_values(keys)
    err.to_csv(os.path.join(RESULTS, "errors.csv"), index=False)
    eig.to_csv(os.path.join(RESULTS, "eigenvalues.csv"), index=False)
    clouds.to_csv(os.path.join(RESULTS, "clouds.csv"), index=False)
    summary = err.groupby(["laplacian", "shape", "ic", "sampling", "N", "K", "t"])["rel_err"].mean().reset_index()
    summary.to_csv(os.path.join(RESULTS, "errors_mean_over_trials.csv"), index=False)

    tab, header = eigen_table(eig)
    tab.to_csv(os.path.join(RESULTS, "eigenvalue_table_square.csv"))
    with open(os.path.join(RESULTS, "eigenvalue_table_square.md"), "w") as fh:
        fh.write("Relative eigenvalue error on the unit square (mean over trials)\n\n")
        fh.write(header + "\n\n")
        fh.write(_markdown(tab))
        fh.write("\n")
    print("Square eigenvalue relative error (mean over trials)")
    print(header)
    print(tab.to_string(float_format="%.2e"))

    print("[3/3] figures ...", flush=True)
    make_figures(err, eig)
    print(f"done in {time.time() - t0:.0f}s -> {RESULTS}")


def _markdown(tab):
    names = list(tab.index.names)
    lines = ["| " + " | ".join(names + list(tab.columns)) + " |",
             "|" + "---|" * (len(names) + len(tab.columns))]
    for idx, row in tab.iterrows():
        idx = idx if isinstance(idx, tuple) else (idx,)
        lines.append("| " + " | ".join([str(v) for v in idx] + [f"{v:.2e}" for v in row]) + " |")
    return "\n".join(lines)

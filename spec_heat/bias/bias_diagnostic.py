"""Diagnose the low-mode eigenvalue bias of robust_laplacian.point_cloud_laplacian.

    python bias/bias_diagnostic.py        (from spec_heat/, env spec-heat)

Clouds are the uniform clouds of the main experiment (same seeds, trials 0-2).
Outputs (this folder): mass_ratio.csv, mass_rescaled_eigs.csv, energy_ratio.csv,
n_neighbors.csv, calibrated.csv, bias_tables.md
"""
import os
import sys

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

from multiprocessing import Pool

import numpy as np
import pandas as pd
import robust_laplacian
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import skfem
from skfem.helpers import dot, grad
from skfem.models.poisson import laplace, mass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
# keep gmsh/FLTK preference files inside the project (this process only)
os.environ["HOME"] = os.path.join(os.path.dirname(HERE), ".gmsh_home")
os.makedirs(os.environ["HOME"], exist_ok=True)

import fem_reference as fem  # noqa: E402
from experiments import SEED, SHAPE_NAMES, SAMPLINGS, exact_eigenvalues  # noqa: E402
from laplacian import SHAPES, sample_point_cloud  # noqa: E402

NS = [1000, 4000, 8000]
N_TRIALS = 3
NEIGHBORS = [10, 20, 30, 50]
K_LOW = 10                  # "low modes" = eigen indices 1..10
K_EIG = K_LOW + 1
H_REF = 0.01                # P2 FEM reference mesh size
# test functions for the energy check: plane waves |k| = pi, 2pi in 4 directions
WAVES = [(a * np.cos(th), a * np.sin(th)) for a in (np.pi, 2 * np.pi)
         for th in np.deg2rad([0, 45, 90, 135])]


# ------------------------------------------------------------ references --
def fem_reference(shape_name):
    """P2 FEM eigenvalues (first K_EIG) and exact integrals of the test functions."""
    mesh = fem.build_mesh(shape_name, step_center=_any_inner_point(shape_name),
                          step_radius=0.05, h=H_REF)
    basis = skfem.Basis(mesh, skfem.ElementTriP2(), intorder=8)
    K, M = laplace.assemble(basis), mass.assemble(basis)
    lam = np.sort(sla.eigsh(K, k=K_EIG, M=M, sigma=-1e-8)[0])

    ints = []
    for kx, ky in WAVES:
        @skfem.Functional
        def f2(w, kx=kx, ky=ky):
            return np.cos(kx * w.x[0] + ky * w.x[1]) ** 2

        @skfem.Functional
        def g2(w, kx=kx, ky=ky):
            return (kx ** 2 + ky ** 2) * np.sin(kx * w.x[0] + ky * w.x[1]) ** 2
        ints.append((f2.assemble(basis), g2.assemble(basis)))

    @skfem.Functional
    def one(w):
        return 1.0 + 0 * w.x[0]
    return dict(lam=lam, wave_ints=np.array(ints), area_mesh=one.assemble(basis))


def _any_inner_point(shape_name):
    return {"square": (0.25, 0.25), "disk": (0.0, -0.5), "square_hole": (0.15, 0.15)}[shape_name]


# ------------------------------------------------------------------ tasks --
def cloud(shape_name, N, trial):
    rng = np.random.default_rng([SEED, SHAPE_NAMES.index(shape_name),
                                 SAMPLINGS.index("uniform"), N, trial])
    return sample_point_cloud(SHAPES[shape_name], N, "uniform", rng)[0]


def low_eigs(L, M):
    return np.sort(sla.eigsh(L, k=K_EIG, M=M, sigma=-1e-8)[0])


def task(args):
    shape_name, N, trial, nn, ref = args
    shape = SHAPES[shape_name]
    pts = cloud(shape_name, N, trial)
    X = np.c_[pts, np.zeros(len(pts))]
    L, M = robust_laplacian.point_cloud_laplacian(X, n_neighbors=nn)
    L, M = L.tocsc(), M.tocsc()
    lam_ref = ref["lam"]
    key = dict(shape=shape_name, N=N, trial=trial, n_neighbors=nn)
    msum = M.diagonal().sum()
    lam = low_eigs(L, M)
    out = dict(key=key, mass_sum=msum, mass_ratio=msum / shape.area,
               bias=np.mean(lam[1:] / lam_ref[1:] - 1))
    if nn != 30:
        return out

    # (2) mass rescaled to the true area: M' = M * area / sum(M)
    lam_s = low_eigs(L, M * (shape.area / msum))
    out["bias_mass_rescaled"] = np.mean(lam_s[1:] / lam_ref[1:] - 1)

    # energy check: f^T L f vs ∫|∇f|², f^T M f vs ∫f² for smooth plane waves
    Md = M.diagonal()
    rs, rm = [], []
    for (kx, ky), (int_f2, int_g2) in zip(WAVES, ref["wave_ints"]):
        f = np.cos(kx * pts[:, 0] + ky * pts[:, 1])
        rs.append(f @ (L @ f) / int_g2)
        rm.append(Md @ f ** 2 / int_f2)
    out["stiffness_ratio"], out["mass_fn_ratio"] = np.mean(rs), np.mean(rm)

    # linear precision: exact ∫|∇x|² = area; self-calibrating factor needs no reference
    lin = np.mean([pts[:, 0] @ (L @ pts[:, 0]), pts[:, 1] @ (L @ pts[:, 1])])
    c = msum / lin
    out["linear_energy_ratio"] = lin / shape.area
    out["calib_factor"] = c
    out["bias_calibrated"] = np.mean(c * lam[1:] / lam_ref[1:] - 1)
    return out


# ------------------------------------------------------------------- main --
def fmt(df):
    return df.to_string(float_format=lambda v: f"{v:.4f}")


def md(df, floatfmt="{:+.4f}"):
    df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * df.shape[1]]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v)
                                       for v in r) + " |")
    return "\n".join(lines)


def main():
    print("FEM (P2) reference eigenvalues ...", flush=True)
    with Pool(3) as p:
        refs = dict(zip(SHAPE_NAMES, p.map(fem_reference, SHAPE_NAMES)))
    for s in ("square", "disk"):
        ex = exact_eigenvalues(s, K_EIG)
        print(f"  {s}: P2-FEM vs analytic, max rel err (i=1..10) = "
              f"{np.max(np.abs(refs[s]['lam'][1:] / ex[1:] - 1)):.1e}")
        refs[s]["lam"] = ex                     # analytic where available
    for s in SHAPE_NAMES:
        print(f"  {s}: mesh area / true area = {refs[s]['area_mesh'] / SHAPES[s].area:.6f}")

    tasks = [(s, N, tr, nn, refs[s]) for N in NS[::-1] for s in SHAPE_NAMES
             for tr in range(N_TRIALS) for nn in NEIGHBORS]
    with Pool(min(36, os.cpu_count())) as p:
        res = p.map(task, tasks)
    df = pd.DataFrame([{**r.pop("key"), **r} for r in res])
    g = df[df.n_neighbors == 30].groupby(["shape", "N"])

    t1 = g[["mass_sum", "mass_ratio"]].mean()
    t1.insert(0, "true_area", [SHAPES[s].area for s, _ in t1.index])
    t2 = g[["bias", "bias_mass_rescaled"]].mean()
    t3 = g[["stiffness_ratio", "mass_fn_ratio", "linear_energy_ratio"]].mean()
    t4 = df.groupby(["shape", "N", "n_neighbors"])["bias"].mean().unstack("n_neighbors")
    t4.columns = [f"nn={c}" for c in t4.columns]
    t5 = g[["calib_factor", "bias", "bias_calibrated"]].mean()

    t1.to_csv(os.path.join(HERE, "mass_ratio.csv"))
    t2.to_csv(os.path.join(HERE, "mass_rescaled_eigs.csv"))
    t3.to_csv(os.path.join(HERE, "energy_ratio.csv"))
    t4.to_csv(os.path.join(HERE, "n_neighbors.csv"))
    t5.to_csv(os.path.join(HERE, "calibrated.csv"))
    df.to_csv(os.path.join(HERE, "all_trials.csv"), index=False)

    sections = [
        ("1. sum(diag M) vs true area (n_neighbors=30)", t1, "{:.4f}"),
        ("2. low-mode bias mean_{i=1..10}(λ_i/λ_i^ref − 1): original M vs M' = M·area/sum(M)", t2, "{:+.4f}"),
        ("2b. energy ratios for smooth test functions: fᵀLf/∫|∇f|², fᵀMf/∫f², xᵀLx/area", t3, "{:.4f}"),
        ("3. low-mode bias vs n_neighbors", t4, "{:+.4f}"),
        ("4. self-calibrated stiffness L' = c·L, c = sum(M) / mean(xᵀLx, yᵀLy)", t5, "{:+.4f}"),
    ]
    with open(os.path.join(HERE, "bias_tables.md"), "w") as fh:
        fh.write("# robust_laplacian point-cloud bias — tables\n\n"
                 f"Uniform clouds of the main experiment, mean of {N_TRIALS} trials. Reference "
                 "eigenvalues: analytic (square, disk), P2-FEM h=0.01 (square_hole).\n")
        for title, t, ff in sections:
            print(f"\n{title}\n{fmt(t)}")
            fh.write(f"\n## {title}\n\n{md(t, ff)}\n")


if __name__ == "__main__":
    main()

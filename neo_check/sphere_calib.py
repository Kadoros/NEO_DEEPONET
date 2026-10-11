"""Part 0: surface stiffness calibration on the sphere.

c = 2·sum(M) / (xᵀLx + yᵀLy + zᵀLz)   (on a surface Σ_i |∇_Γ x_i|² = 2, so the
exact energy of the three coordinate functions is 2·area)

ARPACK eigenvalues of the robust point-cloud Laplacian, before (λ) and after (c·λ),
vs the analytic l(l+1)/R², for the N=2048 and N=8000 sphere samples.

    python neo_check/sphere_calib.py      (env neo)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse.linalg as sla
from robust_laplacian import point_cloud_laplacian

HERE = Path(__file__).resolve().parent
K = 96
L_MAX = 9                                   # degrees 0..9 cover indices 0..99

rows, bands = [], []
for N, folder in [(2048, "samples"), (8000, "samples_n8000")]:
    pts = np.load(HERE / folder / "sphere/input/points.npy").astype(np.float64)
    pts = pts - pts.mean(0)
    R = np.linalg.norm(pts, axis=1).mean()
    deg = np.concatenate([[l] * (2 * l + 1) for l in range(L_MAX + 1)])[:K]
    exact = deg * (deg + 1) / R ** 2
    for nn in (20, 30):
        L, M = point_cloud_laplacian(pts, n_neighbors=nn)
        lam = np.sort(sla.eigsh(L.tocsc(), k=K, M=M.tocsc(), sigma=-1e-8)[0])
        c = 2 * M.diagonal().sum() / sum(pts[:, i] @ (L @ pts[:, i]) for i in range(3))
        for name, v in [("raw", lam), ("calibrated", c * lam)]:
            err = v[1:] / exact[1:] - 1
            rows.append(dict(N=N, n_neighbors=nn, laplacian=name, c=c, R=R,
                             mean_relerr=err.mean(), mean_abs_relerr=np.abs(err).mean(),
                             l1=err[deg[1:] == 1].mean(), l2_4=err[(deg[1:] >= 2) & (deg[1:] <= 4)].mean(),
                             l5_7=err[(deg[1:] >= 5) & (deg[1:] <= 7)].mean(),
                             l8_9=err[deg[1:] >= 8].mean()))
            for l in range(1, L_MAX + 1):
                sel = deg[1:] == l
                if sel.any():
                    bands.append(dict(N=N, n_neighbors=nn, laplacian=name, l=l, relerr=err[sel].mean()))

df = pd.DataFrame(rows)
bd = pd.DataFrame(bands).pivot_table(index=["n_neighbors", "laplacian", "N"], columns="l", values="relerr")
df.to_csv(HERE / "results/sphere_calib.csv", index=False)
bd.to_csv(HERE / "results/sphere_calib_by_degree.csv")
pd.set_option("display.width", 200)
out = (df.to_string(index=False, float_format=lambda v: f"{v:+.4f}") + "\n\nsigned rel. error by degree l\n"
       + bd.to_string(float_format=lambda v: f"{v:+.4f}"))
print(out)
(HERE / "results/sphere_calib.txt").write_text(out + "\n")

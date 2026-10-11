# data_neo (NEO volume PILOT data): manifest and regeneration

The pilot data (9.8 GB) was deleted on 2026-09-27 to free disk space, with the user's permission.
- **Kept:** the two small result tables that `NEO_PILOT.md` cites, `data_neo/index.csv` and `data_neo/crosscheck.csv`.
- **Deleted:** `train/` (10 h5 file pairs, 7.9 GB), `test_in/` (1.2 GB), `test_out/` (0.7 GB). The test folders held the h5 files and 640 `ops/*.npz` operator files.

## What it was
- **Generator:** `stage2/neo_vol_data.py` (md5 `09433203b5a19e6202bf9be8bfb65ba5`).
- **Dependencies:** `volop.py` (md5 `a35d0048d1dd964a494524a84ef2df76`) and `stage1/shapes.py` (md5 `3b9abbe5dbebc4df29e67ef830f22809`).
- **SEED = 20260926.** Each shape draws from its own stream `np.random.default_rng([SEED, split_index, family_index, idx])`. The stream is independent of process count and ordering.
- **Families:**
  - train / test_in: sh_ball, superquadric, torus, cup, shell
  - test_out: c_tube
- **Shape counts:**
  - train: 500 per family (2500 total)
  - test_in: 20 per family
  - test_out: 60
- **Samples per shape:**
  - train: 2 quasi + 2 random at N=2048
  - test: quasi / random × N=2048 / 8192
- **Targets:** 96 Delaunay-P1 eigenvectors plus the lumped mass, in the NEO pretrain_tet h5 layout. Test splits also store `ops/{family}_{idx:04d}_{kind}_{N}.npz` (stiffness K, mass, eigenvalues, points).

## Regenerate
Takes about 1 h of CPU (Pool of 20 workers). Output goes to `stage2/data_neo/`.
```bash
cd /home/cvmlserver7/Hyeon/NEO_DEEPONET && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run -n neo python stage2/neo_vol_data.py
```
- **Check:** compare the regenerated `index.csv` with the kept one. Per-sample `t_s` will differ; all other columns must match.
- **Possible nondeterminism:**
  - Point sets, operators and eigenvalues are deterministic.
  - Eigenvectors come from ARPACK (random start vector). Their sign, and the rotation inside a degenerate eigenvalue group, may differ between runs.
  - The spanned subspaces, and therefore span-loss metrics, are unaffected.

## Scripts that read this path (pilot only; none used by current work)
- `pilot_train.py`, `pilot_eval.py`, `neo_vol_crosscheck.py` (default `--data neo`), and `neo_vol_data.py` (writer).
- `pilot/*/cfg.yaml` and `hparams.yaml` (`data_dir`, recorded for history).
- Loading a pilot checkpoint does not need the data. `main_eval.py` loads `pilot/ft_lr3e-4/best.ckpt`; this was verified after the deletion.
- **Current scripts use other data:** `data_neo2` (main NEO) and `coef/b2_data` (branch B). `coef_b1.py` imports only functions from `neo_vol_data.py`, not its data.

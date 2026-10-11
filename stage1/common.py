"""Shared settings for stage1.  Importing this module first pins every cache /
temp directory used by numpy/torch/matplotlib/triton to stage1/.cache — for
this Python process only (no shell or user config is touched)."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, ".cache")
for var, sub in [("MPLCONFIGDIR", "mpl"), ("XDG_CACHE_HOME", ""), ("TRITON_CACHE_DIR", "triton"),
                 ("TORCH_HOME", "torch"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                 ("CUDA_CACHE_PATH", "nv"), ("TMPDIR", "tmp")]:
    os.environ[var] = os.path.join(CACHE, sub)
    os.makedirs(os.environ[var], exist_ok=True)

NEO_REPO = os.path.join(os.path.dirname(ROOT), "NEO")
NEO_CKPT = os.path.join(os.path.dirname(ROOT), "neo_check", "ckpts", "Ours-Large.ckpt")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

SEED = 20260925
N_SHAPES = 600                      # 200 per family
FAMILIES = ["sh_sphere", "superquadric", "torus"]
N_VERT = 4000                       # mesh vertices (= point cloud)
AREA = 4 * 3.141592653589793        # surface area after normalisation (unit-sphere area)
N_IC = 4
TAUS = [float(v) for v in __import__("numpy").geomspace(1e-3, 3.0, 8)]   # τ = t·λ1(mesh)
K_MODES = 96
PC_NEIGHBORS = 20
SPLIT = (160, 20, 20)               # per family: train / val / test  -> 480 / 60 / 60

"""Reproduce everything:  python run_all.py  [--fresh]

--fresh  deletes cache/ (FEM references) before running.
"""
import os
import shutil
import sys

# one BLAS thread per process; parallelism comes from the process pool
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")
# keep gmsh/FLTK preference files inside the project (this process only)
os.environ["HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gmsh_home")
os.makedirs(os.environ["HOME"], exist_ok=True)

import experiments  # noqa: E402

if __name__ == "__main__":
    if "--fresh" in sys.argv and os.path.isdir(experiments.CACHE):
        shutil.rmtree(experiments.CACHE)
    experiments.main()

"""C (d): elastic "model error / lump signal" on the real shapes with the A-decided basis -> e1/e1_v2_real*.csv

Runs stage3/e1_v2.py unchanged (same bases (a)(b)(c), dimension-matched exact modes, no-rotation variant,
ANP reference, loads, depth-wise lumps, costs) on the real-shape quasi-8192 operators (cleaned, with the NEO
basis), by pointing e1_basis' cache at real_shapes/e1cache/<shape>_0.npz and its family list at the shape names.

    python stage3/c_elastic.py [--nu 0.3,0.49] [--procs 10]
"""
import glob
import os
import shutil
import sys

import e1_basis as E

RS = os.path.join(E.ROOT, "real_shapes")
CACHE = os.path.join(RS, "e1cache")


def prepare():
    os.makedirs(CACHE, exist_ok=True)
    names = []
    for p in sorted(glob.glob(os.path.join(RS, "cache", "*_quasi_8192.npz"))):
        name = os.path.basename(p)[:-len("_quasi_8192.npz")]
        dst = os.path.join(CACHE, f"{name}_0.npz")
        if not os.path.exists(dst):
            shutil.copy(p, dst)
        names.append(name)
    return names


if __name__ == "__main__":
    names = prepare()
    E.CACHE = CACHE
    E.FAMILIES = names                                                 # used for the seed and the default shape list
    import e1_v2 as V
    shapes = ",".join(f"{n}:0" for n in names)
    nu = next((a.split("=")[1] for a in sys.argv if a.startswith("--nu=")), "0.3,0.49")
    procs = next((a.split("=")[1] for a in sys.argv if a.startswith("--procs=")), "10")
    rev = ["--reverse"] if "--reverse" in sys.argv else []              # 2nd instance from the end of the list
    sys.argv = ["e1_v2.py", "--nu", nu, "--shapes", shapes, "--tag", "real", "--procs", procs] + rev
    V.main()

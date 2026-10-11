"""A1 gate: does the ANP error against the Lame solution decrease with N (8k -> 16k -> 32k) for every nu?
Pass = the seed-mean relative L2 error strictly decreases 8k > 16k > 32k at nu = 0.3, 0.45 and 0.49.
Writes lame/a1_gate.txt; exit code 0 = pass, 1 = fail (the chain stops before A2)."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(ROOT, "lame", "lame.csv"))
t = df.pivot_table(index="N", columns=["op", "nu"], values="relL2", aggfunc="mean")
ratio = df.pivot_table(index="N", columns=["op", "nu"], values="inner_ur_ratio", aggfunc="mean")
ok, lines = True, ["relative L2 vs Lame (seed mean)", t.to_string(float_format=lambda v: f"{v:.4g}"),
                   "\ninner radial displacement / exact", ratio.to_string(float_format=lambda v: f"{v:.4g}"), ""]
for nu in (0.3, 0.45, 0.49):
    e = t[("anp", nu)].values
    dec = bool(np.all(np.diff(e) < 0))
    rate = np.log(e[0] / e[-1]) / np.log(t.index[-1] / t.index[0]) * 3        # order in h (h ~ N^{-1/3})
    lines.append(f"ANP nu={nu}: {' > '.join(f'{v:.4g}' for v in e)}  decreasing={dec}  observed order in h ~ {rate:.2f}")
    ok &= dec
lines.append(f"\nGATE: {'PASS' if ok else 'FAIL'}")
txt = "\n".join(lines)
open(os.path.join(ROOT, "lame", "a1_gate.txt"), "w").write(txt)
print(txt)
sys.exit(0 if ok else 1)

"""Branch B-2: evaluate trained coefficient networks on another split -> coef/b2/<run>/<split>.csv (+ slices)

    CUDA_VISIBLE_DEVICES=<gpu> python stage2/coef_b2_eval.py --runs base long400 --split test_ood
(test_ood = torus and cup shapes, families never seen in B-2 training)
"""
import argparse
import glob
import os
import time
from types import SimpleNamespace

import coef_b2 as C

import numpy as np
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="test_ood")
    ap.add_argument("--refine-iters", type=int, default=300)
    ap.add_argument("--basis", default="arpack128")
    a = ap.parse_args()
    shapes = [C.Shape(p, a.basis) for p in sorted(glob.glob(os.path.join(C.DATA, a.split, "*.npz")))]
    print(f"{len(shapes)} shapes in {a.split}", flush=True)
    for run in a.runs:
        out = os.path.join(C.OUT, run)
        net = C.CoefNet(shapes[0].x.shape[1]).to(C.DEV)
        net.load_state_dict(torch.load(os.path.join(out, "net.pt"), map_location=C.DEV))
        net.eval()
        ra = SimpleNamespace(reg=1e-4, regtype="h1")
        rows = []
        for s in shapes:
            with torch.no_grad():
                th = net(s)
            r_net, k_net = C.evaluate(s, th, ra)
            ti = time.time()
            r_ref, k_ref = C.evaluate(s, C.refine(s, th, ra, a.refine_iters), ra)
            with torch.no_grad():
                oracle = s.data_loss(torch.log(torch.tensor(s.kappa, device=C.DEV))).item()
            rows.append(dict(name=s.name, oracle_data=oracle, t_refine_s=time.time() - ti,
                             **{f"net_{k}": v for k, v in r_net.items()}, **{f"ref_{k}": v for k, v in r_ref.items()}))
            C.slice_fig(s, k_net, k_ref, os.path.join(out, f"{a.split}_slices_{s.name}.png"),
                        f"B-2 {run} [{a.split}] {s.name}: network relL2 {r_net['rel_l2']:.3f}, matched "
                        f"{r_net['n_matched']}/{len(s.centres)} (+{r_net['n_false']}) | +refine relL2 {r_ref['rel_l2']:.3f}, "
                        f"matched {r_ref['n_matched']}/{len(s.centres)} (+{r_ref['n_false']})")
        E = pd.DataFrame(rows)
        E.to_csv(os.path.join(out, f"{a.split}.csv"), index=False)
        print(f"== {run} on {a.split}: lumps {int(E.net_n_lumps.sum())}, matched net {int(E.net_n_matched.sum())} / "
              f"refine {int(E.ref_n_matched.sum())}, false net {int(E.net_n_false.sum())} / refine {int(E.ref_n_false.sum())}")
        print(E.drop(columns="name").mean().to_string(), flush=True)


if __name__ == "__main__":
    main()

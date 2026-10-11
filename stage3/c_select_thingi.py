"""C: seeded selection of 10 closed manifold Thingi10K models, stratified by genus -> real_shapes/raw/thingi10k/

Criteria (from metadata/geometry_data.csv + input_summary.csv, Thingi10K v1.5.0):
  closed (0 boundary edges), vertex- and edge-manifold, oriented, single connected component,
  no self-intersections, no degenerate or duplicated faces, 2k <= faces <= 200k, license = Creative Commons,
  not in the package's known-corrupt list.
  genus g = (2 - euler) / 2;  strata: g = 0 (4 models), g = 1 (3), g >= 2 (3; g <= 10).
  Random draw inside each stratum with numpy default_rng(SEED).
Downloads only the chosen npz files (https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/v1.5.0/npz/<id>.npz).

    python stage3/c_select_thingi.py
"""
import os
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "real_shapes", "raw", "thingi10k")
SEED = 20260927
STRATA = [("g0", 0, 0, 4), ("g1", 1, 1, 3), ("g2+", 2, 10, 3)]
URL = "https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/v1.5.0/npz/{}.npz"
CORRUPT = {49911, 74463, 286163, 77942}


def main():
    g = pd.read_csv(os.path.join(D, "geometry_data.csv"))
    s = pd.read_csv(os.path.join(D, "input_summary.csv")).rename(columns={"ID": "file_id"})
    df = g.merge(s[["file_id", "Thing ID", "License", "Link"]], on="file_id")
    ok = ((df.num_boundary_edges == 0) & (df.vertex_manifold == 1) & (df.edge_manifold == 1) & (df.oriented == 1)
          & (df.num_connected_components == 1) & (df.num_self_intersections == 0)
          & (df.num_geometrical_degenerated_faces == 0) & (df.num_combinatorial_degenerated_faces == 0)
          & (df.num_duplicated_faces == 0) & df.num_faces.between(2000, 200000)
          & df.License.str.contains("Creative Commons", na=False) & ~df.file_id.isin(CORRUPT))
    df = df[ok].copy()
    df["genus"] = (2 - df.euler_characteristic) // 2
    rng = np.random.default_rng(SEED)
    picks = []
    for name, lo, hi, n in STRATA:
        pool = df[df.genus.between(lo, hi)].sort_values("file_id")
        idx = rng.choice(len(pool), n, replace=False)
        sel = pool.iloc[np.sort(idx)].assign(stratum=name, pool_size=len(pool))
        picks.append(sel)
    P = pd.concat(picks)
    for fid in P.file_id:
        path = os.path.join(D, f"{fid}.npz")
        if not os.path.exists(path):
            urllib.request.urlretrieve(URL.format(fid), path)
        z = np.load(path)
        print(fid, list(z.keys()), {k: z[k].shape for k in z.keys()})
    cols = ["stratum", "file_id", "Thing ID", "genus", "num_vertices", "num_faces", "License", "Link", "pool_size"]
    P[cols].to_csv(os.path.join(D, "selected.csv"), index=False)
    print(f"eligible models: {len(df)}  (g0 {int((df.genus == 0).sum())}, g1 {int((df.genus == 1).sum())}, "
          f"g2-10 {int(df.genus.between(2, 10).sum())})")
    print(P[cols[:-2]].to_string(index=False))


if __name__ == "__main__":
    main()

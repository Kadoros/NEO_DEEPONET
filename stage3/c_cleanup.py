"""C: remove face-disconnected tet pieces from the real-shape operators (in place, real_shapes/cache/*.npz)

On thin features (e.g. the cow's legs/ears) the point-based Delaunay + SDF carving can leave a few tets attached to
the main body only through an edge or a vertex.  For the scalar Laplacian this is harmless (nodes are shared), but
for elasticity every such piece is a hinge: an extra zero-strain-energy mechanism (seen as a 7th near-zero Korn
eigenvalue).  Fix: keep the largest face-connected component of tets, drop the nodes that no longer belong to any
tet (a handful), re-assemble the P1 fields, recompute the scalar ARPACK eigenpairs; the NEO basis (Pn) is removed
so that c_eval_a.py recomputes it on the cleaned point cloud.  Records n_pieces_removed / n_nodes_removed.

    python stage3/c_cleanup.py [--procs 6]
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.sparse.csgraph import connected_components

import e1_basis as E
from volop import p1_elements

RS = os.path.join(E.ROOT, "real_shapes")


def face_components(T):
    F = np.concatenate([np.sort(T[:, idx], 1) for idx in ([1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2])])
    tid = np.tile(np.arange(len(T)), 4)
    _, inv = np.unique(F, axis=0, return_inverse=True)
    inv = inv.ravel()
    o = np.argsort(inv, kind="stable")
    a, b = o[:-1], o[1:]
    same = inv[a] == inv[b]
    G = sp.coo_matrix((np.ones(same.sum()), (tid[a[same]], tid[b[same]])), shape=(len(T), len(T)))
    return connected_components(G, directed=False)


def clean(path):
    z = dict(np.load(path))
    if "n_pieces_removed" in z:
        return os.path.basename(path), "already clean"
    n, lab = face_components(z["tets"])
    if n == 1 and len(np.unique(z["tets"])) == len(z["Y"]):
        z.update(n_pieces_removed=0, n_nodes_removed=0)
        np.savez(path, **z)
        return os.path.basename(path), "ok (1 component)"
    main = np.argmax(np.bincount(lab))
    T = z["tets"][lab == main]
    used = np.unique(T)
    remap = -np.ones(len(z["Y"]), np.int64)
    remap[used] = np.arange(len(used))
    Y = z["Y"][used]
    T, G, vol = p1_elements(Y, remap[T])
    m = np.zeros(len(Y))
    np.add.at(m, T.ravel(), np.repeat(vol / 4, 4))
    Kl = vol[:, None, None] * np.einsum("tki,tkj->tij", G, G)
    K = sp.coo_matrix((Kl.ravel(), (np.repeat(T, 4, 1).ravel(), np.tile(T, (1, 4)).ravel())), shape=(len(Y),) * 2).tocsr()
    ls, Ps = sla.eigsh(K.tocsc(), k=256, M=sp.diags(m).tocsc(), sigma=-1e-6)
    o = np.argsort(ls)
    z.update(Y=Y, tets=T, grads=G, vol=vol, m=m, Ps=Ps[:, o], lam_s=ls[o], n_pieces_removed=n - 1,
             n_nodes_removed=len(z["m"]) - len(Y))
    z.pop("Pn", None); z.pop("t_neo", None)
    np.savez(path, **z)
    return os.path.basename(path), f"removed {n - 1} pieces ({int((lab != main).sum())} tets), {len(remap) - len(used)} nodes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(RS, "cache", "*.npz")))
    with Pool(a.procs) as pool:
        for name, msg in pool.imap_unordered(clean, files):
            print(name, msg, flush=True)


if __name__ == "__main__":
    main()

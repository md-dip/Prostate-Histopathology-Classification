"""
Build the SLIC superpixel graph cache for the hybrid model (Section 3.3.3).
============================================================================

The GCN branch of the proposed hybrid consumes one region-adjacency graph per
patch. Building those graphs on the fly costs more than the forward pass, so
they are pre-computed once into a pickle keyed by absolute image path:

    { "<abs path to patch>": {"x": (n, 8) float32,
                              "edge_index": (2, e) int64}, ... }

IMPORTANT - read before using this script
------------------------------------------
The published models were trained against cache/graph_cache.pkl, which is
committed to this repository and is the authoritative artefact. The notebook
that originally generated it was not preserved, so the recipe below is a
RECONSTRUCTION recovered from the cache itself and from the external-validation
inference code. It reproduces the schema, the edge rule and the feature layout,
but it does NOT reproduce the original file bit for bit: SLIC seed placement
and connectivity enforcement changed between scikit-image releases, and the
version used in mid-2025 was not recorded. On scikit-image 0.26 this builder
matches the stored node count on roughly a third of patches and lands within
a few superpixels on the rest.

Consequences:
  * To reproduce the published numbers, use the committed cache. Every result
    in the manuscript was produced against it.
  * Use this script to build graphs for NEW data (a different cohort, an
    external validation set), where matching the old cache is not the point.

Reconstructed recipe:
  * SLIC over the RGB patch at its native 256x256 resolution - the graph is
    built BEFORE the 224x224 resize that feeds the CNN branch
  * n_segments=75, compactness=10, sigma=1, start_label=0
  * one node per superpixel
  * 8-dim node feature: [R_mean, G_mean, B_mean, R_std, G_std, B_std,
                         cy_norm, cx_norm] - the "channel-wise statistics and
                         two colour differences" of Section 3.3.3
  * edges: k-nearest-neighbour (k=5) in centroid space, then SYMMETRISED
    (A = A + A^T, binarised), no self-loops - GCNConv adds them internally
    as A~ = A + I. Symmetrisation gives the ~5.8 edges per node observed in
    the stored cache.

Run:
    # ONCE AFTER CLONING - repoint the shipped cache at your dataset/ copy
    python preprocessing/build_slic_graph_cache.py --rekey cache/graph_cache.pkl

    # compare this builder against the committed cache
    python preprocessing/build_slic_graph_cache.py --verify cache/graph_cache.pkl

    # build a fresh cache (only needed for new data)
    python preprocessing/build_slic_graph_cache.py --out cache/my_new_cache.pkl

Time: ~20-30 min for the 6,002-patch subset on a single CPU core.
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.segmentation import slic
from sklearn.neighbors import kneighbors_graph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


# ─────────────────────────────────────────────────────────────────────
def build_graph_from_image(img_arr, n_superpix=C.N_SUPERPIXELS, k=C.KNN_K):
    """img_arr: HxWx3 uint8 or float in [0,1]. Returns dict {x, edge_index}."""
    if img_arr.dtype != np.float64 and img_arr.dtype != np.float32:
        img_arr = img_arr.astype(np.float32) / 255.0
    H, W = img_arr.shape[:2]

    segs = slic(img_arr, n_segments=n_superpix, compactness=10, sigma=1,
                start_label=0, channel_axis=-1)
    seg_ids = np.unique(segs)
    n = len(seg_ids)

    feats = np.zeros((n, 8), dtype=np.float32)
    centroids = np.zeros((n, 2), dtype=np.float32)
    for i, sid in enumerate(seg_ids):
        mask = segs == sid
        if not mask.any():
            continue
        pix = img_arr[mask]
        feats[i, 0:3] = pix.mean(axis=0)
        feats[i, 3:6] = pix.std(axis=0)
        ys, xs = np.where(mask)
        cy, cx = ys.mean() / H, xs.mean() / W
        feats[i, 6] = cy
        feats[i, 7] = cx
        centroids[i] = [cy, cx]

    k_use = min(k, max(1, n - 1))
    A = kneighbors_graph(centroids, n_neighbors=k_use, mode="connectivity",
                         include_self=False)
    A = ((A + A.T) > 0).astype(np.int8)   # symmetrise: edges are undirected
    coo = A.tocoo()
    edge_index = np.vstack([coo.row, coo.col]).astype(np.int64)
    return {"x": feats, "edge_index": edge_index}


def load_patch(path):
    """Load a patch at native resolution. The graph is built before the 224x224
    resize that feeds the CNN branch, so no resampling happens here."""
    return np.array(Image.open(path).convert("RGB"))


def list_patches():
    files = []
    for cls in ("Benign", "Malignant"):
        d = C.DATA_DIR / cls
        if not d.exists():
            sys.exit(f"Missing class folder: {d}\n"
                     f"Set PROSTATE_DATA_DIR or place the subset under dataset/.")
        files += sorted(p for p in d.iterdir() if p.suffix.lower() in C.IMG_EXTS)
    return files


# ─────────────────────────────────────────────────────────────────────
def verify(reference_cache, n_samples=25):
    """Rebuild n random graphs and check they match an existing cache."""
    with open(reference_cache, "rb") as f:
        ref = pickle.load(f)
    print(f"Reference cache: {reference_cache}\n  {len(ref):,} graphs")

    keys = sorted(ref.keys())
    rng = np.random.default_rng(C.SEED)
    picks = rng.choice(len(keys), size=min(n_samples, len(keys)), replace=False)

    ok = miss = 0
    for idx in picks:
        key = keys[idx]
        local = C.DATA_DIR / Path(key).parent.name / Path(key).name
        if not local.exists():
            miss += 1
            continue
        got = build_graph_from_image(load_patch(local))
        exp = ref[key]
        x_ok = (got["x"].shape == np.asarray(exp["x"]).shape and
                np.allclose(got["x"], np.asarray(exp["x"]), atol=1e-5))
        e_ok = (got["edge_index"].shape == np.asarray(exp["edge_index"]).shape and
                np.array_equal(got["edge_index"], np.asarray(exp["edge_index"])))
        status = "OK  " if (x_ok and e_ok) else "DIFF"
        if x_ok and e_ok:
            ok += 1
        else:
            print(f"  {status} {Path(key).name}  "
                  f"x{got['x'].shape}vs{np.asarray(exp['x']).shape}  "
                  f"e{got['edge_index'].shape}vs{np.asarray(exp['edge_index']).shape}")
    checked = len(picks) - miss
    print(f"\n  matched {ok}/{checked} sampled graphs"
          + (f"   ({miss} not found locally)" if miss else ""))
    return ok == checked and checked > 0


def rekey(cache_path):
    r"""Rewrite cache keys to point at the current DATA_DIR.

    The cache is keyed by absolute image path, and it was built on the original
    author's machine (F:\Prostate Data\dataset\...). After a clone those keys
    match nothing, so every lookup silently misses and the GCN branch receives
    empty graphs. Run this once after cloning:

        python preprocessing/build_slic_graph_cache.py --rekey cache/graph_cache.pkl
    """
    cache_path = Path(cache_path)
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    remapped, already, lost = {}, 0, 0
    for k, v in cache.items():
        kp = Path(k)
        new = C.DATA_DIR / kp.parent.name / kp.name
        if str(new) == k:
            already += 1
        if new.exists():
            remapped[str(new)] = v
        else:
            lost += 1
    if lost:
        print(f"  WARNING: {lost:,} keys have no matching file under {C.DATA_DIR}")
    if already == len(cache):
        print(f"  Already keyed to {C.DATA_DIR} - nothing to do.")
        return True

    backup = cache_path.with_suffix(".pkl.bak")
    if not backup.exists():
        cache_path.replace(backup)
        print(f"  Original saved as {backup.name}")
    with open(cache_path, "wb") as f:
        pickle.dump(remapped, f)
    print(f"  Rekeyed {len(remapped):,}/{len(cache):,} graphs -> {C.DATA_DIR}")
    return lost == 0


def build(out=None):
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(out) if out else C.GRAPH_CACHE_HYBRID
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"Cache already exists: {out}\nDelete it to rebuild.")
        return

    files = list_patches()
    print(f"Building SLIC graphs for {len(files):,} patches "
          f"(n_segments={C.N_SUPERPIXELS}, k={C.KNN_K}) ...")

    cache, t0 = {}, time.time()
    for i, p in enumerate(files, 1):
        cache[str(p)] = build_graph_from_image(load_patch(p))
        if i % 250 == 0 or i == len(files):
            el = time.time() - t0
            print(f"  {i:5d}/{len(files)}  {el/60:5.1f} min elapsed  "
                  f"~{el/i*(len(files)-i)/60:5.1f} min left")

    with open(out, "wb") as f:
        pickle.dump(cache, f)
    print(f"\nSaved {len(cache):,} graphs -> {out}  "
          f"({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", metavar="CACHE.pkl",
                    help="check the builder reproduces an existing cache instead of building")
    ap.add_argument("--rekey", metavar="CACHE.pkl", default=None,
                    help="repoint an existing cache at the local dataset (run once after cloning)")
    ap.add_argument("--out", metavar="CACHE.pkl", default=None,
                    help="where to write the new cache (default cache/graph_cache.pkl)")
    ap.add_argument("--samples", type=int, default=25,
                    help="graphs to sample when verifying (default 25)")
    a = ap.parse_args()
    if a.rekey:
        sys.exit(0 if rekey(a.rekey) else 1)
    if a.verify:
        sys.exit(0 if verify(a.verify, a.samples) else 1)
    build(a.out)

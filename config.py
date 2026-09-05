r"""
Central path and hyperparameter configuration.

Every script and notebook in this repository resolves its paths through this
module, so the code runs from a clean clone with no editing. The original
research code used absolute Windows paths (``F:\Prostate Data\...``); those
have been replaced by the constants below.

Override the dataset location with the PROSTATE_DATA_DIR environment variable
if you keep the patches outside the repository:

    set PROSTATE_DATA_DIR=D:\my\diagset\subset      (Windows)
    export PROSTATE_DATA_DIR=/data/diagset_subset      (Linux/macOS)
"""

import os
from pathlib import Path

# ── Locations ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent

# The 6,002-patch class-balanced DiagSet subset, committed under dataset/.
# Expected layout:  dataset/Benign/*.jpg  and  dataset/Malignant/*.jpg
DATA_DIR = Path(os.environ.get("PROSTATE_DATA_DIR", REPO_ROOT / "dataset"))

RESULTS_DIR   = REPO_ROOT / "results"
HYBRID_DIR    = RESULTS_DIR / "hybrid"
BASELINES_DIR = RESULTS_DIR / "baselines"
ABLATION_DIR  = RESULTS_DIR / "ablation"
FIGURES_DIR   = REPO_ROOT / "figures"

# Derived artefacts that are NOT tracked in git (see .gitignore):
#   *.pkl  SLIC / k-NN graph caches   -> rebuild with preprocessing/
#   *.pth  trained model weights      -> available on request (see README)
CACHE_DIR      = Path(os.environ.get("PROSTATE_CACHE_DIR", REPO_ROOT / "cache"))
CHECKPOINT_DIR = Path(os.environ.get("PROSTATE_CKPT_DIR",  REPO_ROOT / "checkpoints"))

GRAPH_CACHE_HYBRID = CACHE_DIR / "graph_cache.pkl"          # SLIC superpixel graphs
GRAPH_CACHE_LGCN   = CACHE_DIR / "lgcn_graph_cache.pkl"     # 7x7 grid, k-NN k=6
GRAPH_CACHE_GAT    = CACHE_DIR / "gat_graph_cache.pkl"      # 7x7 grid, k-NN k=6
GRAPH_CACHE_JGCN   = CACHE_DIR / "jgcn_graph_cache.pkl"     # 7x7 grid, joint edge rule

# ── Protocol constants (Section 3.4-3.5 of the manuscript) ───────────
SEED        = 42
IMG_SIZE    = 224
BATCH_SIZE  = 32
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}

# SLIC superpixel graph (Section 3.3.3)
N_SUPERPIXELS = 75
KNN_K         = 5

# Training (Table 2)
NUM_EPOCHS      = 80
EARLY_STOP_PATIENCE = 15
LR_BACKBONE     = 1e-4
LR_HEAD         = 3e-4
WEIGHT_DECAY    = 1e-4
FOCAL_GAMMA     = 2.0
FOCAL_ALPHA     = 0.75
HIDDEN_DIM      = 256
GAT_HEADS       = 4
GCN_LAYERS      = 3
DROPOUT         = 0.4
TTA_N           = 5

# Evaluation (Section 3.6)
N_BOOTSTRAP  = 1000
ECE_BINS     = 10
BONFERRONI_ALPHA = 0.0125   # 0.05 / 4 hybrid-vs-baseline comparisons


def ensure_dirs():
    """Create the output directories that scripts write into."""
    for d in (RESULTS_DIR, HYBRID_DIR, BASELINES_DIR, ABLATION_DIR,
              FIGURES_DIR, CACHE_DIR, CHECKPOINT_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"REPO_ROOT : {REPO_ROOT}")
    print(f"DATA_DIR  : {DATA_DIR}   exists={DATA_DIR.exists()}")
    if DATA_DIR.exists():
        for cls in ("Benign", "Malignant"):
            p = DATA_DIR / cls
            n = len([f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS]) if p.exists() else 0
            print(f"  {cls:10s}: {n:,} patches")

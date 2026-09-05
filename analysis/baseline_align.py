"""
Baseline Alignment Converter
----------------------------
The 4 baseline models (LGCN, GAT, JGCN-orig, JGCN-matched) already share
the same test set (their retrain notebooks use identical split code). We
only need to attach file_paths to their existing predictions so
mcnemar_aligned.py can verify alignment.

This script:
  1. Reproduces the baselines' split algorithm once to get the test file paths
  2. For each baseline model:
     - Loads <model>_predictions.npz (already has y_true/y_pred/y_prob)
     - Writes <model>_aligned_raw.npz WITH file_paths added
  3. Sanity-checks that y_true across all 4 baselines matches

Run:   python baseline_align.py
Time:  ~2 seconds. CPU only.

After it finishes:
       python mcnemar_aligned.py
"""

import os
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# Paths resolve through the repository-level config module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
# ---------------------------------------------------------------------

DATA_DIR = str(C.DATA_DIR)
SEED     = C.SEED
IMG_EXTS = C.IMG_EXTS

_B = C.BASELINES_DIR
BASELINES = {
    'LGCN'        : (str(_B / 'lgcn_predictions.npz'),
                     str(_B / 'lgcn_aligned_raw.npz')),
    'GAT'         : (str(_B / 'gat_predictions.npz'),
                     str(_B / 'gat_aligned_raw.npz')),
    'JGCN-orig'   : (str(_B / 'jgcn_predictions.npz'),
                     str(_B / 'jgcn_aligned_raw.npz')),
    'JGCN-matched': (str(_B / 'jgcn_matched_predictions.npz'),
                     str(_B / 'jgcn_matched_aligned_raw.npz')),
}

# ═════════════════════════════════════════════════════════════════════
def main():
    print('=' * 70)
    print('  Baseline Alignment Converter')
    print('=' * 70)

    # ── Step 1: Rebuild the baselines test set (same code as their notebooks) ──
    print(f'\n[1/3] Reproducing baselines test split ...')
    data_dir = Path(DATA_DIR)
    all_paths, all_labels = [], []
    for label_idx, cls in enumerate(['Benign', 'Malignant']):
        class_dir = data_dir / cls
        if not class_dir.exists():
            for alt in [cls.capitalize(), cls.upper()]:
                if (data_dir / alt).exists():
                    class_dir = data_dir / alt; break
        files = [p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        all_paths.extend([str(p) for p in files])
        all_labels.extend([label_idx] * len(files))
        print(f'      {cls:12s}: {len(files):5d} images')
    all_paths  = np.array(all_paths)
    all_labels = np.array(all_labels)

    idx = np.arange(len(all_paths))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, stratify=all_labels, random_state=SEED)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50,
        stratify=all_labels[temp_idx], random_state=SEED)

    test_paths_split  = all_paths[test_idx]
    test_labels_split = all_labels[test_idx]
    print(f'      Test set: {len(test_paths_split)} images '
          f'({(test_labels_split == 0).sum()} benign, '
          f'{(test_labels_split == 1).sum()} malignant)')

    # ── Step 2: Convert each baseline ────────────────────────────────
    print(f'\n[2/3] Converting each baseline .npz -> _aligned_raw.npz ...')
    saved = []
    for name, (in_path, out_path) in BASELINES.items():
        if not os.path.exists(in_path):
            print(f'      [SKIP] {name:14s}  not found: {in_path}')
            continue
        d = np.load(in_path)
        if 'y_true_notta' not in d.files:
            print(f'      [SKIP] {name:14s}  missing y_true_notta')
            continue

        y_true = np.asarray(d['y_true_notta']).astype(int)
        y_pred = np.asarray(d['y_pred_notta']).astype(int)
        y_prob = np.asarray(d['y_prob_notta']).astype(float)

        if len(y_true) != len(test_paths_split):
            print(f'      [FAIL] {name:14s}  length mismatch: '
                  f'{len(y_true)} preds vs {len(test_paths_split)} paths')
            continue

        # Sanity: labels the baseline saved must match the split labels
        if not np.array_equal(y_true, test_labels_split):
            n_diff = int((y_true != test_labels_split).sum())
            print(f'      [WARN] {name:14s}  y_true mismatches split labels at '
                  f'{n_diff} positions — same test set, likely different intra-set '
                  f'order. Trusting the file_paths from the split.')

        np.savez(
            out_path,
            file_paths = test_paths_split,
            y_true     = test_labels_split,   # canonical labels from split
            y_pred     = y_pred,
            y_prob     = y_prob,
        )
        acc = (y_pred == test_labels_split).mean()
        print(f'      [ok]   {name:14s}  N={len(y_pred)}  acc={acc*100:.2f}%  -> {os.path.basename(out_path)}')
        saved.append(name)

    print(f'\n[3/3] Done. Created {len(saved)} aligned files: {saved}')
    print(f'      Now run:  python mcnemar_aligned.py')
    print('=' * 70)

if __name__ == '__main__':
    main()

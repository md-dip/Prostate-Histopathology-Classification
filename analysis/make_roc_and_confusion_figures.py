"""
make_roc_and_confusion_figures.py
Generates Figure 7 (ROC overlay, 4 models) and Figure 8 (confusion-matrix
grid, 2x2) of the manuscript, plus the exact metrics reported in Table 3.

Reads *_predictions.npz from results/ and uses the single-view (non-TTA)
predictions at threshold 0.5.

Run:  conda activate abide  &&  python make_roc_and_confusion_figures.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

OUT = str(C.FIGURES_DIR)
os.makedirs(OUT, exist_ok=True)

# Map: display name -> path to its predictions archive
MODELS = {
    "Proposed Hybrid":       str(C.HYBRID_DIR    / "hybrid_predictions.npz"),
    "Linear GCN":            str(C.BASELINES_DIR / "lgcn_predictions.npz"),
    "Standalone GAT":        str(C.BASELINES_DIR / "gat_predictions.npz"),
    "JGCN (matched freeze)": str(C.BASELINES_DIR / "jgcn_matched_predictions.npz"),
}

# Consistent colours across both figures
COLOURS = {
    "Proposed Hybrid":       "#1f77b4",  # blue
    "Linear GCN":            "#2ca02c",  # green
    "Standalone GAT":        "#ff7f0e",  # orange
    "JGCN (matched freeze)": "#d62728",  # red
}

# ------------------------------------------------------------------
# Load predictions
# ------------------------------------------------------------------
data = {}
for name, path in MODELS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    npz = np.load(path, allow_pickle=True)
    # Standard keys agreed in Cell 6B artefact protocol
    y_true = npz["y_true_notta"].astype(int)
    y_prob = npz["y_prob_notta"].astype(float)
    y_pred = npz["y_pred_notta"].astype(int)
    data[name] = {"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred}
    acc = (y_pred == y_true).mean() * 100
    print(f"  {name:25s}  N={len(y_true)}  Acc={acc:.2f}%")

# ==================================================================
# FIGURE 8 — ROC overlay
# ==================================================================
fig, ax = plt.subplots(figsize=(8, 7))

for name, d in data.items():
    fpr, tpr, _ = roc_curve(d["y_true"], d["y_prob"])
    auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr,
            color=COLOURS[name],
            linewidth=2.2,
            label=f"{name} (AUC = {auc_val:.4f})")

# Reference diagonal
ax.plot([0, 1], [0, 1],
        color="black", linestyle="--", linewidth=1.2,
        label="Random classifier (AUC = 0.50)")

ax.set_xlim([-0.005, 1.0])
ax.set_ylim([0.0, 1.005])
ax.set_xlabel("False Positive Rate  (1 − Specificity)", fontsize=13)
ax.set_ylabel("True Positive Rate  (Sensitivity)",       fontsize=13)
ax.set_title("Figure 7. ROC curves on the DiagSet test set (N = 901 patches)\n"
             "Non-TTA predictions at threshold 0.5",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
roc_path = os.path.join(OUT, "Figure7_ROC_overlay.png")
plt.savefig(roc_path, dpi=600, bbox_inches="tight")
plt.close()
print(f"\nSaved: {roc_path}")

# ==================================================================
# FIGURE 9 — Confusion matrix grid (2 x 2)
# ==================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
axes = axes.flatten()

for ax, (name, d) in zip(axes, data.items()):
    cm = confusion_matrix(d["y_true"], d["y_pred"], labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc  = (tn + tp) / cm.sum() * 100
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    im = ax.imshow(cm, cmap="Blues", vmin=0)

    # Cell text
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            # White text on dark cells, dark text on light cells
            colour = "white" if val > cm.max() * 0.55 else "black"
            ax.text(j, i, str(val),
                    ha="center", va="center",
                    fontsize=20, fontweight="bold",
                    color=colour)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Malignant"], fontsize=11)
    ax.set_yticklabels(["Benign", "Malignant"], fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label",     fontsize=12)
    ax.set_title(f"{name}\nAcc = {acc:.2f}%   Sens = {sens:.2f}%   Spec = {spec:.2f}%",
                 fontsize=12, fontweight="bold", pad=8)

    # Colourbar per subplot, small
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Figure 8. Confusion matrices on the DiagSet test set (N = 901 patches)\n"
             "Non-TTA predictions at threshold 0.5",
             fontsize=14, fontweight="bold", y=1.00)

plt.tight_layout()
cm_path = os.path.join(OUT, "Figure8_ConfusionMatrix_grid.png")
plt.savefig(cm_path, dpi=600, bbox_inches="tight")
plt.close()
print(f"Saved: {cm_path}")

# ==================================================================
# Print exact metrics for §3.1 Table 4
# ==================================================================
print("\n" + "="*72)
print("  METRICS FOR TABLE 3")
print("="*72)
print(f"  {'Model':<25s}  {'Acc%':>7s}  {'AUC':>7s}  {'Sens%':>7s}  {'Spec%':>7s}  {'MCC':>7s}")
print("  " + "-"*70)
for name, d in data.items():
    fpr, tpr, _ = roc_curve(d["y_true"], d["y_prob"])
    auc_val = auc(fpr, tpr)
    tn, fp, fn, tp = confusion_matrix(d["y_true"], d["y_pred"], labels=[0,1]).ravel()
    acc  = (tn + tp) / (tn + tp + fp + fn) * 100
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100
    # MCC
    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc  = (tp*tn - fp*fn) / denom if denom > 0 else 0.0
    print(f"  {name:<25s}  {acc:7.2f}  {auc_val:7.4f}  {sens:7.2f}  {spec:7.2f}  {mcc:7.4f}")
print("="*72)

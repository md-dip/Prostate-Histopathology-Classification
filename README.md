# Hybrid CNN–GNN for Prostate Histopathology Classification

Code and result artefacts for **"Investigating Graph Construction Strategies and Adaptive Fusion in Hybrid CNN–Graph Neural Network Models for Prostate Histopathology Classification."**

A three-branch model for binary Benign/Malignant classification of prostate H&E patches: a partially fine-tuned EfficientNet-B4, a two-layer graph attention network on a batch-level graph, and a three-layer graph convolutional network on SLIC superpixel graphs. The three branch outputs are concatenated into an 832-dimensional vector and reweighted per sample by a Squeeze-and-Excitation gate.

On the 901-patch DiagSet test set the hybrid reaches **98.34%** accuracy (95% CI 97.27–99.07), AUC 0.9970, sensitivity 98.39%, specificity 98.28%, MCC 0.9667.

---

## Reproducing the published numbers

Every table and figure in the paper regenerates from the artefacts committed here. No GPU and no retraining required — the saved predictions are enough.

```bash
git clone https://github.com/md-dip/Prostate-Histopathology-Classification
cd Prostate-Histopathology-Classification

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python config.py                                   # confirms 3,000 + 3,002 patches
python analysis/make_roc_and_confusion_figures.py   # Table 3, Figures 7 and 8
python analysis/mcnemar_aligned.py                  # Table 4
python analysis/ece_reliability.py                  # Table 5, Figure 9
```

Each script prints its table to stdout and writes the figures to `figures/`. All three run on CPU in seconds.

| Paper item | Produced by | Reads |
|---|---|---|
| Table 3, Figures 7–8 | `analysis/make_roc_and_confusion_figures.py` | `results/hybrid/`, `results/baselines/` |
| Table 4 | `analysis/mcnemar_aligned.py` | `*_aligned_raw.npz` |
| Table 5, Figure 9 | `analysis/ece_reliability.py` | `*_predictions.npz` |
| Table 6, Figure 10 | `notebooks/ablation/V5_se_gated_and_summary.ipynb` | `results/ablation/` |

## Retraining from scratch

```bash
# once, to repoint the shipped graph cache at your copy of dataset/
python preprocessing/build_slic_graph_cache.py --rekey cache/graph_cache.pkl

jupyter notebook notebooks/01_hybrid_proposed.ipynb
```

Roughly 6 hours per model on an RTX 3060 (12 GB) at the 80-epoch budget with patience-15 early stopping. The four baselines build their own graph caches on first run.

## Layout

```
config.py                  every path and hyperparameter; no hardcoded paths anywhere
dataset/                   the 6,002-patch DiagSet subset (3,000 B / 3,002 M)
cache/graph_cache.pkl      pre-built SLIC superpixel graphs — see the caveat below
notebooks/
  01_hybrid_proposed.ipynb        the proposed three-branch model
  baselines/                      LGCN, standalone GAT, JGCN (original + matched freeze)
  ablation/                       V1–V5, the five-variant ablation of Table 6
analysis/
  make_roc_and_confusion_figures.py
  mcnemar_aligned.py              Table 4 — the test reported in the paper
  ece_reliability.py              Table 5 — ECE and temperature scaling
  hybrid_realign.py               re-infers the hybrid on the baselines' test split
  baseline_align.py               attaches file paths to baseline predictions
  mcnemar_pairwise.py             superseded — see "Notes" below
preprocessing/
  build_slic_graph_cache.py       SLIC graph construction (Section 3.3.3)
results/                   predictions, training histories, environment stamps, reports
figures/                   regenerated figures
```

Notebooks are committed **with their outputs intact**, so the training curves and per-epoch logs behind the reported runs are readable without re-executing anything.

## Notes and caveats

**The graph cache is the authoritative artefact.** The hybrid's GCN branch reads pre-built SLIC graphs from `cache/graph_cache.pkl`, committed here because the published models were trained against it. The notebook that originally generated it was not preserved. `preprocessing/build_slic_graph_cache.py` documents the recipe recovered from the cache itself (SLIC at native 256×256, `n_segments=75`, `compactness=10`, `sigma=1`; 8-dim node features; symmetrised k-NN with k=5), and it reproduces the schema and edge rule — but not the file bit for bit, because SLIC seed placement changed between scikit-image releases and the version used was not recorded. Use the committed cache to reproduce the paper; use the script to build graphs for new data. `--verify` reports how closely the two agree on your scikit-image version.

**Cache keys need remapping after cloning.** The cache is keyed by image path. Run `--rekey` once (above) or every lookup silently misses and the GCN branch receives empty graphs.

**Two V5 prediction files, deliberately.** `results/ablation/v5_predictions_original_partition.npz` is the 98.34% headline of Table 3; `v5_predictions_aligned_partition.npz` is the 99.33% of Table 6. Same trained model, two different 15% test samples — the re-aligned partition exists to keep the ablation and the retrained baselines byte-comparable. Section 4.4 of the paper explains the 0.99 pp gap.

**`mcnemar_pairwise.py` is superseded.** It compares predictions positionally; `mcnemar_aligned.py` sorts by file path first and is what Table 4 reports. Kept for completeness — do not use it for new comparisons.

**Trained weights are not in the repository.** Nineteen checkpoints totalling ~2.2 GB, each above GitHub's 100 MB per-file limit. Available from the corresponding author on request.

**Evaluation is image-level, not patient-level.** The DiagSet subset carries no patient identifiers, so patches from one scan can fall in both training and test. The reported accuracies are upper bounds relative to a strict patient-wise split. This is Section 5.3 of the paper and it bounds every number in this repository.

**One small discrepancy.** `results/ablation/ablation_report.txt` was generated from an earlier bootstrap draw than the manuscript: it gives the V5 interval as [98.78, 99.78] and the V4→V5 gain as +3.00 pp, where Table 6 reports [98.67, 99.89] and +2.99 pp. Point estimates, AUC and MCC are identical; only the resampled interval differs.

## Environment

Python 3.10.20, PyTorch 2.5.1, CUDA 12.1, PyTorch Geometric 2.8.0, on one NVIDIA RTX 3060 (12 GB) under Windows. Seed 42 with `cudnn.deterministic=True` throughout. Each run's exact environment is stamped in the matching `results/**/[model]_env.json`.

## Data

`dataset/` holds a class-balanced binary subset of **DiagSet** (Koziarski et al. 2024), obtained pre-assembled from a prior research effort; the upstream patch selection, magnification filtering and class balancing are not documented within the scope of this study. DiagSet was released fully anonymised on already-scanned material. See `LICENSE` for the distinction between the MIT-licensed code and the dataset's own terms.

```bibtex
@article{koziarski2024diagset,
  title   = {DiagSet: a dataset for prostate cancer histopathological image classification},
  author  = {Koziarski, Micha{\l} and Cyganek, Bogus{\l}aw and Niedziela, Przemys{\l}aw and others},
  journal = {Scientific Reports},
  volume  = {14}, pages = {6780}, year = {2024},
  doi     = {10.1038/s41598-024-52183-4}
}
```

## Intended use

Pathologist-supervised screening support, not autonomous diagnosis. The model is trained and validated on one dataset with an image-level split and no external cohort; generalisation across scanners, stains and laboratories is not established here.

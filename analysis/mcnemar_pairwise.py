"""
E3 - Pairwise McNemar's Test (Hybrid vs each baseline)
------------------------------------------------------
Reads .npz files saved by each retrain notebook (Cell 6B / 7B),
verifies all five models were evaluated on the SAME test set,
and runs McNemar's exact test on the paired non-TTA predictions.

Run:  python mcnemar_pairwise.py
CPU only. Runs in seconds.
"""
import os
import sys
import numpy as np
from pathlib import Path
from itertools import combinations

# ---------------------------------------------------------------------
# Paths resolve through the repository-level config module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
# ---------------------------------------------------------------------

MODELS = {
    'Hybrid'      : str(C.HYBRID_DIR / 'hybrid_predictions.npz'),
    'LGCN'        : str(C.BASELINES_DIR / 'lgcn_predictions.npz'),
    'GAT'         : str(C.BASELINES_DIR / 'gat_predictions.npz'),
    'JGCN-orig'   : str(C.BASELINES_DIR / 'jgcn_predictions.npz'),
    'JGCN-matched': str(C.BASELINES_DIR / 'jgcn_matched_predictions.npz'),
}
OUT_REPORT = str(C.RESULTS_DIR / 'mcnemar_report.txt')

preds, truths = {}, {}
for name, p in MODELS.items():
    if not os.path.exists(p):
        print(f'  [skip] {name:14s}  file not found: {p}')
        continue
    d = np.load(p)
    if 'y_pred_notta' not in d.files or 'y_true_notta' not in d.files:
        print(f'  [skip] {name:14s}  npz missing y_pred_notta / y_true_notta')
        continue
    preds[name]  = np.asarray(d['y_pred_notta']).astype(int)
    truths[name] = np.asarray(d['y_true_notta']).astype(int)
    print(f'  [ok]   {name:14s}  N={len(preds[name])}')

if len(preds) < 2:
    sys.exit('\nNeed at least 2 models loaded - aborting.')

ref_name = list(truths.keys())[0]
ref_true = truths[ref_name]
for name, yt in truths.items():
    if len(yt) != len(ref_true) or not np.array_equal(yt, ref_true):
        sys.exit(f'\nERROR: {name} y_true differs from {ref_name}.')
print(f'\n  All models share the same test set of N={len(ref_true)} samples.\n')

def mcnemar(y_true, y_pred_A, y_pred_B):
    a = (y_pred_A == y_true)
    bb = (y_pred_B == y_true)
    b = int(np.sum( a & ~bb))
    c = int(np.sum(~a &  bb))
    n = b + c
    if n == 0:
        p_exact = 1.0
    else:
        from math import comb
        k = min(b, c)
        p_one = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
        p_exact = min(1.0, 2 * p_one)
    chi2 = 0.0 if n == 0 else (abs(b - c) - 1) ** 2 / n
    return b, c, p_exact, chi2

lines = []
def emit(s=''):
    print(s); lines.append(s)

emit('=' * 78)
emit("  McNemar's Exact Test - Pairwise Model Comparisons (non-TTA)")
emit(f'  Test set: N = {len(ref_true)} patches (seed=42, image-level 70/15/15 split)')
emit('=' * 78)
emit(f'  {"Pair":<28s}  {"b":>5s}  {"c":>5s}  {"|b-c|":>6s}  {"chi2_cc":>8s}  {"p_exact":>10s}  {"signif":<6s}')
emit('  ' + '-' * 74)

names = list(preds.keys())
if 'Hybrid' in names:
    pairs = [('Hybrid', b) for b in names if b != 'Hybrid']
    others = [n for n in names if n != 'Hybrid']
    pairs += list(combinations(others, 2))
else:
    pairs = list(combinations(names, 2))

results = []
for A, B in pairs:
    b, c, p, chi2 = mcnemar(ref_true, preds[A], preds[B])
    sig = ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns')
    emit(f'  {A+" vs "+B:<28s}  {b:>5d}  {c:>5d}  {abs(b-c):>6d}  {chi2:>8.3f}  {p:>10.2e}  {sig:<6s}')
    results.append((A, B, b, c, chi2, p, sig))

emit('  ' + '-' * 74)
emit('  b = A right, B wrong    c = A wrong, B right')
emit('  *** p<0.001    ** p<0.01    * p<0.05    ns not significant')
emit('=' * 78)

emit('\n\n% --- LaTeX table snippet ---')
emit(r'\begin{table}[htb]')
emit(r'\centering')
emit(r'\caption{McNemar\'s exact test on paired non-TTA predictions (N=%d).}' % len(ref_true))
emit(r'\label{tab:mcnemar}')
emit(r'\begin{tabular}{lccrr}')
emit(r'\toprule')
emit(r'Comparison & $b$ & $c$ & $\chi^2_{cc}$ & $p$-value \\')
emit(r'\midrule')
for A, B, b, c, chi2, p, sig in results:
    pstr = f'{p:.2e}' if p < 0.001 else f'{p:.3f}'
    emit(rf'{A} vs {B} & {b} & {c} & {chi2:.2f} & {pstr} \\')
emit(r'\bottomrule')
emit(r'\end{tabular}')
emit(r'\end{table}')

with open(OUT_REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f'\n  Report also written to: {OUT_REPORT}')

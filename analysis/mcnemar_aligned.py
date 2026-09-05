"""
E3 (v2) - Aligned Pairwise McNemar's Test
------------------------------------------
Reads *_aligned_raw.npz files (each contains file_paths + y_true/y_pred/y_prob).
Sorts each by file_paths -> canonical alignment, then runs McNemar's exact test.

Run AFTER each of the 5 retrain notebooks has run its alignment cell.

CPU only. Runs in seconds.
"""
import os, sys, numpy as np
from pathlib import Path
from itertools import combinations

# ---------------------------------------------------------------------
# Paths resolve through the repository-level config module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
# ---------------------------------------------------------------------

MODELS = {
    'Hybrid'      : str(C.HYBRID_DIR / 'hybrid_aligned_raw.npz'),
    'LGCN'        : str(C.BASELINES_DIR / 'lgcn_aligned_raw.npz'),
    'GAT'         : str(C.BASELINES_DIR / 'gat_aligned_raw.npz'),
    'JGCN-orig'   : str(C.BASELINES_DIR / 'jgcn_aligned_raw.npz'),
    'JGCN-matched': str(C.BASELINES_DIR / 'jgcn_matched_aligned_raw.npz'),
}
OUT_REPORT = str(C.RESULTS_DIR / 'mcnemar_aligned_report.txt')

# ── Load & sort by file_paths ─────────────────────────────────────────
loaded = {}
for name, p in MODELS.items():
    if not os.path.exists(p):
        print(f'  [MISSING] {name:14s}  not found: {p}')
        print(f'            -> run the alignment cell in that notebook first')
        continue
    d = np.load(p, allow_pickle=True)
    for k in ['file_paths', 'y_true', 'y_pred', 'y_prob']:
        if k not in d.files:
            sys.exit(f'{name}: npz missing key "{k}"')
    paths = d['file_paths']
    order = np.argsort(paths)
    loaded[name] = dict(
        paths  = paths[order],
        y_true = d['y_true'][order].astype(int),
        y_pred = d['y_pred'][order].astype(int),
        y_prob = d['y_prob'][order].astype(float),
    )
    print(f'  [ok] {name:14s}  N={len(paths)}  sorted by file_path')

if len(loaded) < 2:
    sys.exit('\nNeed at least 2 aligned models to run McNemar.')

# ── Verify alignment across all loaded models ─────────────────────────
ref_name = list(loaded.keys())[0]
ref_paths = loaded[ref_name]['paths']
ref_true  = loaded[ref_name]['y_true']
for name, d in loaded.items():
    if not np.array_equal(d['paths'], ref_paths):
        mm = np.where(d['paths'] != ref_paths)[0]
        sys.exit(f'ERROR: {name} file paths do not align with {ref_name} '
                 f'(first mismatch at index {mm[0]}):\n'
                 f'  {ref_name}: {ref_paths[mm[0]]}\n'
                 f'  {name}    : {d["paths"][mm[0]]}')
    if not np.array_equal(d['y_true'], ref_true):
        sys.exit(f'ERROR: {name} y_true differs from {ref_name} after sort.')
print(f'\n  All {len(loaded)} models aligned to same {len(ref_paths)} file paths.\n')

# ── McNemar's exact test ──────────────────────────────────────────────
def mcnemar(y_true, yA, yB):
    a = (yA == y_true); bb = (yB == y_true)
    b = int(np.sum( a & ~bb))
    c = int(np.sum(~a &  bb))
    n = b + c
    if n == 0:
        p = 1.0
    else:
        from math import comb
        k = min(b, c)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    chi2 = 0.0 if n == 0 else (abs(b - c) - 1) ** 2 / n
    return b, c, p, chi2

lines = []
def emit(s=''):
    print(s); lines.append(s)

emit('=' * 82)
emit("  McNemar's Exact Test - Pairwise Model Comparisons (aligned by file path)")
emit(f'  Test set: N = {len(ref_paths)} patches, seed=42, stratified 70/15/15 split')
emit('=' * 82)
emit(f'  {"Pair":<28s}  {"b":>5s}  {"c":>5s}  {"|b-c|":>6s}  {"chi2_cc":>8s}  {"p_exact":>10s}  signif')
emit('  ' + '-' * 78)

names = list(loaded.keys())
if 'Hybrid' in names:
    pairs = [('Hybrid', b) for b in names if b != 'Hybrid']
    others = [n for n in names if n != 'Hybrid']
    pairs += list(combinations(others, 2))
else:
    pairs = list(combinations(names, 2))

results = []
for A, B in pairs:
    b, c, p, chi2 = mcnemar(ref_true, loaded[A]['y_pred'], loaded[B]['y_pred'])
    sig = ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns')
    label = f'{A} vs {B}'
    emit(f'  {label:<28s}  {b:>5d}  {c:>5d}  {abs(b-c):>6d}  {chi2:>8.3f}  {p:>10.2e}  {sig}')
    results.append((A, B, b, c, chi2, p, sig))

emit('  ' + '-' * 78)
emit('  b = A right, B wrong    c = A wrong, B right')
emit('  *** p<0.001    ** p<0.01    * p<0.05    ns not significant')
emit('=' * 82)

emit('\n\n% --- LaTeX table snippet ---')
emit(r'\begin{table}[htb]')
emit(r'\centering')
emit(r'\caption{McNemar\'s exact test on paired non-TTA predictions (N=%d), aligned by file path.}' % len(ref_paths))
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

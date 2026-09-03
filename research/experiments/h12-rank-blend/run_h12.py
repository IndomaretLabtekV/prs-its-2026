"""H12 — Boundary-local rank blend V3 x refiner-v2.
Muat ulang STACK_RANK (replikasi V8), ambil ctrl/trt scores dari H10.
Pilih w pada folds selection (1-3, indeks 0-2), konfirmasi pada folds 4-5.
"""
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/athilla/Documents/IF_ITB/Lomba/DAC-PRS-ITS/prs-its')
NB = json.load(open(ROOT / 'notebooks/V8_Robust_Boundary.ipynb'))
CELL = {i: ''.join(NB['cells'][i].get('source', [])) for i in [1, 3, 5]}
NS = {}
for i in [1, 3, 5]:
    exec(compile(CELL[i], f'<nb-cell-{i:02d}>', 'exec'), NS)
g = globals()
for k, v in NS.items():
    g[k] = v
ID_COL, TARGET_COL = NS['ID_COL'], NS['TARGET_COL']
train_model, test_model, y = NS['train_model'], NS['test_model'], NS['y']
TIE_KEYS, TEST_TIE_KEYS = NS['TIE_KEYS'], NS['TEST_TIE_KEYS']
metric_row, nrc, CFG = NS['metric_row'], NS['normalized_recall_at_capacity'], NS['CFG']
train_clean = NS['train_clean']


def probability_logit(values):
    p = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fold_local_percentile(values, fold_id):
    values = np.asarray(values, dtype=float)
    out = np.zeros(len(values))
    for fold in sorted(np.unique(fold_id)):
        idx = np.flatnonzero(fold_id == fold)
        out[idx] = pd.Series(values[idx]).rank(method='average', pct=True).to_numpy()
    return out


z8 = zipfile.ZipFile(ROOT / 'results/v8/results.zip')
fa = pd.read_csv(io.BytesIO(z8.read('v8_artifacts/primary_fold_assignments.csv')))
FOLD_ID = fa['fold'].to_numpy(dtype=int)
if sorted(np.unique(FOLD_ID).tolist()) == [1, 2, 3, 4, 5]:
    FOLD_ID = FOLD_ID - 1
PRIMARY_SPLITS = [(np.flatnonzero(FOLD_ID != f), np.flatnonzero(FOLD_ID == f)) for f in range(5)]

z7 = zipfile.ZipFile(ROOT / 'results/v7/results.zip')
BASE_RESULTS = {}
for name in ['CatBoost V3 seed 2026', 'CatBoost V3 seed 2127', 'XGBoost V3 one-hot', 'Logistic V3', 'LightGBM V3 one-hot']:
    safe = name.lower().replace(' ', '_').replace('/', '_')
    d = np.load(io.BytesIO(z7.read(f'v7_artifacts/predictions_{safe}.npz')), allow_pickle=True)
    BASE_RESULTS[name] = {'oof': np.clip(d['oof'].astype(float), 1e-7, 1 - 1e-7)}
META_OOF = pd.DataFrame({f'base__{name.lower().replace(" ", "_").replace("-", "_")}__logit': probability_logit(r['oof']) for name, r in BASE_RESULTS.items()})
for name, r in BASE_RESULTS.items():
    META_OOF[f'base__{name.lower().replace(" ", "_").replace("-", "_")}__rank'] = fold_local_percentile(r['oof'], FOLD_ID)
META_OOF['ensemble__mean'] = np.column_stack([r['oof'] for r in BASE_RESULTS.values()]).mean(axis=1)
META_OOF = META_OOF.replace([np.inf, -np.inf], np.nan).fillna(0.0)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
stack_oof = np.full(len(y), np.nan)
for fold, (fit_idx, valid_idx) in enumerate(PRIMARY_SPLITS):
    m = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(C=1.0, max_iter=700, solver='lbfgs', random_state=CFG.seed + 900 + fold))])
    m.fit(META_OOF.iloc[fit_idx], y[fit_idx])
    stack_oof[valid_idx] = m.predict_proba(META_OOF.iloc[valid_idx])[:, 1]
STACK_RANK_OOF = fold_local_percentile(np.clip(stack_oof, 1e-7, 1 - 1e-7), FOLD_ID)
cand_mask = STACK_RANK_OOF >= (1.0 - 0.10 - 1e-12)
print('stack rank rebuilt', flush=True)

h10 = np.load(ROOT / 'research/experiments/h10-refiner-v2/h10_scores.npz')
ctrl, trt = h10['ctrl'].astype(float), h10['trt'].astype(float)


def boundary_rank_score(base_rank, refiner_prob, mask, rate, weight):
    out = np.asarray(base_rank, dtype=float).copy()
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if not len(idx):
        return out
    local_base = np.clip((base_rank[idx] - (1 - rate)) / rate, 0, 1)
    ref_rank = pd.Series(refiner_prob[idx]).rank(method='average', pct=True).to_numpy()
    out[idx] = np.clip((1 - weight) * local_base + weight * ref_rank, 0, 1) * rate + (1 - rate)
    return out


rank_ctrl = pd.Series(ctrl).rank(method='average', pct=True).to_numpy()
rank_trt = pd.Series(trt).rank(method='average', pct=True).to_numpy()

sel = FOLD_ID <= 2
conf = FOLD_ID >= 3
rows = []
for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
    mixed = (1 - w) * rank_ctrl + w * rank_trt
    score = boundary_rank_score(STACK_RANK_OOF, mixed, cand_mask, 0.10, 1.0)
    m_sel = metric_row(y[sel], score[sel], TIE_KEYS[sel], train_clean.loc[sel, 'jkpst'], train_clean.loc[sel, 'umur'], f'w{w}')
    m_conf = metric_row(y[conf], score[conf], TIE_KEYS[conf], train_clean.loc[conf, 'jkpst'], train_clean.loc[conf, 'umur'], f'w{w}')
    rows.append({'w': w, 'sel_NR': m_sel['NormalizedRecall@5%'], 'sel_TP': m_sel['fraud_captured'], 'conf_NR': m_conf['NormalizedRecall@5%'], 'conf_TP': m_conf['fraud_captured']})
df = pd.DataFrame(rows)
print(df.to_string(index=False), flush=True)
best_sel = df.sort_values('sel_NR', ascending=False).iloc[0]
print('best on selection: w=%.2f conf_NR=%.5f (ctrl w0 conf_NR=%.5f)' % (best_sel.w, best_sel.conf_NR, df[df.w == 0].conf_NR.iloc[0]), flush=True)
df.to_csv(ROOT / 'research/experiments/h12-rank-blend/h12_blend.csv', index=False)
print('H12 DONE')

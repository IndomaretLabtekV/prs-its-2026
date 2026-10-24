"""H10 — Refiner-v2: fitur V9 di refiner boundary V3.
Berjalan di env conda datsci, CUDA_VISIBLE_DEVICES=0.
Replikasi stack V8 + refiner V3 (kontrol) vs refiner-v2 (V9_FEATURES).
"""
import io
import json
import os
import time
import zipfile
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

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
FULL_FEATURES = NS['FULL_FEATURES']
CATEGORICAL_COLUMNS = NS['CATEGORICAL_COLUMNS']
TIE_KEYS, TEST_TIE_KEYS = NS['TIE_KEYS'], NS['TEST_TIE_KEYS']
metric_row = NS['metric_row']
nrc = NS['normalized_recall_at_capacity']
CFG = NS['CFG']

# ---- fold otoritatif artefak ----
z8 = zipfile.ZipFile(ROOT / 'results/v8/results.zip')
fa = pd.read_csv(io.BytesIO(z8.read('v8_artifacts/primary_fold_assignments.csv')))
assert fa[ID_COL].astype(str).tolist() == TIE_KEYS.tolist()
FOLD_ID = fa['fold'].to_numpy(dtype=int)
if sorted(np.unique(FOLD_ID).tolist()) == [1, 2, 3, 4, 5]:
    FOLD_ID = FOLD_ID - 1
PRIMARY_SPLITS = [(np.flatnonzero(FOLD_ID != f), np.flatnonzero(FOLD_ID == f)) for f in range(5)]
print('folds ok, overlap by construction', flush=True)

# ---- rebuild 5 model base predictions + meta (stack V8) dari v7 zip ----
z7 = zipfile.ZipFile(ROOT / 'results/v7/results.zip')
DEFS = [('CatBoost V3 seed 2026', 2026), ('CatBoost V3 seed 2127', 2127),
        ('XGBoost V3 one-hot', 1), ('Logistic V3', 2), ('LightGBM V3 one-hot', 3)]
BASE_RESULTS = {}
for name, _ in DEFS:
    safe = name.lower().replace(' ', '_').replace('/', '_')
    d = np.load(io.BytesIO(z7.read(f"v7_artifacts/predictions_{safe}.npz")), allow_pickle=True)
    assert d['oof'].shape == (len(y),) and d['test'].shape == (len(test_model),)
    BASE_RESULTS[name] = {'name': name, 'oof': np.clip(d['oof'].astype(float), 1e-7, 1 - 1e-7),
                          'test': np.clip(d['test'].astype(float), 1e-7, 1 - 1e-7)}
print('5 base preds loaded', flush=True)


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


META_OOF = pd.DataFrame({f'base__{name.lower().replace(" ", "_").replace("-", "_")}__logit': probability_logit(r['oof']) for name, r in BASE_RESULTS.items()})
for name, r in BASE_RESULTS.items():
    META_OOF[f'base__{name.lower().replace(" ", "_").replace("-", "_")}__rank'] = fold_local_percentile(r['oof'], FOLD_ID)
META_OOF['ensemble__mean'] = np.column_stack([r['oof'] for r in BASE_RESULTS.values()]).mean(axis=1)
META_OOF = META_OOF.replace([np.inf, -np.inf], np.nan).fillna(0.0)
print('meta built', META_OOF.shape, flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def crossfit_stack():
    oof = np.full(len(y), np.nan)
    for fold, (fit_idx, valid_idx) in enumerate(PRIMARY_SPLITS):
        m = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(C=1.0, max_iter=700, solver='lbfgs', random_state=CFG.seed + 900 + fold))])
        m.fit(META_OOF.iloc[fit_idx], y[fit_idx])
        oof[valid_idx] = m.predict_proba(META_OOF.iloc[valid_idx])[:, 1]
    assert not np.isnan(oof).any()
    return np.clip(oof, 1e-7, 1 - 1e-7)


STACK_OOF = crossfit_stack()
STACK_RANK_OOF = fold_local_percentile(STACK_OOF, FOLD_ID)
print('stack OOF built', flush=True)


def candidate_mask_from_rank(rank, rate):
    return np.asarray(rank) >= (1.0 - rate - 1e-12)


candidate_mask = candidate_mask_from_rank(STACK_RANK_OOF, 0.10)
print('pool size:', int(candidate_mask.sum()), flush=True)


def catboost_ready(frame, categorical):
    out = frame.copy()
    for c in categorical:
        if c in out:
            out[c] = out[c].astype('string').fillna('__MISSING__').astype(str)
    return out


def fit_refiner(X_fit, target, X_valid, features, cats, seed):
    from catboost import CatBoostClassifier
    params = dict(loss_function='Logloss', allow_writing_files=False, verbose=False,
                  random_seed=seed, task_type='GPU', devices='0', iterations=900,
                  learning_rate=0.035, depth=7, l2_leaf_reg=10.0, random_strength=0.35,
                  bootstrap_type='Bayesian', bagging_temperature=0.5,
                  max_ctr_complexity=2, border_count=128)
    m = CatBoostClassifier(**params)
    m.fit(catboost_ready(X_fit[features], cats), target, cat_features=cats, verbose=False)
    return m.predict_proba(catboost_ready(X_valid[features], cats))[:, 1]


def boundary_rank_score(base_rank, refiner_prob, cand_mask, rate, weight):
    base_rank = np.asarray(base_rank, dtype=float)
    refiner_prob = np.asarray(refiner_prob, dtype=float)
    cand_mask = np.asarray(cand_mask, dtype=bool)
    out = base_rank.copy()
    idx = np.flatnonzero(cand_mask)
    if not len(idx):
        return out
    local_base = np.clip((base_rank[idx] - (1 - rate)) / rate, 0, 1)
    ref_rank = pd.Series(refiner_prob[idx]).rank(method='average', pct=True).to_numpy()
    out[idx] = np.clip((1 - weight) * local_base + weight * ref_rank, 0, 1) * rate + (1 - rate)
    return out


# ---- fitur V9 (verbatim engineer_v9 dari builder) ----
RISK_KDKC = {'2101', '2103'}
V9_NEW = ['risk_subgroup', 'risk_subgroup__age_band', 'risk_subgroup__los_band', 'risk_subgroup__burden_band']


def engineer_v9(frame):
    b_edges = [-np.inf] + sorted(pd.to_numeric(frame['total_coded_burden'], errors='coerce').fillna(0).quantile([1 / 3, 2 / 3]).tolist()) + [np.inf]
    k = frame['kdkc'].astype('string')
    d = frame['dati2'].astype('string')
    t = frame['typeppk'].astype('string')
    sub = np.where(t.eq('KM'), 'KM', np.where(k.isin(RISK_KDKC), 'KDKC2101_2103', np.where(d.eq('309'), 'DATI2_309', 'OTHER')))
    frame['risk_subgroup'] = pd.Series(sub, index=frame.index).astype('string')
    frame['risk_subgroup__age_band'] = (frame['risk_subgroup'] + '__' + frame['age_band'].astype('string')).astype('string')
    los = pd.to_numeric(frame['los'], errors='coerce').fillna(0)
    los_band = pd.cut(los, [-np.inf, 0, 2, np.inf], labels=['los0', 'los1_2', 'los3p']).astype('string')
    frame['risk_subgroup__los_band'] = (frame['risk_subgroup'] + '__' + los_band).astype('string')
    bb = pd.cut(pd.to_numeric(frame['total_coded_burden'], errors='coerce').fillna(0), bins=b_edges, labels=['b0', 'b1', 'b2']).astype('string')
    frame['risk_subgroup__burden_band'] = (frame['risk_subgroup'] + '__' + bb).astype('string')
    return frame


train_v9 = engineer_v9(train_model.copy())
V9_FEATURES = FULL_FEATURES + V9_NEW
V9_CATS = [c for c in V9_FEATURES if c in CATEGORICAL_COLUMNS or str(train_v9[c].dtype) in {'object', 'string'}]


def evaluate(arm_name, features, frame):
    ref_oof = np.full(len(y), np.nan)
    t0 = time.time()
    for fold, (fit_idx, valid_idx) in enumerate(PRIMARY_SPLITS):
        fit_cand = fit_idx[candidate_mask[fit_idx]]
        val_cand = valid_idx[candidate_mask[valid_idx]]
        assert len(np.unique(y[fit_cand])) == 2, f'single-class pool fold {fold}'
        ref_oof[val_cand] = fit_refiner(frame.iloc[fit_cand], y[fit_cand], frame.iloc[val_cand], features, V9_CATS if features is V9_FEATURES else CATEGORICAL_COLUMNS, CFG.seed + 3000 + fold)
    assert not np.isnan(ref_oof[candidate_mask]).any()
    score = boundary_rank_score(STACK_RANK_OOF, ref_oof, candidate_mask, 0.10, 1.0)
    m = metric_row(y, score, TIE_KEYS, train_clean['jkpst'], train_clean['umur'], arm_name)
    print(f'{arm_name}: {time.time()-t0:.0f}s NR={m["NormalizedRecall@5%"]:.5f} TP={m["fraud_captured"]} AP={m["AveragePrecision"]:.5f}', flush=True)
    return score, m


ctrl_score, m_ctrl = evaluate('control-refiner-v1', FULL_FEATURES, train_model)
trt_score, m_trt = evaluate('refiner-v2-V9', V9_FEATURES, train_v9)
delta = m_trt['NormalizedRecall@5%'] - m_ctrl['NormalizedRecall@5%']
print(f'DELTA NR: {delta:+.5f} | DELTA TP: {m_trt["fraud_captured"] - m_ctrl["fraud_captured"]:+d}', flush=True)
np.savez_compressed(ROOT / 'research/experiments/h10-refiner-v2/h10_scores.npz',
                    ctrl=ctrl_score.astype(np.float32), trt=trt_score.astype(np.float32))
print('H10 DONE')

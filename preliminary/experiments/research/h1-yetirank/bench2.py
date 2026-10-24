"""Benchmark v2 (FIX):
  - Fold dari artefak V8 langsung (primary_fold_assignments.csv), bukan komputasi
    lokal (hash pandas beda versi -> split beda).
  - YetiRank CPU: query di-chunk <=1000 baris DAN contiguous (syarat CatBoost).
  - LightGBM LambdaRank CPU grup besar (4 grup fold) + GPU chunked.
  1 fold (valid fold 0), 300 iter. NR fold-0 hanya sanity, bukan angka final.
  Env: conda datsci.
"""
import json
import time

import numpy as np
import pandas as pd

NB = json.load(open('notebooks/V8_Robust_Boundary.ipynb'))
CELL = {i: ''.join(NB['cells'][i].get('source', [])) for i in [1, 3, 5]}
NS = {}
for i in [1, 3, 5]:
    exec(compile(CELL[i], f'<nb-cell-{i:02d}>', 'exec'), NS)

g = globals()
for k, v in NS.items():
    g[k] = v

# Fold dari artefak (ground truth, bukan komputasi lokal)
art = pd.read_csv('/tmp/opencode/v8/v8_artifacts/primary_fold_assignments.csv')
assert art[NS['ID_COL']].astype(str).tolist() == NS['TIE_KEYS'].tolist(), \
    'urutan claim_id artefak != train (mapping OOF jebol)'
FOLD_ID = art['fold'].to_numpy(dtype=int)
if sorted(np.unique(FOLD_ID).tolist()) == [1, 2, 3, 4, 5]:
    FOLD_ID = FOLD_ID - 1
PRIMARY_SPLITS = [(np.flatnonzero(FOLD_ID != f), np.flatnonzero(FOLD_ID == f))
                  for f in range(5)]
print('FOLD: dari artefak V8, 5 fold, overlap by construction', flush=True)

ID_COL, TARGET_COL = NS['ID_COL'], NS['TARGET_COL']
train_model, y = NS['train_model'], NS['y']
FULL_FEATURES = NS['FULL_FEATURES']
CATEGORICAL_COLUMNS = NS['CATEGORICAL_COLUMNS']
TIE_KEYS = NS['TIE_KEYS']
nrc = NS['normalized_recall_at_capacity']

ITERS = 300
fit_idx, valid_idx = PRIMARY_SPLITS[0]
yv, tk = y[valid_idx], TIE_KEYS[valid_idx]


def catboost_ready(frame, categorical):
    out = frame.copy()
    for c in categorical:
        if c in out:
            out[c] = out[c].astype('string').fillna('__MISSING__').astype(str)
    return out


CATS = [c for c in FULL_FEATURES if c in CATEGORICAL_COLUMNS]
BASE = dict(allow_writing_files=False, verbose=False, random_seed=2127,
            iterations=ITERS, learning_rate=0.03, depth=8, l2_leaf_reg=8.0,
            random_strength=0.35, bootstrap_type='Bayesian',
            bagging_temperature=0.5, max_ctr_complexity=2, border_count=128)


def bench(name, fn):
    t0 = time.time()
    try:
        p = fn()
        dt = time.time() - t0
        nr = nrc(yv, np.asarray(p, dtype=float), 0.05, tk)
        print(f'{name}: {dt:.1f}s NR_fold0={nr:.5f}', flush=True)
    except Exception as e:
        print(f'{name}: GAGAL ({type(e).__name__}: {str(e)[:150]})', flush=True)


def arm_logloss_gpu():
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(loss_function='Logloss', task_type='GPU',
                           devices='0', **BASE)
    m.fit(catboost_ready(train_model.iloc[fit_idx][FULL_FEATURES], CATS),
          y[fit_idx], cat_features=CATS, verbose=False)
    return m.predict_proba(
        catboost_ready(train_model.iloc[valid_idx][FULL_FEATURES], CATS))[:, 1]


def arm_yetirank_cpu_chunked():
    from catboost import CatBoostRanker
    rng = np.random.RandomState(2127)
    perm = rng.permutation(len(fit_idx))
    chunks = np.array_split(fit_idx[perm], max(1, len(fit_idx) // 1000))
    qid = np.empty(len(y), dtype=np.int64)
    for q, ch in enumerate(chunks):
        qid[ch] = q
    order = np.argsort(qid[fit_idx], kind='stable')  # contiguous per query
    oi = fit_idx[order]
    Xf = catboost_ready(train_model.iloc[oi][FULL_FEATURES], CATS)
    m = CatBoostRanker(loss_function='YetiRank', task_type='CPU', **BASE)
    m.fit(Xf, y[oi], group_id=qid[oi], cat_features=CATS, verbose=False)
    return m.predict(
        catboost_ready(train_model.iloc[valid_idx][FULL_FEATURES], CATS))


def lgbm_frames():
    Xf = train_model.iloc[fit_idx][FULL_FEATURES].copy()
    Xv = train_model.iloc[valid_idx][FULL_FEATURES].copy()
    for c in CATS:
        if c in Xf:
            Xf[c] = Xf[c].astype('string').fillna('__MISSING__').astype('category')
            Xv[c] = Xv[c].astype('string').fillna('__MISSING__').astype('category')
    for c in FULL_FEATURES:
        if c not in CATS:
            Xf[c] = pd.to_numeric(Xf[c], errors='coerce').astype('float32')
            Xv[c] = pd.to_numeric(Xv[c], errors='coerce').astype('float32')
    return Xf, Xv


def arm_lgbm_rank_cpu():
    import lightgbm as lgb
    Xf, Xv = lgbm_frames()
    groups = np.bincount(FOLD_ID[fit_idx]).tolist()
    ds = lgb.Dataset(Xf, label=y[fit_idx], group=groups,
                     categorical_feature=CATS, free_raw_data=False)
    m = lgb.train(dict(objective='lambdarank', device='cpu',
                       num_iterations=ITERS, learning_rate=0.05,
                       num_leaves=127, min_data_in_leaf=100,
                       seed=2127, verbose=-1), ds)
    return m.predict(Xv)


bench('A logloss-GPU  ', arm_logloss_gpu)
bench('B yetirank-chunk', arm_yetirank_cpu_chunked)
bench('C lgbm-rank-CPU', arm_lgbm_rank_cpu)
print('BENCH2 DONE')

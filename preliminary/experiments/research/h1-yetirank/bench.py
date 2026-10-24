"""Benchmark compute lokal (FIX dari smoke.py: tanpa DAC_FAST_MODE agar CFG full 5-fold,
match artefak V8). Ukur WAKTU saja, 1 fold (valid fold 0), 300 iter:
  arm A: CatBoost Logloss GPU (baseline verbatim)
  arm B: CatBoost Logloss CPU (pembanding)
  arm C: LightGBM LambdaRank GPU, group = fold asal (4 grup train)
  arm D: CatBoost YetiRank CPU, query di-chunk <=1000 baris (hindari OOM)
NR fold-0 dilaporkan apa adanya (300 iter, BUKAN angka final).
"""
import json
import time

import numpy as np
import pandas as pd

NB = json.load(open('notebooks/V8_Robust_Boundary.ipynb'))
CELL = {i: ''.join(NB['cells'][i].get('source', [])) for i in [1, 3, 5, 7]}
NS = {}
for i in [1, 3, 5, 7]:
    exec(compile(CELL[i], f'<nb-cell-{i:02d}>', 'exec'), NS)

g = globals()
for k, v in NS.items():
    g[k] = v
art = pd.read_csv('/tmp/opencode/v8/v8_artifacts/primary_fold_assignments.csv')
art_folds = art['fold'].to_numpy(dtype=int)
if sorted(np.unique(art_folds).tolist()) == [1, 2, 3, 4, 5]:
    art_folds = art_folds - 1
print('FOLD MATCH ARTIFACT:', bool(np.array_equal(NS['FOLD_ID'], art_folds)))

ID_COL, TARGET_COL = NS['ID_COL'], NS['TARGET_COL']
train_model, y = NS['train_model'], NS['y']
FULL_FEATURES = NS['FULL_FEATURES']
CATEGORICAL_COLUMNS = NS['CATEGORICAL_COLUMNS']
TIE_KEYS = NS['TIE_KEYS']
PRIMARY_SPLITS = NS['PRIMARY_SPLITS']
FOLD_ID = NS['FOLD_ID']
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
    cats = [c for c in FULL_FEATURES if c in CATEGORICAL_COLUMNS]
    m = CatBoostClassifier(loss_function='Logloss', allow_writing_files=False,
                           verbose=False, random_seed=2127, task_type='GPU',
                           devices='0', iterations=ITERS, learning_rate=0.03,
                           depth=8, l2_leaf_reg=8.0, random_strength=0.35,
                           bootstrap_type='Bayesian', bagging_temperature=0.5,
                           max_ctr_complexity=2, border_count=128)
    m.fit(catboost_ready(train_model.iloc[fit_idx][FULL_FEATURES], cats),
          y[fit_idx], cat_features=cats, verbose=False)
    return m.predict_proba(
        catboost_ready(train_model.iloc[valid_idx][FULL_FEATURES], cats))[:, 1]


def arm_logloss_cpu():
    from catboost import CatBoostClassifier
    cats = [c for c in FULL_FEATURES if c in CATEGORICAL_COLUMNS]
    m = CatBoostClassifier(loss_function='Logloss', allow_writing_files=False,
                           verbose=False, random_seed=2127, task_type='CPU',
                           iterations=ITERS, learning_rate=0.03,
                           depth=8, l2_leaf_reg=8.0, random_strength=0.35,
                           bootstrap_type='Bayesian', bagging_temperature=0.5,
                           max_ctr_complexity=2, border_count=128)
    m.fit(catboost_ready(train_model.iloc[fit_idx][FULL_FEATURES], cats),
          y[fit_idx], cat_features=cats, verbose=False)
    return m.predict_proba(
        catboost_ready(train_model.iloc[valid_idx][FULL_FEATURES], cats))[:, 1]


def arm_lgbm_lambdarank_gpu():
    import lightgbm as lgb
    cats = [c for c in FULL_FEATURES if c in CATEGORICAL_COLUMNS]
    Xf = train_model.iloc[fit_idx][FULL_FEATURES].copy()
    Xv = train_model.iloc[valid_idx][FULL_FEATURES].copy()
    for c in cats:
        if c in Xf:
            Xf[c] = Xf[c].astype('string').fillna('__MISSING__').astype('category')
            Xv[c] = Xv[c].astype('string').fillna('__MISSING__').astype('category')
    for c in FULL_FEATURES:
        if c not in cats:
            Xf[c] = pd.to_numeric(Xf[c], errors='coerce').astype('float32')
            Xv[c] = pd.to_numeric(Xv[c], errors='coerce').astype('float32')
    groups = np.bincount(FOLD_ID[fit_idx]).tolist()
    ds = lgb.Dataset(Xf, label=y[fit_idx], group=groups, categorical_feature=cats,
                     free_raw_data=False)
    params = dict(objective='lambdarank', device='gpu', num_iterations=ITERS,
                  learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                  seed=2127, verbose=-1)
    m = lgb.train(params, ds)
    return m.predict(Xv)


def arm_yetirank_cpu_chunked():
    from catboost import CatBoostRanker
    cats = [c for c in FULL_FEATURES if c in CATEGORICAL_COLUMNS]
    rng = np.random.RandomState(2127)
    perm = rng.permutation(len(fit_idx))
    chunk = np.array_split(fit_idx[perm], max(1, len(fit_idx) // 1000))
    qid = np.empty(len(y), dtype=int)
    for q, ch in enumerate(chunk):
        qid[ch] = q
    m = CatBoostRanker(loss_function='YetiRank', allow_writing_files=False,
                       verbose=False, random_seed=2127, task_type='CPU',
                       iterations=ITERS, learning_rate=0.03,
                       depth=8, l2_leaf_reg=8.0, random_strength=0.35,
                       bootstrap_type='Bayesian', bagging_temperature=0.5,
                       max_ctr_complexity=2, border_count=128)
    m.fit(catboost_ready(train_model.iloc[fit_idx][FULL_FEATURES], cats),
          y[fit_idx], group_id=qid[fit_idx], cat_features=cats, verbose=False)
    return m.predict(
        catboost_ready(train_model.iloc[valid_idx][FULL_FEATURES], cats))


bench('A logloss-GPU ', arm_logloss_gpu)
bench('B logloss-CPU ', arm_logloss_cpu)
bench('C lgbm-rank-GPU', arm_lgbm_lambdarank_gpu)
bench('D yetirank-chunk', arm_yetirank_cpu_chunked)
print('BENCH DONE')

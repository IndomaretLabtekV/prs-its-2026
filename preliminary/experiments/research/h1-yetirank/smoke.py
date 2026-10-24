"""H1 SMOKE (machine-extracted pipeline, verbatim dari notebook V8).
Uji: 1 fold (valid fold 0), 300 iter — Logloss (fit_cat verbatim sel 09)
vs YetiRank (CatBoostRanker). Ukur waktu + sanity NR fold-0 resmi.
"""
import json
import os
import time

os.environ.setdefault('DAC_FAST_MODE', '1')  # hindari raise LightGBM; iter/fold dioverride manual

import numpy as np
import pandas as pd

NB = json.load(open('notebooks/V8_Robust_Boundary.ipynb'))
CELL = {i: ''.join(NB['cells'][i].get('source', [])) for i in [1, 3, 5, 7, 9]}
NS = {}
for i in [1, 3, 5, 7]:
    exec(compile(CELL[i], f'<nb-cell-{i:02d}>', 'exec'), NS)

# Validasi: fold komputasi == fold artefak V8 (kontrak reproduksibilitas)
g = globals()
for k, v in NS.items():
    g[k] = v
art = pd.read_csv('/tmp/opencode/v8/v8_artifacts/primary_fold_assignments.csv')
art_folds = art['fold'].to_numpy(dtype=int)
if sorted(np.unique(art_folds).tolist()) == [1, 2, 3, 4, 5]:
    art_folds = art_folds - 1
print('FOLD MATCH ARTIFACT:', bool(np.array_equal(NS['FOLD_ID'], art_folds)))

ID_COL, TARGET_COL = NS['ID_COL'], NS['TARGET_COL']
train_model, test_model, y = NS['train_model'], NS['test_model'], NS['y']
FULL_FEATURES = NS['FULL_FEATURES']
CATEGORICAL_COLUMNS = NS['CATEGORICAL_COLUMNS']
TIE_KEYS = NS['TIE_KEYS']
PRIMARY_SPLITS = NS['PRIMARY_SPLITS']
CAT_TASK, CAT_DEVICES = NS['CAT_TASK'], NS['CAT_DEVICES']
nrc = NS['normalized_recall_at_capacity']
print('task:', CAT_TASK, 'devices:', CAT_DEVICES, '| features:', len(FULL_FEATURES))


def catboost_ready(frame, categorical):
    out = frame.copy()
    for c in categorical:
        if c in out:
            out[c] = out[c].astype('string').fillna('__MISSING__').astype(str)
    return out


def fit_logloss(fit_idx, valid_idx, seed, iterations):
    from catboost import CatBoostClassifier
    cats = [c for c in FULL_FEATURES if c in CATEGORICAL_COLUMNS]
    params = dict(loss_function='Logloss', allow_writing_files=False, verbose=False,
                  random_seed=seed, task_type=CAT_TASK, iterations=iterations,
                  learning_rate=0.03, depth=8, l2_leaf_reg=8.0, random_strength=0.35,
                  bootstrap_type='Bayesian', bagging_temperature=0.5,
                  max_ctr_complexity=2, border_count=128)
    if CAT_DEVICES:
        params['devices'] = CAT_DEVICES
    model = CatBoostClassifier(**params)
    model.fit(catboost_ready(train_model.iloc[fit_idx][FULL_FEATURES], cats),
              y[fit_idx], cat_features=cats, verbose=False)
    return model.predict_proba(
        catboost_ready(train_model.iloc[valid_idx][FULL_FEATURES], cats))[:, 1]


def fit_yetirank(fit_idx, valid_idx, seed, iterations):
    from catboost import CatBoostRanker
    cats = [c for c in FULL_FEATURES if c in CATEGORICAL_COLUMNS]
    params = dict(loss_function='YetiRank', allow_writing_files=False, verbose=False,
                  random_seed=seed, task_type=CAT_TASK, iterations=iterations,
                  learning_rate=0.03, depth=8, l2_leaf_reg=8.0, random_strength=0.35,
                  bootstrap_type='Bayesian', bagging_temperature=0.5,
                  max_ctr_complexity=2, border_count=128)
    if CAT_DEVICES:
        params['devices'] = CAT_DEVICES
    model = CatBoostRanker(**params)
    model.fit(catboost_ready(train_model.iloc[fit_idx][FULL_FEATURES], cats),
              y[fit_idx], group_id=NS['FOLD_ID'][fit_idx],
              cat_features=cats, verbose=False)
    return model.predict(
        catboost_ready(train_model.iloc[valid_idx][FULL_FEATURES], cats))


ITERS = 300
fit_idx, valid_idx = PRIMARY_SPLITS[0]
yv = y[valid_idx]
tk = TIE_KEYS[valid_idx]

t0 = time.time()
p_log = fit_logloss(fit_idx, valid_idx, 2127, ITERS)
t_log = time.time() - t0
nr_log = nrc(yv, p_log, 0.05, tk)
print(f'LOGLOSS  fold0: {t_log:.1f}s NR={nr_log:.5f}')

t0 = time.time()
p_yr = fit_yetirank(fit_idx, valid_idx, 2127, ITERS)
t_yr = time.time() - t0
nr_yr = nrc(yv, p_yr, 0.05, tk)
print(f'YETIRANK fold0: {t_yr:.1f}s NR={nr_yr:.5f}')

full_est = (t_log + t_yr) / 2 / ITERS * 2860 * 5
print(f'estimasi FULL (5fold x 2860iter x 2 arms, 1 GPU): ~{full_est/60:.0f} mnt')
print('SMOKE DONE')

"""H15: positive-cost refiner, one variable over the H10 setup."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/athilla/Documents/IF_ITB/Lomba/DAC-PRS-ITS/prs-its')
H10 = ROOT / 'research/experiments/h10-refiner-v2/run_h10.py'

# Reuse the audited data, folds, features, stack, and selector definitions.
prefix = H10.read_text(encoding='utf-8').split(
    "ctrl_score, m_ctrl = evaluate('control-refiner-v1', FULL_FEATURES, train_model)",
    1,
)[0]
NS = {'__name__': '__main__'}
exec(compile(prefix, str(H10), 'exec'), NS)
globals().update(NS)

from catboost import CatBoostClassifier

CURRENT_POSITIVE_WEIGHT = 1.0


def fit_refiner(X_fit, target, X_valid, features, cats, seed, iterations=900):
    params = dict(
        loss_function='Logloss', allow_writing_files=False, verbose=False,
        random_seed=seed, task_type='GPU', devices='0', iterations=iterations,
        learning_rate=0.035, depth=7, l2_leaf_reg=10.0,
        random_strength=0.35, bootstrap_type='Bayesian', bagging_temperature=0.5,
        max_ctr_complexity=2, border_count=128,
        class_weights=[1.0, CURRENT_POSITIVE_WEIGHT],
    )
    model = CatBoostClassifier(**params)
    model.fit(catboost_ready(X_fit[features], cats), target,
              cat_features=cats, verbose=False)
    return model.predict_proba(catboost_ready(X_valid[features], cats))[:, 1]


def run_weight(weight):
    global CURRENT_POSITIVE_WEIGHT
    CURRENT_POSITIVE_WEIGHT = float(weight)
    ref_oof = np.full(len(y), np.nan)
    for fold, (fit_idx, valid_idx) in enumerate(PRIMARY_SPLITS):
        fit_cand = fit_idx[candidate_mask[fit_idx]]
        val_cand = valid_idx[candidate_mask[valid_idx]]
        ref_oof[val_cand] = fit_refiner(
            train_v9.iloc[fit_cand], y[fit_cand], train_v9.iloc[val_cand],
            V9_FEATURES, V9_CATS, CFG.seed + 3000 + fold,
        )
    assert not np.isnan(ref_oof[candidate_mask]).any()
    score = boundary_rank_score(STACK_RANK_OOF, ref_oof, candidate_mask, 0.10, 1.0)
    metrics = metric_row(
        y, score, TIE_KEYS, train_clean['jkpst'], train_clean['umur'],
        f'positive_weight_{weight:g}',
    )
    base_fold_tp = []
    trial_fold_tp = []
    for _, valid_idx in PRIMARY_SPLITS:
        base_selected = top_k_mask(BASELINE[valid_idx], 0.05, TIE_KEYS[valid_idx])
        trial_selected = top_k_mask(score[valid_idx], 0.05, TIE_KEYS[valid_idx])
        base_fold_tp.append(int(y[valid_idx][base_selected].sum()))
        trial_fold_tp.append(int(y[valid_idx][trial_selected].sum()))
    metrics['tp_delta'] = int(metrics['fraud_captured'] - BASE_METRICS['fraud_captured'])
    metrics['nr_delta'] = float(metrics['NormalizedRecall@5%'] - BASE_METRICS['NormalizedRecall@5%'])
    metrics['fold_tp'] = trial_fold_tp
    metrics['fold_tp_delta'] = [b - a for a, b in zip(base_fold_tp, trial_fold_tp)]
    print(
        f"weight={weight:g} NR={metrics['NormalizedRecall@5%']:.5f} "
        f"TP={metrics['fraud_captured']} delta={metrics['tp_delta']:+d} "
        f"fold_delta={metrics['fold_tp_delta']}",
        flush=True,
    )
    return score, metrics


baseline_path = ROOT / 'research/experiments/h10-refiner-v2/h10_scores.npz'
BASELINE = np.load(baseline_path)['trt'].astype(float)
BASE_METRICS = metric_row(
    y, BASELINE, TIE_KEYS, train_clean['jkpst'], train_clean['umur'],
    'h10_control_v9',
)
results = {}
for weight in [1.25, 1.5, 2.0]:
    results[weight] = run_weight(weight)

out = ROOT / 'research/experiments/h15-positive-cost-refiner'
rows = []
for weight, (_, metrics) in results.items():
    rows.append({
        'positive_weight': weight,
        'NormalizedRecall@5%': metrics['NormalizedRecall@5%'],
        'fraud_captured': metrics['fraud_captured'],
        'NR_delta_vs_h10': metrics['nr_delta'],
        'TP_delta_vs_h10': metrics['tp_delta'],
        'AveragePrecision': metrics['AveragePrecision'],
        'fold_tp_delta': json.dumps(metrics['fold_tp_delta']),
    })
pd.DataFrame(rows).to_csv(out / 'h15_registry.csv', index=False)
np.savez_compressed(out / 'h15_scores.npz', **{
    f'w{str(weight).replace(".", "_")}': score.astype(np.float32)
    for weight, (score, _) in results.items()
})
print('H15 DONE', flush=True)

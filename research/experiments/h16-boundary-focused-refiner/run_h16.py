"""H16 - Boundary-focused refiner: V9 refiner with sample weights near the
5% audit boundary, one variable over the audited H10 setup.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/athilla/Documents/IF_ITB/Lomba/DAC-PRS-ITS/prs-its')


def boundary_sample_weights(rank, alpha, boundary=0.95, band=0.02):
    rank = np.asarray(rank, dtype=float)
    focus = np.maximum(0.0, 1.0 - np.abs(rank - boundary) / band)
    return 1.0 + float(alpha) * focus


def self_check_weights():
    rank = np.array([0.90, 0.93, 0.95, 0.97, 1.00])
    assert np.allclose(boundary_sample_weights(rank, 1), [1, 1, 2, 1, 1])
    assert np.allclose(boundary_sample_weights(rank, 7).max(), 8)
    assert np.all(boundary_sample_weights(rank, 3) >= 1)
    print('H16_WEIGHT_CHECK_PASS', flush=True)


if '--self-check' in sys.argv:
    self_check_weights()
    raise SystemExit(0)

# Reuse the audited H10 bootstrap through the line before its evaluate calls.
H10 = ROOT / 'research/experiments/h10-refiner-v2/run_h10.py'
H10_EVALUATE_MARKER = "ctrl_score, m_ctrl = evaluate('control-refiner-v1', FULL_FEATURES, train_model)"
h10_source = H10.read_text(encoding='utf-8')
if H10_EVALUATE_MARKER not in h10_source:
    raise RuntimeError(
        f'H10 evaluate marker not found in {H10}; refusing to bootstrap. '
        'Expected the exact line: ' + H10_EVALUATE_MARKER
    )
prefix = h10_source.split(H10_EVALUATE_MARKER, 1)[0]
NS = {'__name__': '__main__'}
exec(compile(prefix, str(H10), 'exec'), NS)
globals().update(NS)

from catboost import CatBoostClassifier  # noqa: E402


def fit_refiner(X_fit, target, sample_weight, X_valid, seed):
    params = dict(
        loss_function='Logloss', allow_writing_files=False, verbose=False,
        random_seed=seed, task_type='GPU', devices='0', iterations=900,
        learning_rate=0.035, depth=7, l2_leaf_reg=10.0,
        random_strength=0.35, bootstrap_type='Bayesian',
        bagging_temperature=0.5, max_ctr_complexity=2, border_count=128,
    )
    model = CatBoostClassifier(**params)
    model.fit(
        catboost_ready(X_fit[V9_FEATURES], V9_CATS), target,
        sample_weight=sample_weight, cat_features=V9_CATS, verbose=False,
    )
    return model.predict_proba(
        catboost_ready(X_valid[V9_FEATURES], V9_CATS)
    )[:, 1]


baseline_path = ROOT / 'research/experiments/h10-refiner-v2/h10_scores.npz'
BASELINE = np.load(baseline_path)['trt'].astype(float)
assert BASELINE.shape == (len(y),)
assert np.isfinite(BASELINE).all()
BASE_METRICS = metric_row(
    y, BASELINE, TIE_KEYS, train_clean['jkpst'], train_clean['umur'],
    'h10_control_v9',
)


def run_alpha(alpha):
    ref_oof = np.full(len(y), np.nan)
    for fold, (fit_idx, valid_idx) in enumerate(PRIMARY_SPLITS):
        fit_cand = fit_idx[candidate_mask[fit_idx]]
        val_cand = valid_idx[candidate_mask[valid_idx]]
        assert len(np.unique(y[fit_cand])) == 2
        weights = boundary_sample_weights(STACK_RANK_OOF[fit_cand], alpha)
        ref_oof[val_cand] = fit_refiner(
            train_v9.iloc[fit_cand], y[fit_cand], weights,
            train_v9.iloc[val_cand], CFG.seed + 3000 + fold,
        )
    assert not np.isnan(ref_oof[candidate_mask]).any()
    score = boundary_rank_score(STACK_RANK_OOF, ref_oof, candidate_mask, 0.10, 1.0)
    metrics = metric_row(
        y, score, TIE_KEYS, train_clean['jkpst'], train_clean['umur'],
        f'boundary_alpha_{alpha:g}',
    )
    base_fold_tp = []
    trial_fold_tp = []
    for _, valid_idx in PRIMARY_SPLITS:
        base_fold_tp.append(int(y[valid_idx][top_k_mask(BASELINE[valid_idx], 0.05, TIE_KEYS[valid_idx])].sum()))
        trial_fold_tp.append(int(y[valid_idx][top_k_mask(score[valid_idx], 0.05, TIE_KEYS[valid_idx])].sum()))
    metrics['tp_delta'] = int(metrics['fraud_captured'] - BASE_METRICS['fraud_captured'])
    metrics['nr_delta'] = float(metrics['NormalizedRecall@5%'] - BASE_METRICS['NormalizedRecall@5%'])
    metrics['fold_tp_delta'] = [b - a for a, b in zip(base_fold_tp, trial_fold_tp)]
    metrics['ap_delta'] = float(metrics['AveragePrecision'] - BASE_METRICS['AveragePrecision'])
    metrics['fold_wins'] = int(sum(d > 0 for d in metrics['fold_tp_delta']))
    metrics['worst_fold_delta'] = int(min(metrics['fold_tp_delta']))
    metrics['promote'] = bool(
        metrics['tp_delta'] >= 8 and metrics['nr_delta'] >= 0.001
        and metrics['fold_wins'] >= 3 and metrics['worst_fold_delta'] >= -2
        and metrics['ap_delta'] >= -0.001
    )
    print(
        f"alpha={alpha:g} NR={metrics['NormalizedRecall@5%']:.5f} "
        f"TP={metrics['fraud_captured']} delta={metrics['tp_delta']:+d} "
        f"fold_delta={metrics['fold_tp_delta']}",
        flush=True,
    )
    return score, metrics


results = {}
for alpha in [1, 3, 7]:
    results[alpha] = run_alpha(alpha)

out = ROOT / 'research/experiments/h16-boundary-focused-refiner'
rows = []
for alpha, (_, metrics) in results.items():
    rows.append({
        'alpha': float(alpha),
        'boundary': 0.95,
        'band': 0.02,
        'NormalizedRecall@5%': metrics['NormalizedRecall@5%'],
        'fraud_captured': metrics['fraud_captured'],
        'NR_delta_vs_h10': metrics['nr_delta'],
        'TP_delta_vs_h10': metrics['tp_delta'],
        'AP_delta_vs_h10': metrics['ap_delta'],
        'fold_wins': metrics['fold_wins'],
        'worst_fold_delta': metrics['worst_fold_delta'],
        'promote': metrics['promote'],
        'AveragePrecision': metrics['AveragePrecision'],
        'fold_tp_delta': json.dumps(metrics['fold_tp_delta']),
    })
pd.DataFrame(rows).to_csv(out / 'h16_registry.csv', index=False)
np.savez_compressed(
    out / 'h16_scores.npz',
    **{f'a{alpha:g}': score.astype(np.float32)
       for alpha, (score, _) in results.items()},
)
print('H16 DONE', flush=True)
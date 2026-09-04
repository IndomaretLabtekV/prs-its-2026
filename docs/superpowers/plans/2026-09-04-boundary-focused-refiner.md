# H16 Boundary-Focused Refiner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one bounded, feature-only H16 experiment that trains the existing V9 boundary refiner with sample weights concentrated near the 5% audit boundary and evaluates it on the authoritative five-fold OOF split.

**Architecture:** Reuse the audited H10 setup for data loading, V8 stack reconstruction, fold assignments, V9 feature engineering, and metric calculation. Add only a pure boundary-weight function and pass its output as CatBoost `sample_weight`; keep the candidate pool, rank combination, seeds, and H10 model parameters fixed.

**Tech Stack:** Python 3, pandas, NumPy, CatBoost 1.2.10 GPU, scikit-learn, existing V8/V7 result archives, standard-library JSON/path utilities, and CSV/NPZ artifacts.

---

## Scope and Files

- Create `research/experiments/h16-boundary-focused-refiner/protocol.md` as the execution copy of the approved spec.
- Create `research/experiments/h16-boundary-focused-refiner/run_h16.py` as the only experiment runner.
- Create `research/experiments/h16-boundary-focused-refiner/h16.log`, `h16_registry.csv`, `h16_scores.npz`, and `analysis.md` from the run.
- Modify `research/findings.md`, `research/research-log.md`, and `research/research-state.yaml` only after the saved metrics have been validated.
- Do not modify V7/V8 artifacts, H10 artifacts, the V9 notebook, or any submission file.
- Do not stage or commit changes during execution because git-history mutation was not requested.

### Task 1: Lock the experiment protocol

**Files:**
- Create: `research/experiments/h16-boundary-focused-refiner/protocol.md`

- [ ] **Step 1: Write the fixed protocol before runner execution.** Include the exact fixed values below:

```markdown
# H16 - Boundary-Focused Refiner

## Status

PROTOCOL LOCKED - 2026-09-04

## Fixed setup

- Authoritative five-fold assignments from `results/v8/results.zip`.
- Frozen V8 stack and fold-local `STACK_RANK_OOF`.
- Candidate pool is the frozen top 10% of `STACK_RANK_OOF`.
- Refiner frame is `train_v9` with H10 `V9_FEATURES`.
- CatBoost Logloss, depth 7, 900 iterations, learning rate 0.035, and all
  other H10 parameters unchanged.
- Fold seed is `CFG.seed + 3000 + fold`.
- Control is the saved unweighted H10 V9 OOF score.

For candidate training row `i`, let `r = STACK_RANK_OOF[i]`, `b = 0.95`, and
`d = 0.02`. For each fixed `alpha` in `[1, 3, 7]`, use:

    weight(i, alpha) = 1 + alpha * max(0, 1 - abs(r - b) / d)

The weight is label-independent and is passed only as CatBoost `sample_weight`.
The model uses V9 features only. The H10 boundary rank combination is unchanged.

## Constraints

- No test labels, reconstructed labels, public leaderboard feedback, external
  data, `claim_id`, source row position, or storage-order proxy.
- OOF selection only; no test submission is generated.

## Gate

Promote only an arm with at least +8 pooled TP, NR delta at least +0.001, at
least three fold wins, worst fold delta no worse than -2 TP, and AP delta at
least -0.001 versus the H10 V9 control. Otherwise record H16 as refuted and do
not expand the alpha or band search.
```

- [ ] **Step 2: Check the protocol file for whitespace errors.**

Run:

```bash
git diff --check -- research/experiments/h16-boundary-focused-refiner/protocol.md
```

Expected: no output and exit code 0.

### Task 2: Implement the minimal H16 runner

**Files:**
- Create: `research/experiments/h16-boundary-focused-refiner/run_h16.py`

- [ ] **Step 1: Reuse the H10 bootstrap without changing its data path.** Read and execute `research/experiments/h10-refiner-v2/run_h10.py` only through the line before `ctrl_score, m_ctrl = evaluate(...)`, then use the resulting `y`, `train_v9`, `V9_FEATURES`, `V9_CATS`, `PRIMARY_SPLITS`, `candidate_mask`, `STACK_RANK_OOF`, `metric_row`, `top_k_mask`, `TIE_KEYS`, `train_clean`, and `CFG` objects. Load `h10_scores.npz` key `trt` as the baseline and assert its shape is `(len(y),)`.

- [ ] **Step 2: Add the pure boundary-weight function before model execution.** Use this exact implementation and self-check:

```python
import sys


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
```

Support `--self-check` before the H10 bootstrap so this command exits without loading data:

```bash
conda run --no-capture-output -n datsci python research/experiments/h16-boundary-focused-refiner/run_h16.py --self-check
```

Expected output: `H16_WEIGHT_CHECK_PASS` and exit code 0.

- [ ] **Step 3: Pass weights into the unchanged CatBoost configuration.** Keep the H10 parameters exactly as follows and add only `sample_weight`:

```python
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
```

- [ ] **Step 4: Build each OOF arm from training-fold rows only.** For each `alpha` in `[1, 3, 7]`, use this fold loop:

```python
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
```

Use `boundary_rank_score(STACK_RANK_OOF, ref_oof, candidate_mask, 0.10, 1.0)` for the final score, exactly as H10. Compute pooled metrics and fold-local TP deltas against the H10 V9 baseline. Save one registry row per alpha with the formula parameters, pooled metrics, AP, and JSON-encoded fold deltas.

- [ ] **Step 5: Save only OOF artifacts.** Write:

```python
pd.DataFrame(rows).to_csv(out / 'h16_registry.csv', index=False)
np.savez_compressed(
    out / 'h16_scores.npz',
    **{f'a{alpha:g}': score.astype(np.float32)
       for alpha, (score, _) in results.items()},
)
print('H16 DONE', flush=True)
```

### Task 3: Static and runtime verification

**Files:**
- Verify: `research/experiments/h16-boundary-focused-refiner/run_h16.py`

- [ ] **Step 1: Compile the runner.**

Run:

```bash
conda run --no-capture-output -n datsci python -m py_compile research/experiments/h16-boundary-focused-refiner/run_h16.py
```

Expected: no output and exit code 0.

- [ ] **Step 2: Run all three fixed arms with persistent logging.**

Run:

```bash
conda run --no-capture-output -n datsci python -u research/experiments/h16-boundary-focused-refiner/run_h16.py > research/experiments/h16-boundary-focused-refiner/h16.log 2>&1
```

Expected: the log contains `H16 DONE`; the registry has exactly three rows; the NPZ contains three arrays with shape `(160174,)`.

- [ ] **Step 3: Validate artifact integrity before interpreting metrics.**

Run:

```bash
conda run --no-capture-output -n datsci python -c "import numpy as np, pandas as pd; z=np.load('research/experiments/h16-boundary-focused-refiner/h16_scores.npz'); d=pd.read_csv('research/experiments/h16-boundary-focused-refiner/h16_registry.csv'); assert len(d)==3; assert all(v.shape==(160174,) and np.isfinite(v).all() for v in z.values()); assert d['alpha'].tolist()==[1.0,3.0,7.0]; print('H16_ARTIFACT_CHECK_PASS')"
```

Expected output: `H16_ARTIFACT_CHECK_PASS`.

### Task 4: Evaluate the locked gate and record the result

**Files:**
- Create: `research/experiments/h16-boundary-focused-refiner/analysis.md`
- Modify: `research/findings.md`
- Modify: `research/research-log.md`
- Modify: `research/research-state.yaml`

- [ ] **Step 1: Compute the gate fields from the saved registry.** Parse `fold_tp_delta` with `ast.literal_eval`, count values greater than zero as fold wins, and compute the minimum fold delta. Print one line per alpha with TP delta, NR delta, AP delta versus H10 V9, fold wins, worst fold, and the boolean gate decision. Never use any test-label file in this command.

- [ ] **Step 2: Write `analysis.md`.** Record the exact three rows, sanity checks, gate outcome, and a short interpretation. If every arm fails, state that H16 is refuted and that no alpha/band sweep is authorized. If an arm passes, identify only the passing arm and preserve the other rows as ablations.

- [ ] **Step 3: Update the research narrative.** Add H16 to `closed_hypotheses` in `research-state.yaml`, append one dated entry to `research-log.md`, and update `findings.md` with the measured result. Keep `next` aligned with the gate outcome and retain the existing no-leakage constraints.

- [ ] **Step 4: Run final repository checks.**

Run:

```bash
git diff --check
conda run --no-capture-output -n datsci python -c "import yaml; d=yaml.safe_load(open('research/research-state.yaml')); assert 'H16 boundary-focused refiner' in d['state']['closed_hypotheses']; print('STATE_PARSE_PASS')"
```

Expected: the first command is silent with exit code 0 and the second prints `STATE_PARSE_PASS`.

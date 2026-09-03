# DAC 2026 V8 Robust Boundary Design

## Status

Approved direction: Robust NR@5% with low deployment churn.

Target runtime: Kaggle T4 x2. The notebook must also support a reduced fast smoke mode.

## Problem

V7 stopped after the base models and cross-fitted logistic stack. The boundary refiner, hard-negative specialist, pairwise ranker, promotion gates, and submission cells were not executed. The available artifacts therefore validate the base layer but not the proposed V7 specialist.

The completed V7 stack improved the CatBoost anchor from 7,770 to 7,792 true positives at 5% audit capacity, with positive deltas on both confirmation folds. The remaining risk is not global model capacity. It is stable ordering around a narrow top-5% boundary.

The graph and prior artifacts expose two competing facts:

- location and facility proxies are highly predictive;
- unseen-KDKC stress reduces Technical Score by about 0.19, so the same proxies are fragile under transport.

V8 must improve the boundary without amplifying that fragility or recreating V5-style portfolio churn.

## Goals

1. Reuse valid V7 base artifacts instead of repeating expensive training.
2. Complete a leakage-safe hard-negative boundary experiment.
3. Compare a full-proxy specialist against a proxy-light specialist.
4. Evaluate incremental specialist transport on unseen KDKC and dati2 groups.
5. Promote a challenger only when primary performance, fold stability, stress behavior, and deployment churn all pass predeclared gates.
6. Produce one deterministic primary submission and a complete audit bundle.

## Non-goals

- No public-leaderboard row probing.
- No claim ID or source row position as a predictive feature.
- No broad hyperparameter search.
- No new global deep-tabular model.
- No pairwise ranker in V8. V7 did not produce evidence that its extra complexity is useful.
- No demographic post-processing to alter selection by protected group.

## Evidence Baseline

### Data profile

- Train: 160,174 rows, 53 columns.
- Test: 40,043 rows, 52 columns.
- Label counts: 80,203 fraud and 79,971 legitimate.
- Exact cleaned-feature duplicate signatures: 17,906 rows in 6,747 repeated groups.
- Primary validation must keep each exact signature in one fold.

### V7 completed evidence

| Candidate | NR@5% | TP@5% | AP |
|---|---:|---:|---:|
| CatBoost anchor | 0.970280 | 7,770 | 0.823793 |
| Cross-fitted logistic stack | 0.973027 | 7,792 | 0.825861 |

Stack fold deltas versus the anchor are positive on folds 1, 2, 4, and 5. Both confirmation folds are positive.

### Graph-derived risks

- Unseen-KDKC stress: about -0.19 Technical Score.
- Location proxy removal: score falls from about 0.92 to 0.82-0.83.
- Top-5% boundary score spread across roughly 500 nearby ranks: about 0.013.
- Blend, seed-mean, calibration, and protected-removal ablations are near zero.

These results favor a small boundary intervention with explicit transport checks.

## Notebook Deliverable

Create `notebooks/V8_Robust_Boundary.ipynb`.

The notebook must be standalone on Kaggle. It may reuse artifacts when available, but absence of the artifact bundle must not make the notebook unusable.

### Runtime modes

- Full mode: Kaggle T4 x2, five primary folds, five KDKC stress folds, and five dati2 stress folds.
- Fast mode: two primary folds, two KDKC stress folds, two dati2 stress folds, reduced iterations, and one model seed. It verifies behavior only and must never be reported as competition evidence.

## Architecture

### 1. Configuration and reproducibility

A frozen configuration object owns:

- seeds and fold counts;
- artifact reuse policy;
- base and refiner iterations;
- candidate pool rate;
- action bands and mixing weights;
- proxy-light exclusion rules;
- promotion gates;
- output paths.

All random libraries receive explicit seeds. The selected configuration and environment metadata are saved.

Full-mode defaults preserve the validated V7 base:

- CatBoost base iterations: `2860`;
- XGBoost and LightGBM estimators: `2600`;
- logistic maximum iterations: `700`;
- V3 boundary-refiner iterations: `900`;
- hard-negative specialist iterations: `700`;
- specialist candidate pool: top `10%`;
- hard-negative legitimate-class weight: `12.0`;
- inner target-encoding folds: `3`;
- primary and stress folds: `5`;
- full-data specialist seeds: `2026`, `2127`, and `3407`.

Fast mode uses `180` base iterations or estimators, `120` refiner and specialist iterations, two outer folds, two inner folds, and seed `2026` only.

### 2. Data discovery and contracts

Discover only schema-consistent `train.csv`, `test.csv`, and `sample_submission.csv` roles. Reject paths containing `ground_truth`. Preserve semantic categorical strings and align test rows exactly to the sample submission.

Required assertions:

- unique train and test claim IDs;
- binary train labels;
- exact train/test feature-schema parity;
- exact sample-submission ID coverage and order;
- finite output probabilities in `[0, 1]`.

### 3. Deterministic feature package

Reuse the V7 cleaning and feature engineering contracts. Claim ID remains alignment-only.

Create two explicit feature views:

- `FULL_FEATURES`: the validated V7 representation;
- `PROXY_LIGHT_FEATURES`: the same representation after removing location/facility identifiers and derived interactions.

The proxy-light exclusion set includes direct or derived forms of:

- `kdkc`;
- `dati2`;
- `typeppk`;
- `geo__facility`;
- `service__facility`;
- `kdkc__dati2`;
- `diagnosis__facility`;
- `cmg__facility`;
- frequency or target-estimate columns derived from those fields.

Retained proxy-light evidence includes diagnosis, CMG, severity, service type, age, length of stay, diagnosis/procedure burden, model probabilities, ranks, and model disagreement.

An assertion must prove that no forbidden proxy token remains in the proxy-light view.

### 4. Artifact-first base layer

Search Kaggle inputs and local project paths for `results.zip` or an extracted `v7_artifacts` directory.

If artifacts are present:

1. validate the expected file set;
2. validate row counts, finite values, fold coverage, and model names;
3. validate feature-column metadata;
4. load base OOF/test predictions, fold IDs, hierarchy estimates, and meta features;
5. reconstruct the logistic stack deterministically.

If artifacts are absent, run the V7 base pipeline and save the same artifact contract. If artifacts are present but invalid, fail loudly instead of silently retraining.

Expected reusable files:

- base prediction NPZ files;
- `primary_fold_assignments.csv`;
- `hierarchical_target_estimates.npz`;
- `meta_features.npz`;
- base and stack metric CSV files.

### 5. Frozen controls

Reconstruct and freeze:

1. CatBoost anchor;
2. cross-fitted logistic stack;
3. V3 top-10% CatBoost boundary control.

All V8 candidates rerank only within a declared action band. Scores outside the band must remain bitwise unchanged.

### 6. Hard-negative specialists

The target is `1` for a legitimate claim inside the high-risk candidate pool and `0` for fraud. Specialists learn which baseline-positive claims should move down.

Train two cross-fitted CatBoost specialists:

- `HN full`: uses `FULL_FEATURES` and the existing full-proxy conditional encodings;
- `HN proxy-light`: uses `PROXY_LIGHT_FEATURES` and clinical-only conditional encodings.

Proxy-light conditional encodings may use combinations of diagnosis, CMG, severity, and service type. They must not use KDKC, dati2, facility type, or derivatives.

Training-candidate encodings are inner-cross-fitted. Outer-validation encodings use only outer-training candidates. Test encodings use the full training candidate pool.

Class weighting addresses the rare legitimate class inside the high-risk pool. Do not combine class weighting with resampling by default.

### 7. Surgical reranking

Use two action bands:

- locked top 4.0%, outer boundary 6.5%;
- locked top 3.5%, outer boundary 7.0%.

Use mixing weights `0.35`, `0.60`, and `0.85`. Weight `1.0` is excluded because it discards the proven baseline order and increases churn risk.

Within each band, mix baseline-local rank and specialist-local rank, then map the mixed rank back into the original score interval. Outside the band, preserve the frozen V3 score exactly.

Candidate matrix:

- V3 control;
- stack control;
- six `HN full` challengers;
- six `HN proxy-light` challengers.

No adaptive grid expansion is allowed.

### 8. Validation suites

#### Primary validation

Use five-fold `StratifiedGroupKFold` with the exact cleaned-feature signature as the group. Save fold assignments. Assert zero signature overlap.

Folds 1-3 are selection folds. Folds 4-5 are confirmation folds. Confirmation labels influence only predeclared gate evaluation, not grid expansion.

#### Specialist transport stress

Run separate grouped stress evaluations with:

- `groups = kdkc`;
- `groups = dati2`.

Each stress validation category must be absent from its specialist training partition. The frozen V3 OOF score remains the common control. The stress suite measures the incremental reranker transport, not absolute global-model transport. The report must state this limitation.

Stress selection compares each challenger with V3 on the identical held-out rows and top-5% capacity.

### 9. Metrics

For every candidate, record:

- Normalized Recall@5%;
- true positives and legitimate audits at 5%;
- Average Precision;
- Protection score and subgroup gaps;
- fold-level TP deltas;
- confirmation-fold gain;
- worst-fold loss;
- one-way swaps, overlap, and churn;
- TP gain per 100 swaps;
- test churn against V3;
- KDKC and dati2 stress deltas.

Calibration is fitted only after ranking selection. A monotone sigmoid converts the final score to probabilities without changing order.

## Promotion Gates

A challenger is eligible only when all conditions hold:

1. Primary NR@5% gain versus V3 is at least `0.001`, approximately eight TPs.
2. At least three of five primary folds improve.
3. Total confirmation-fold TP gain is nonnegative.
4. Worst primary-fold TP delta is at least `-2`.
5. Average Precision delta is at least `-0.001`.
6. OOF one-way top-5% churn is at most `0.08`.
7. TP gain per 100 swaps is at least `1.0`.
8. Test churn is at most `min(0.10, 1.25 * OOF churn + 0.015)`.
9. KDKC stress delta versus V3 is nonnegative.
10. dati2 stress delta versus V3 is nonnegative.

Among eligible challengers, select by:

1. highest minimum stress delta;
2. highest confirmation-fold TP gain;
3. highest primary TP gain;
4. highest TP gain per 100 swaps;
5. lowest test churn;
6. lowest complexity, preferring proxy-light on an exact tie.

If no challenger passes, select V3 control and explain every failed gate. Falling back to the control is a valid outcome, not an execution failure.

## Outputs

Write to `/kaggle/working/v8_artifacts` or a local temporary equivalent:

- `artifact_manifest.json`;
- `base_model_summary.csv`;
- `stack_comparison.csv`;
- `v8_candidate_registry.csv`;
- `v8_primary_fold_deltas.csv`;
- `v8_stress_metrics.csv`;
- `v8_test_churn.csv`;
- `v8_scores.npz`;
- `selected_config.json`;
- `submission_v8.csv`.

`selected_config.json` records the selected candidate, all promotion metrics, artifact provenance, official-data-only status, and explicit confirmation that claim ID and row position were not predictive features.

Only one primary submission is emitted. V8 does not generate family challenger submissions.

## Failure Handling

- Missing artifacts: run the deterministic base fallback.
- Invalid present artifacts: stop with the failed contract; do not silently retrain.
- GPU failure: retry CatBoost on GPU 0 with reduced CTR complexity, matching the existing repository pattern.
- Missing LightGBM in full mode: install the declared notebook dependency or fail before training. Fast mode may omit it only when the reduced roster is recorded and every fast-mode result remains labeled non-comparable.
- Empty or single-class candidate pools: reject that fold/candidate with an explicit diagnostic.
- No eligible challenger: select V3 control.

## Behavioral Verification

The notebook must verify:

1. metric and label-neutral top-k boundary cases;
2. exact-signature group disjointness;
3. KDKC and dati2 stress-group disjointness;
4. artifact shape, model-name, fold, and finite-value contracts;
5. absence of forbidden proxy columns in the proxy-light specialist;
6. inner and outer target-encoding completeness;
7. surgical reranking leaves scores outside the action band unchanged;
8. candidate registry contains every predeclared candidate exactly once;
9. promotion is deterministic and falls back to V3 when no challenger passes;
10. submission schema, row order, uniqueness, range, and finiteness.

Fast mode must execute the end-to-end path through artifact loading or reduced fallback training, candidate evaluation, deterministic selection, and submission writing. Fast-mode metrics must be labeled non-comparable.

## Implementation Ownership

The coding implementation will be delegated to the available GPT-5.6 task worker. The parent agent retains architecture decisions, reviews the produced notebook, runs structural and behavioral verification, and integrates fixes. The harness does not expose an agent alias named `GPT 5.6 Luna Max`; no false model attribution will be made.

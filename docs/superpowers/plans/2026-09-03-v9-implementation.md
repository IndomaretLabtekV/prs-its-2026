# DAC 2026 V9 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a standalone Kaggle-compatible V9 notebook (`notebooks/V9_Representation.ipynb`) that reuses the validated V8 checkpoint, trains exactly 3 new CatBoost arms (seed-3407, 2× interaction-featured), evaluates 4 predeclared candidates against the frozen V3 control under V8 gates (minus stress gates 9-10), and emits one audited submission. Full runtime target: under ~1 hour on Kaggle T4 x2.

**Architecture:** Clone the V8 notebook structure and DELETE the specialist/rerank/stress sections; ADD the V9 feature block (`engineer_v9`), the 3-arm training roster, the 4-row registry, and the fold-2/4 diagnostic. Reuse verbatim: data discovery/contracts, cleaning, V7 feature engineering, grouped folds, artifact validation, metric/gate/selection/calibration/submission code, behavioral asserts.

**Tech Stack:** Python 3, Jupyter notebook JSON, pandas/numpy, scikit-learn, CatBoost (Logloss only), standard-library zip/json/path. No XGBoost/LightGBM training (not needed; drop the V8 LightGBM hard requirement since nothing in V9 trains it).

## Global Constraints

- Candidate roster is FIXED at 4 entries (C0–C3 per spec). No grid expansion, no extra seeds, no extra weights.
- V3 control OOF/test must match checkpoint bitwise; assert before any training.
- V8's 10 gate thresholds and selection cascade are copied exactly, control = V3 control.
- Claim ID alignment-only; row position never predictive; no `ground_truth` paths; no LB probing.
- Mixed-signature flag is leakage-quarantined (§3 spec); interaction columns derive from allowed columns only.
- Only one primary submission (`submission_v9.csv`); fallback C0 re-emit is a valid outcome.
- Existing V7/V8 files and artifacts remain untouched.

---

### Task 1: Notebook scaffold from V8

**Files:**
- Create: `notebooks/V9_Representation.ipynb`

**Interfaces:**
- Produces frozen `Config` (V9: seeds 2127/2026/3407, 5 folds, cat 2860 iter, same CatBoost hyperparams), env metadata, output dir, deterministic seeds, Kaggle/local path discovery.

- [ ] Copy V8 cells 01 (imports minus XGB/LGBM requirement — keep optional-import pattern, drop the full-mode raise), 03 (metrics/contract utils verbatim), 05 (discovery/clean/V7 features verbatim).
- [ ] Copy V8 cell 07 fold-signature block verbatim but REPLACE locally-computed splits with checkpoint assignments when `ARTIFACT_FOLD_MATCHES_CURRENT` (keep computation + assert as validation, use artifact IDs for training).
- [ ] Copy V8 cell 09 artifact-validation block verbatim (V8 checkpoint + V7 fallback paths); assert V3-control OOF equals `selected_oof` from `v8_scores.npz` bitwise.
- [ ] Verify: scaffold parses as JSON, behavioral asserts 1–4 pass on local `data/` in fast smoke (2-fold override allowed ONLY for smoke, labeled non-comparable).

### Task 2: V9 features + leakage quarantine

**Files:** `notebooks/V9_Representation.ipynb`

**Interfaces:**
- Produces `engineer_v9(frame)` → adds `risk_subgroup`, `risk_subgroup__age_band`, `risk_subgroup__los_band`, `risk_subgroup__burden_band`; produces `V9_FEATURES = FULL_FEATURES + new`.

- [ ] Implement `risk_subgroup` (KM / KDKC2101_2103 / DATI2_309 / OTHER) + three crosses with age_band (reuse `age_groups`), los bins (0/1-2/3+), burden tertiles (computed on train, applied to test).
- [ ] Write `v9_feature_audit.csv`: coverage per new column, univariate fraud rate per level, and a chi-square vs label (report-only).
- [ ] Asserts: new columns from allowed columns only; no ID/row derivation; proxy-light rules untouched (V9 does not use proxy-light view); all finite.

### Task 3: 3-arm training roster

**Files:** `notebooks/V9_Representation.ipynb`

**Interfaces:**
- Produces OOF/test predictions for C1 (seed-3407, FULL_FEATURES), C2/C3 (seeds 2127/2026, V9_FEATURES); C0 loaded frozen.

- [ ] Reuse `fit_cat` verbatim (Logloss, hyperparams V7; sample_weight hook retained but unused — all arms unweighted).
- [ ] C1–C3 train on artifact folds (5-fold), 2860 iters, depth 8, lr 0.03, GPU with V8 retry pattern; save per-arm NPZs (`predictions_v9_<name>.npz` with oof/test/test_folds).
- [ ] Verify: C-arm OOF shapes finite; C1 pooled NR reported next to V4 cat_mean expectation (+3 TP scale); training log records per-fold wall time.

### Task 4: Registry, gates, selection, submission

**Files:** `notebooks/V9_Representation.ipynb`

**Interfaces:**
- Produces `v9_candidate_registry.csv` (4 rows, V8 schema minus gate_9/10), `v9_primary_fold_deltas.csv`, `v9_test_churn.csv`, `selected_config.json`, `v9_scores.npz`, `submission_v9.csv`, `artifact_manifest.json`.

- [ ] Copy V8 metric/gate/selection/calibration/submission cells verbatim with `expected_candidate_names()` replaced by the 5 V9 names and control = V3 control.
- [ ] Monotone sigmoid post-ranking only; assert ordering unchanged.
- [ ] Submission contract checks (schema, sample order/IDs, uniqueness, [0,1], finite) + deterministic-selection check; fallback C0 path exercised in code.
- [ ] Fold-2/fold-4 diagnostic section: per-fold covariate profile + per-candidate fold NR table (report-only, no gating).

### Targeted Verification

- [ ] Parse notebook JSON; syntax-compile concatenated code cells.
- [ ] Fast smoke on local `data/`: 2-fold, 180 iter, all 4 candidates evaluated, registry cardinality 4, gates deterministic, submission contract green, outputs labeled non-comparable.
- [ ] Confirm V3-control bitwise match assert passes against local `results/v8`.
- [ ] Do not run formatters, linters, project-wide suites, or Git mutations.

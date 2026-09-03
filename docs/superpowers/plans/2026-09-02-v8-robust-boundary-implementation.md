# DAC 2026 V8 Robust Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a standalone Kaggle-compatible V8 notebook that reuses or deterministically rebuilds V7 base evidence, evaluates leakage-safe full and proxy-light hard-negative rerankers under primary and transport stress validation, applies every promotion gate, and emits one audited submission.

**Architecture:** A single notebook follows the established V7 pipeline: discovery and semantic cleaning, engineered FULL and PROXY_LIGHT views, exact-signature grouped folds, artifact-first base/stack reconstruction with deterministic fallback, frozen V3 boundary control, nested target encodings, two CatBoost hard-negative specialists, fixed surgical candidate grid, primary/stress metrics, deterministic gates and selection, monotone sigmoid calibration, and artifact/submission writing. Fast mode uses the same functions and contracts with reduced folds/iterations and explicitly non-comparable labeling.

**Tech Stack:** Python 3, Jupyter notebook JSON, pandas/numpy, scikit-learn, CatBoost, XGBoost, optional LightGBM, scipy, standard-library zip/json/path utilities.

## Global Constraints

- Full mode defaults: 5 primary, 5 KDKC stress, 5 dati2 stress folds; CatBoost 2860, XGBoost/LightGBM 2600, logistic 700, refiner 900, specialist 700 iterations.
- Fast mode: 2 primary and 2 stress folds, 180 base, 120 refiner/specialist iterations, 2 inner folds, seed 2026 only; all results are labeled non-comparable.
- Claim ID is alignment-only and row position is never a predictive feature.
- Primary groups are exact cleaned-feature signatures with zero overlap; stress groups are KDKC and dati2 and must be absent from specialist training partitions.
- No pairwise ranker, public leaderboard probing, demographic post-processing, broad search, or new global deep-tabular model.
- Existing V7 files/artifacts remain untouched; invalid present artifacts fail loudly and missing artifacts use deterministic fallback training.
- Only the fixed two action bands, weights 0.35/0.60/0.85, and V3/stack/full/proxy-light candidate roster are evaluated.
- GPU failures follow V7 fallback to GPU 0 with reduced CTR complexity; missing LightGBM fails full mode before training and may be omitted only in labeled fast mode.

---

### Task 1: Notebook scaffold and reproducibility

**Files:**
- Create: `notebooks/V8_Robust_Boundary.ipynb`

**Interfaces:**
- Produces frozen `Config`, environment metadata, output directory, deterministic random seeds, and portable Kaggle/local path discovery.
- Uses notebook cells with short professional markdown and no em-dash or emoji.

- [ ] Add imports, dependency checks, `Config` defaults/fast override, GPU setup, and save selected config/environment metadata.
- [ ] Add explicit dependency failure behavior and CatBoost GPU retry pattern from V7.
- [ ] Add mode labels that mark fast outputs as non-comparable.

### Task 2: Data contracts and feature views

**Files:** `notebooks/V8_Robust_Boundary.ipynb`

**Interfaces:**
- Produces `train_raw`, `test_raw`, `sample_submission`, `train_model`, `test_model`, `FULL_FEATURES`, `PROXY_LIGHT_FEATURES`, `TIE_KEYS`, and labels.

- [ ] Discover only schema-consistent train/test/sample CSV roles and reject `ground_truth` paths.
- [ ] Enforce unique IDs, binary labels, schema parity, sample coverage/order, and finite/range checks.
- [ ] Reuse V7 cleaning/feature engineering, excluding claim ID and target from predictors.
- [ ] Remove direct/derived location/facility columns and frequency/target-estimate variants from proxy-light; assert forbidden tokens absent while retaining clinical and model-evidence columns.

### Task 3: Grouped folds and artifact-first base layer

**Files:** `notebooks/V8_Robust_Boundary.ipynb`

**Interfaces:**
- Produces primary fold IDs/splits, stress splits, fold-safe hierarchy estimates, base predictions, meta features, and logistic stack.
- Writes/validates `primary_fold_assignments.csv`, base prediction NPZs, hierarchy/meta NPZs, and summaries.

- [ ] Build exact cleaned signatures and primary `StratifiedGroupKFold`; assert disjointness and save assignments.
- [ ] Search local/Kaggle paths and ZIP members for V7 artifacts, validate file set, shapes, finite values, folds, names, and metadata.
- [ ] Load valid artifacts and reconstruct stack deterministically; otherwise run a deterministic reduced CatBoost fallback roster with fixed seeds and record the reduced roster.
- [ ] Build fold-safe target estimates and stack meta features without ID or row-position inputs.

### Task 4: Frozen controls and hard-negative specialists

**Files:** `notebooks/V8_Robust_Boundary.ipynb`

**Interfaces:**
- Produces frozen `V3_OOF_SCORE`/`V3_TEST_SCORE`, `STACK_OOF`, and specialist outputs for full and proxy-light views.

- [ ] Reconstruct V3 top-10% CatBoost boundary control and preserve scores outside action bands.
- [ ] Build full/proxy-light hard-negative target encodings with inner cross-fitting for training candidates, outer-training-only encodings for validation, and full-candidate encodings for test.
- [ ] Reject empty or single-class candidate pools with explicit diagnostics.
- [ ] Fit cross-fitted CatBoost specialists with rare legitimate class weight 12.0, GPU fallback, no resampling, and no pairwise ranker.

### Task 5: Fixed surgical candidate matrix and validation

**Files:** `notebooks/V8_Robust_Boundary.ipynb`

**Interfaces:**
- Produces exactly 14 registry entries: V3 control, stack control, six HN full, six HN proxy-light. Writes `v8_candidate_registry.csv`, fold deltas, stress metrics, and test churn.

- [ ] Implement two action bands (locked 4.0/outer 6.5 and locked 3.5/outer 7.0), three weights (0.35/0.60/0.85), local-rank mixing, and bitwise outside-band preservation.
- [ ] Evaluate primary 5-fold selection/confirmation roles and separate KDKC/dati2 grouped stress suites against V3 on identical held-out rows at 5% capacity.
- [ ] Record NR@5%, TP, legitimate audits, AP, protection/gaps, fold deltas, confirmation gain, worst loss, swaps/overlap/churn, gain per 100 swaps, test churn, and both stress deltas.
- [ ] Include executable behavioral checks for metric-neutral top-k ties, group disjointness, artifact contracts, proxy exclusions, encoding completeness, surgical invariance, registry uniqueness, and stress group exclusion.

### Task 6: Promotion, calibration, and audited submission

**Files:** `notebooks/V8_Robust_Boundary.ipynb`

**Interfaces:**
- Produces `selected_config.json`, `v8_scores.npz`, `submission_v8.csv`, and `artifact_manifest.json` in the configured output directory.

- [ ] Implement all ten gates exactly: NR gain >= .001, >=3 fold wins, nonnegative confirmation gain, worst delta >= -2, AP delta >= -.001, OOF churn <= .08, gain/100 swaps >=1, test churn <= min(.10, 1.25 OOF + .015), nonnegative KDKC stress, nonnegative dati2 stress.
- [ ] Select eligible challenger by minimum stress, confirmation gain, primary gain, swap efficiency, test churn, then complexity with proxy-light exact-tie preference; fall back to V3 and report failed gates if none pass.
- [ ] Fit monotone sigmoid calibration only after ranking selection and assert ordering unchanged.
- [ ] Write one submission only with exact sample order/IDs, uniqueness, finite [0,1] probabilities, all required audit files, provenance, official-data-only status, and explicit ID/row-position exclusion.
- [ ] Execute deterministic-selection and submission-contract checks in notebook outputs.

### Targeted Verification

- [ ] Parse notebook JSON and execute every code cell in a controlled kernel or `nbclient` where dependencies permit.
- [ ] Run Python syntax compilation on concatenated code cells.
- [ ] Run the fast end-to-end path on local `data/` and verify output files, registry cardinality, ten gates, ten behavioral assertions, and non-comparable labels.
- [ ] Do not run formatters, linters, project-wide suites, or Git mutations.

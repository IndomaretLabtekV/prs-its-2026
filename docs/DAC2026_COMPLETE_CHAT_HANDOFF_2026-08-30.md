# DAC 2026 — Complete Project and Conversation Handoff

**Cut-off date:** 30 August 2026, Asia/Jakarta  
**Workspace:** `/home/rian/Projects/prs`  
**Purpose:** transfer the complete recoverable analytical state to another chat without restarting the project, repeating completed research, or confusing historical V1 evidence with executed V2 evidence.

---

## Navigation

- Sections 1–6: current status, original objective, chronology, evidence method, official metric, and literature research.
- Sections 7–12: dataset forensics, complete V1 architecture/results, verified faults, verified strengths, and score bottlenecks.
- Sections 13–21: V2 hypotheses, architecture, executed experiments, audit allocation, error analysis, features, fairness, explainability, and robustness.
- Sections 22–25: rejected directions, what to apply now, evidence-status boundaries, and answered research questions.
- Sections 26–31: filesystem/artifact state, exact continuation instructions, operations, limitations, bibliography, and final recovery checklist.

---

## 1. Executive handoff

The project has progressed beyond the initial V1 notebook.

1. The complete V1 notebook was authored, run on Kaggle T4 x2, and produced a user-reported Public Leaderboard score of **0.95833**. The prompt-time Top-1 score was **0.98000**, a public gap of **0.02167**. This is only a coarse external signal; it cannot identify erroneous claims or justify row-level leaderboard probing.
2. V1 artifacts were recovered and preserved locally under `results/v1/`.
3. A forensic audit found genuine V1 evaluation and deployment-design weaknesses, especially source-row tie breaking, cross-fold calibration rank distortion, a validation/deployment blend mismatch, arbitrary ordinal encoding in XGBoost, and full-refit versus fold-average portfolio divergence.
4. A complete V2 notebook was subsequently built and **executed successfully on Kaggle**. The executed file is `notebooks/DAC2026_V2_Ultimate.ipynb`.
5. V2 selected **CatBoost V2 seed 2127**, using fold-averaged raw test predictions. Its corrected five-fold OOF metrics are:

   - Normalized Recall@5%: **0.97053**
   - Average Precision: **0.82318**
   - Brier score: **0.172383**
   - Protection Score: **0.998847**
   - Technical Score: **0.92246**
   - Fold-level Technical Score SD: **0.00273**

6. At the 5% OOF audit budget, V2 selected 8,008 claims, captured 7,772 observed fraud labels, and placed 236 legitimate claims in the audit portfolio.
7. The scalar-highest blend reached about **0.9230**, but it failed the predeclared confirmation/stability gate. V2 therefore correctly retained the simpler single-seed CatBoost as primary and generated no secondary submission.
8. The successful V2 Kaggle output has now been copied locally to `results/v2/`, including `submission_primary.csv`, model OOF/test prediction NPZ files, selected predictions, registries, diagnostics, and figures. The local primary submission has been independently checked against `data/sample_submission.csv`: exactly 40,043 aligned unique IDs, the exact two required columns, and finite probabilities in `[0,1]`.
9. No V2 leaderboard score has been supplied in this conversation. It is not valid to claim that V2 improves the public or private leaderboard until its generated primary submission is evaluated.

The most important interpretive warning is this:

> V2's reported **+0.01905** gain is versus a **corrected V1-like anchor** evaluated inside the V2 framework, not versus the historical V1 artifact score. The historical V1 CatBoost OOF Technical Score was already about 0.9222–0.9224. Those two comparisons use different evaluation/training recipes and must not be mixed.

---

## 2. What the user originally asked for

The original assignment was not a generic Kaggle baseline. It required one self-contained, competition-grade analytical notebook that could serve both as:

- a strong Kaggle submission pipeline; and
- the primary evidence base for the official DAC 2026 preliminary-round answer sheet.

The requested quality dimensions were technical performance, statistical rigor, defensibility, interpretability, policyholder protection, operational usefulness, reproducibility, scientific writing, credible literature, and exact compatibility with the official answer-sheet structure.

The hard report order was:

1. `CHAPTER I: Introduction`
2. `CHAPTER II: Theoretical Framework`
3. `CHAPTER III: Analytical Steps`
4. `CHAPTER IV: Analysis of Results`
5. `CHAPTER V: Conclusion and Recommendation`
6. `REFERENCES`
7. `ATTACHMENT`

Both V1 and V2 preserve this order. V2 contains 79 cells: 41 Markdown cells and 38 code cells. Every code cell was executed in the recovered Kaggle output.

The original source hierarchy was also a hard constraint:

1. Official DAC 2026 competition specification and rules.
2. Official 2026 answer-sheet template.
3. The three supplied previous-work references as structural and presentational few-shot examples.
4. The actual competition data as empirical truth.
5. External scholarly literature and official technical documentation for theory, method, and governance.

Whenever sources could conflict, higher-priority material controls the work. Previous-year submissions were never treated as authority over the 2026 rules.

The non-negotiable implementation requirements carried through the project were:

- cover all competition Tasks A–E, not just modeling;
- implement the official metric exactly and test edge cases;
- investigate data quality, duplicates, implausible values, leakage, proxies, and shift;
- determine validation from the data-generating structure before choosing folds;
- produce genuine OOF predictions and retain fold-level evidence;
- compare progressive baselines and strong tabular models on one primary framework;
- evaluate 3%, 5%, and 7% audit capacity;
- handle calibration without fitting and scoring on the same labels;
- treat gender and official age-band protection as a first-class objective;
- provide global and local explanation without causal claims;
- test robustness, feature ablations, seeds, calibration, blends, capacity, and location transport;
- generate report prose from runtime results instead of fabricating numbers;
- retain a central deterministic configuration and sequential execution;
- use Kaggle-compatible paths and detect one or two GPUs safely;
- write and strictly validate the final probability submission;
- frame the model as human decision support, never automated rejection;
- include governance, monitoring, drift, recalibration, retraining, override, escalation, and appeal;
- preserve honest limitations and avoid public-leaderboard overfitting;
- run expensive training on Kaggle, not the local machine.

---

## 3. Conversation and project chronology

### 3.1 Initial authoring phase

The work began by inspecting:

- `docs/ANSWER SHEET PRELIMINARY ROUND.docx`
- `ref/Notebook_InfoMagangDong (Penyisihan).ipynb`
- `ref/Answer Sheets_sayonarasrama [-1].pdf`
- `ref/Answer Sheet_InfoMagangDong.docx`
- the train, test, and sample-submission files
- the competition rules and scoring definitions

The structural map recovered from those references was:

- a formal problem background leading to explicit research questions;
- theory and equations tied directly to later analytical methods;
- a methods chapter with a clear end-to-end flow;
- results organized as evidence, interpretation, limitation, and decision implication rather than a code log;
- chapter-numbered tables and figures;
- explicit answers to research questions in the conclusion;
- supporting detail moved to attachments instead of cluttering the main argument.

The initial implementation plan was persisted at:

`docs/superpowers/plans/2026-08-28-dac2026-notebook.md`

### 3.2 V1 creation and first Kaggle run

The V1 deliverable was created at:

`notebooks/DAC2026_Ultimate_Baseline_T4x2.ipynb`

It was designed for Kaggle T4 x2, with expensive model work deferred to Kaggle. It was later run successfully. The current local V1 notebook is the executed output, not the original clean authoring copy.

V1 Kaggle run metadata:

- Start: `2026-08-28T10:36:13.815072+00:00`
- End: `2026-08-28T11:37:42.484655+00:00`
- Duration: **3,688.67 seconds** or about **61 minutes 29 seconds**
- Papermill exception: none
- Current SHA-256: `29f17ff1c2e4c885b415c64c7286c7e201cdde9724f625bcea937acf69d54629`
- Cells: 72 total, 34 code, all 34 executed

The first submitted probabilities earned a Public Leaderboard score reported as approximately 0.958.

### 3.3 Recovery after interruption

When the earlier run was interrupted by the local machine becoming unresponsive, recovery instructions required the filesystem to be treated as source of truth. The repository was inspected instead of rebuilding from scratch.

The V1 artifacts were found outside the project and copied non-destructively into:

`results/v1/`

That preserved the executed evidence needed for re-analysis without retraining.

### 3.4 Post-first-run diagnosis and V2 development

The post-run assignment explicitly prohibited leaderboard probing and demanded an evidence-led audit of metrics, validation, predictions, calibration, features, ensembles, fairness, robustness, and inference strategy.

Cheap artifact analysis was performed first. This uncovered the source-order tie defect and several V1 architecture mismatches. Those findings informed a new V2 notebook rather than a cosmetic rewrite of V1.

The V2 deliverable is:

`notebooks/DAC2026_V2_Ultimate.ipynb`

V2 Kaggle run metadata:

- Start: `2026-08-29T15:08:07.618009+00:00`
- End: `2026-08-29T16:18:44.412766+00:00`
- Duration: **4,236.79 seconds** or about **70 minutes 37 seconds**
- Papermill exception: none
- Current SHA-256: `b8bf76d5fa2e09ff9be9a226b987787cb315b157166ba7803c1f6d469ca97051`
- Cells: 79 total, 38 code, all 38 executed
- Runtime observed two Tesla T4 GPUs, each with 15,360 MiB
- CatBoost devices: `0:1`
- XGBoost device: `cuda`
- LightGBM available: yes

The V2 notebook completed all runtime completeness checks and reported creation of a validated primary submission. No evidence-qualified secondary was produced.

### 3.5 First V2 failure, root-cause repair, and successful rerun

The first full V2 Kaggle T4 x2 attempt completed almost the entire expensive model phase before failing in downstream selection. It had already run:

- metric sanity checks;
- burden baseline;
- corrected V1-like anchor;
- logistic V2;
- CatBoost V2 seeds 2026 and 2127;
- XGBoost V2;
- LightGBM V2.

The fatal line attempted to use `EXPERIMENT_REGISTRY.prebootstrap_eligible` as a Boolean mask:

```python
eligible_pool = EXPERIMENT_REGISTRY[
    EXPERIMENT_REGISTRY.prebootstrap_eligible
].copy()
```

Pandas raised:

```text
ValueError: Cannot mask with non-boolean array containing NA / NaN values
```

The root cause was not merely “Pandas needs `fillna(False)`.” The registry lifecycle was traced end to end:

1. `EXPERIMENT_REGISTRY` was created from every candidate, including `Burden baseline`.
2. `selection_pool` intentionally removed `Burden baseline` because it was a transparent diagnostic benchmark rather than a final competition candidate.
3. `eligibility_rows` was mistakenly created by iterating only over `selection_pool`.
4. The eligibility table therefore had no row for `Burden baseline`.
5. A left merge attached that incomplete table back to the complete experiment registry.
6. The omitted burden row received `NaN` for `prebootstrap_eligible` and related eligibility fields.
7. The Boolean mask then failed after roughly 54 minutes of expensive work.

The missing state meant **explicitly ineligible diagnostic baseline**, not unknown, not unevaluated, and not a valid fallback. The source contract was repaired accordingly:

- eligibility is now constructed for every registry row;
- `Burden baseline` receives explicit native-Boolean `False` and a written exclusion reason;
- the one-to-one eligibility attachment rejects missing or extra experiment IDs;
- `prebootstrap_eligible`, `bootstrap_eligible`, and `selected` must be complete native-Boolean columns;
- no `.fillna(False)` symptom patch is used.

A reusable experiment-registry validator was added. Before any decision mask is consumed, it checks:

- non-empty registry and required columns;
- complete, unique experiment IDs and index;
- known model/candidate types;
- native Boolean decision columns with no `None`, `NaN`, or `pd.NA`;
- finite metric columns and valid `[0,1]` ranges where applicable;
- exact consistency of `CalibrationScore = 1 - Brier`;
- exact consistency of the official weighted Technical Score;
- existence, dimensionality, length, finiteness, and bounds of every eligible OOF/test probability vector;
- calibration method validity, calibrated-vector alignment, and fitted deployment calibrator availability.

An early preflight now runs before the burden baseline or any boosted model. It validates configuration, folds, feature sets, experiment IDs, a rehearsed complete eligibility merge, calibration policy, fixed-round policy, model libraries, sample-submission schema, test/sample order, output paths, bootstrap configuration, and Kaggle GPU visibility.

Downstream path coverage was reviewed for:

- multiple eligible candidates;
- exactly one eligible candidate;
- no conservative survivor despite valid scored singles;
- ensemble rejection;
- ensemble acceptance;
- no justified secondary;
- justified secondary.

An empty candidate pool is now treated in two different ways:

- missing metrics, predictions, IDs, or eligibility is corrupted state and raises a clear error;
- a complete valid registry whose conservative gates reject everybody uses the strongest valid single candidate under the documented robust policy and discloses that methodological fallback.

The repair also added high-value checkpoints under `/kaggle/working/v2_artifacts/`, including partial registries, fold metrics, blend and calibration tables, per-model prediction NPZ files, and selected predictions. The primary submission is written and validated before optional SHAP and transport diagnostics.

Non-fatal warning paths were cleaned without changing the official evaluation:

- CatBoost GPU-unsupported per-iteration PRAUC/Brier custom metrics were removed from the actual fixed-round training path; official metrics remain independently calculated from OOF predictions.
- XGBoost prediction uses explicit native `Booster.predict(DMatrix(...))`, avoiding the sklearn inplace-prediction CPU/GPU device mismatch.
- LightGBM prediction uses native `Booster.predict` on the same sparse representation used for fitting, avoiding feature-name mismatch warnings.

Before rerun, the clean notebook passed all nine targeted V2 contract tests, JSON parsing, code compilation, and Pyflakes. No CatBoost, XGBoost, LightGBM, CV, SHAP, or full notebook execution was performed locally. Static-repair hashes at that point were:

```text
clean repaired V2 notebook    30f315cada49603836e0f2e43ff04c7dbfa53ffba01a763ecd19889602954c62
V2 notebook builder           376f470ac193909b42b4c68e6ab3b1bd0503711a5a0a797dee8b6f490f043111
V2 contract tests             0a10e4b819d87f2ebd0ba679c1885424e6b730f4a05abc64e39818719e34794f
```

The repaired notebook was then rerun by the user on Kaggle T4 x2 and completed successfully. The executed notebook that replaced the clean upload copy has SHA-256 `b8bf76d5fa2e09ff9be9a226b987787cb315b157166ba7803c1f6d469ca97051`; the metrics reported in Sections 14–21 come from that successful execution. The builder scripts are not present in the current workspace, so these historical hashes are provenance, not pointers to currently available files.

### 3.6 Current handoff phase

The present request is to create a complete Markdown transfer document so another chat can continue from the exact current state. This file is that handoff. It is not a claim that a further V3 or another Kaggle submission has already been run.

---

## 4. Why the analysis is able to answer the competition this way

This section gives the verifiable reasoning trace, not private hidden deliberation.

### 4.1 Evidence layers

Every important answer is grounded in one or more of four evidence layers:

1. **Official rules:** determine the target, audit budget, metric formulas, group eligibility, output schema, and decision-support boundary.
2. **Executed data evidence:** shapes, rates, model metrics, calibration, fairness, uncertainty, shift, and error patterns come from actual runtime calculations.
3. **Methodological literature:** supports the use and limitations of precision-recall evaluation, grouped validation, probability calibration, boosting, fairness metrics, bootstrap uncertainty, explainability, human oversight, and drift monitoring.
4. **Operational interpretation:** converts model evidence into investigator workflow while explicitly separating association, prediction, and causality.

### 4.2 No fabricated empirical claims

Dataset-specific prose in the notebooks is generated from executed values through `display(Markdown(...))` and runtime registries. Static report text contains context, definitions, equations, and methodological rationale, but not invented dataset results.

The main runtime objects include report facts, metric dictionaries, table registries, figure registries, prediction registries, calibration results, fold records, experiment registries, hypothesis registries, and artifact manifests.

### 4.3 Competition integrity boundary

The project did not:

- search for hidden test labels;
- row-match evaluation claims against external labeled data;
- use claim IDs as predictive features;
- infer individual labels from leaderboard changes;
- manually alter test probabilities;
- use public leaderboard probing as a model-selection method.

Claim IDs are used only for alignment and as a deterministic label-neutral secondary key when exact score ties cross an audit boundary.

### 4.4 Why the answer is not just “pick the highest CV score”

The Technical Score combines four distinct properties. A tiny scalar gain can come from noise, a favorable tie, a calibration plateau, one unstable fold, or a fairness change. Selection therefore also considers:

- fold stability;
- selection-fold versus confirmation-fold behavior;
- boundary tie sensitivity;
- paired cluster-bootstrap uncertainty;
- deployable equivalence between OOF and test formulas;
- top-5 overlap and prediction diversity;
- model complexity and operational interpretability;
- location transport stress;
- fairness and policyholder exposure.

This is why the V2 notebook did not select the numerically largest blend automatically.

---

## 5. Official competition objective and exact metric

For every evaluation claim, the submission contains:

```text
claim_id
fraud_probability
```

with `fraud_probability` finite and inside `[0, 1]`.

For `N` claims:

```text
K = floor(0.05 * N)
```

Only the top `K` claims form the official audit portfolio. The score is decision support for human investigation, not an automated rejection, denial, or allegation.

Let `P` be the number of observed fraud claims and `TP_K` the observed fraud claims in the top-K portfolio:

```text
Recall@5%     = TP_K / P
Recall_max    = min(1, K / P)
NormalizedRecall@5% = Recall@5% / Recall_max
```

Probability quality:

```text
Brier = mean((p_i - y_i)^2)
CalibrationScore = 1 - Brier
```

For each eligible protected group with at least 100 legitimate observations:

```text
FPR_g = legitimate claims from group g selected for audit
        -------------------------------------------------
        all legitimate claims from group g
```

Age groups are exactly:

- 0–17
- 18–39
- 40–59
- 60+

The official gaps and protection score are:

```text
Gap_gender = max(FPR_gender) - min(FPR_gender)
Gap_age    = max(FPR_age) - min(FPR_age)
ProtectionScore = 1 - max(Gap_gender, Gap_age)
```

The official composite is:

```text
TechnicalScore =
    0.55 * NormalizedRecall@5%
  + 0.20 * AveragePrecision
  + 0.15 * CalibrationScore
  + 0.10 * ProtectionScore
```

V2 independently checks no-fraud behavior, `K=0`, invalid probabilities, exact ties, permutation-invariant key-based tie resolution, constant probabilities, missing groups, under-100 group exclusion, and age-boundary assignments.

---

## 6. Research and literature protocol

### 6.1 Research method

Literature was searched and verified before finalizing the conceptual chapters and recommendations. Source priority was:

1. original methodology papers;
2. peer-reviewed journals and conferences;
3. respected books;
4. WHO, OECD, NIST, and similarly authoritative institutions;
5. official model documentation for implementation details.

For every retained source, title, author, year, venue or publisher, DOI or stable URL, and the claim it supports were checked. Low-quality blogs, marketing summaries, and unverifiable pages were not used.

### 6.2 What the research supports

- Healthcare-fraud analytics and its data/label limitations: WHO, Joudaki et al., du Preez et al.
- Precision-recall evaluation under selective or imbalanced action: Davis and Goadrich; Saito and Rehmsmeier.
- Probability calibration and Brier scoring: Brier; Van Calster et al.; Kull et al.; Niculescu-Mizil and Caruana.
- Leakage and structured validation: Kaufman et al.; Roberts et al.; Cawley and Talbot.
- Gradient boosting and tabular models: Friedman; CatBoost; XGBoost; LightGBM.
- Precision-at-top methods: Kar et al., used to evaluate but ultimately reject a dedicated ranking objective for this dataset.
- Fairness and differential error exposure: Hardt et al.; Rajkomar et al.
- Explainability and its limits: Lundberg and Lee; Kumar et al.; Amann et al.
- Human oversight and governance: WHO ethics guidance; NIST AI RMF; Kelly et al.
- Drift and monitoring: Gama et al.
- Uncertainty: Efron and Tibshirani; Wilson.
- GPU implementation: official CatBoost and XGBoost documentation.

### 6.3 Why literature was used but external data were not

External literature justifies methods and interpretations. It was never used to obtain labels, target proxies unavailable at scoring, or row-specific information. This preserves competition integrity.

---

## 7. Dataset forensics

### 7.1 Schema and target

| Item | Result |
|---|---:|
| Train shape | 160,174 rows × 53 columns |
| Test shape | 40,043 rows × 52 columns |
| Sample submission | 40,043 rows × 2 columns |
| Training fraud prevalence | 50.0724% |
| Train duplicate IDs | 0 |
| Test duplicate IDs | 0 |
| Train/test ID overlap | 0 |
| Missing cells in raw data | 0 |

The approximately balanced target is likely a curated competition sample. It must not be described as the operational fraud base rate. This is why class weights, oversampling, and SMOTE were not applied automatically.

### 7.2 Storage-order defect

The label column has only **one transition** in stored row order: a contiguous fraud block followed by a contiguous legitimate block. Row order is therefore unsafe for:

- predictive features;
- deterministic top-k tie breaking;
- any model training behavior that could preserve source order without deliberate shuffling.

V2 shuffles each training fold with a seeded permutation before fitting and never uses source position as a feature or tie key.

### 7.3 Data types and semantic codes

V1 initially let pandas infer dtypes before converting semantic category fields. V2 instead reads administrative code fields as strings from the CSV boundary, preserving leading zeros.

Executed V2 evidence found:

- 46,210 `kdkc` values begin with zero;
- all 160,174 `dati2` values begin with zero.

This is a data-contract correction even though the isolated two-fold code-hierarchy feature gain was inconclusive.

### 7.4 Duplicate and template structure

| Check | Count |
|---|---:|
| Exact duplicate train rows | 0 |
| Repeated exact-feature signature groups | 6,747 |
| Rows in repeated exact-feature signatures | 17,906 |
| Repeated signatures with conflicting labels | 1,737 |
| Exact test rows matching a train signature | 0 |
| Test rows matching a reduced clinical template | 31,410 |

Consequences:

- exact signatures are grouped into one fold to avoid obvious template leakage;
- conflicting labels prove that a repeated template is not a deterministic fraud rule;
- the large reduced-template overlap can be legitimate shared structure, but it remains a possible dependency not fully isolated by exact-signature grouping;
- no hidden labels are inferred from train/test template overlap.

### 7.5 Missing, invalid, and unusual values

- No raw missing cells.
- Trailing whitespace existed in `typeppk` and was stripped.
- Age range: 0 to 109; no age below 0 or above 120.
- Length of stay range: 0 to 592.
- Ten claims have length of stay above 365 and are review flags, not silently deleted observations.
- No negative grouped diagnosis/procedure counts.
- No non-integer grouped diagnosis/procedure counts.
- Constant columns excluded: `dx2_koo_k93`, `dx2_u00_u99`, `procv00_v89`; V2's derived `missing_field_count` is also non-informative because the raw dataset has no missing cells.

### 7.6 Cardinality and unseen values

Important cardinalities:

- `dati2`: 486 levels, including 35 rare levels with fewer than 20 rows.
- `kdkc`: 126 levels.
- `los`: 128 values; test has 14 numeric values not observed in train.
- `umur`: 105 values.
- `typeppk`: 25 levels.
- `cmg`: 23 levels.
- `diagprimer`: 21 levels.

The executed categorical shift audit found **zero unseen semantic categorical levels in test**. Numeric values outside the set seen in train are not the same as unseen categorical levels.

### 7.7 Observed associations

Strongest categorical Cramér's V values after rare-level handling:

| Feature | Cramér's V |
|---|---:|
| `dati2` | 0.2983 |
| `kdkc` | 0.2509 |
| `typeppk` | 0.1229 |
| `cmg` | 0.1122 |
| `severitylevel` | 0.0777 |
| `jnspelsep` | 0.0533 |
| `diagprimer` | 0.0194 |
| `jkpst` | 0.0114 |

Strongest numeric rank-biserial effect:

- `los`: absolute effect 0.0757.

Most other numeric effects are very small despite extremely small p-values. Large sample size makes statistical significance easy to obtain; effect size and held-out predictive utility are more important.

Fold-safe single-feature AP was highest for:

- `dati2`: 0.6546
- `kdkc`: 0.6276
- `typeppk`: 0.5525
- `cmg`: 0.5426
- `severitylevel`: 0.5218
- `los`: 0.5213

No raw feature crossed the notebook's extreme suspicious-signal rule. Strong location signal was classified as a structural proxy risk, not automatically as target leakage.

### 7.8 Protected-group label prevalence

Observed training rates, which are descriptive and non-causal:

| Group | Claims | Fraud rate |
|---|---:|---:|
| Gender `L` | 74,284 | 0.5068 |
| Gender `P` | 85,890 | 0.4954 |
| Age 0–17 | 40,125 | 0.4926 |
| Age 18–39 | 41,391 | 0.4978 |
| Age 40–59 | 47,844 | 0.5021 |
| Age 60+ | 30,814 | 0.5131 |

These differences do not establish discrimination, medical causality, or behavioral intent.

### 7.9 Train/test shift

- Cross-validated adversarial validation AUC: **0.5068**, near chance.
- Largest univariate shift: `proc39_45`, magnitude **0.0133** by the notebook's combined scale.
- Total unseen semantic categorical levels: **0**.
- Important raw features generally have small measured train/test shifts.

Known conclusion: measured covariate shift is not the main demonstrated limitation of V1 or V2.

Unknown: hidden-label prevalence and concept stability cannot be inferred from unlabeled test features.

---

## 8. V1 analytical architecture

### 8.1 Validation

V1 used five-fold `StratifiedGroupKFold`, grouping by the cleaned full-feature signature. This was a strong design choice because exact repeated templates remained within one fold.

Ordinary stratification was retained only as an optimism diagnostic. V2 later quantified that ordinary folds would create 13,092 cross-fold repeated signature groups.

`kdkc`-held-out validation was used as a separate deployment robustness view, not as the primary Kaggle model-selection view.

### 8.2 V1 feature set

V1 excluded `claim_id` and constant features. It added 15 static engineered features:

- age band;
- pediatric indicator;
- older-adult indicator;
- `log1p_los`;
- diagnosis total, nonzero group count, and maximum group;
- procedure total, nonzero group count, and maximum group;
- total coded burden;
- log coded burden;
- `cmg__severity`;
- `service__facility`;
- `geo__facility`.

### 8.3 V1 model ladder

1. Fold-local prevalence baseline.
2. Low-dimensional burden logistic baseline.
3. Full regularized one-hot logistic regression.
4. CatBoost with native categoricals.
5. XGBoost with ordinal-coded categories.
6. Cross-fitted calibration variants.
7. A conservative CatBoost/XGBoost blend screen.

The progressive ladder was used to show how much each level of sophistication added. Accuracy was not used as the decision metric.

### 8.4 V1 main model settings

CatBoost:

- up to 2,200 iterations;
- learning rate 0.035;
- depth 8;
- L2 leaf regularization 6;
- random strength 0.5;
- Bayesian bootstrap / bagging temperature 0.8;
- early stopping patience 180;
- GPU devices `0:1`.

XGBoost:

- up to 2,200 trees;
- learning rate 0.035;
- max depth 7;
- min child weight 5;
- subsample 0.85;
- column sample 0.82;
- alpha 0.05;
- lambda 6;
- CUDA histogram training.

Class weighting was conditional on minority prevalence below 20%; it did not activate because the training labels were nearly balanced.

### 8.5 V1 calibration and selection

V1 compared raw probabilities, sigmoid/Platt scaling, and isotonic regression using cross-fitted OOF calibration. It attempted to avoid calibration leakage.

The V1 blend screen used CatBoost weights `{0, 0.25, 0.5, 0.75, 1}` and required:

- at least 0.001 global Technical Score gain;
- at least half the folds won;
- correlation below 0.9995.

Models within 0.002 were intended to favor simplicity.

### 8.6 V1 final submission recipe

The submitted V1 probabilities did **not** use fold-averaged test predictions.

The exact path was:

1. train CatBoost OOF predictions;
2. select CatBoost plus sigmoid calibration;
3. fit one sigmoid calibrator on all raw CatBoost OOF predictions;
4. refit a single CatBoost on all train rows using the median fold best iteration;
5. apply the global sigmoid to the full-refit test probabilities;
6. write `/kaggle/working/submission.csv`.

Fold-average CatBoost and XGBoost test predictions were preserved in NPZ artifacts but were not submitted.

---

## 9. V1 executed results

### 9.1 Main candidates

| Candidate | NR@5% | AP | Brier | Protection | Technical | Fold SD | Approx. runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| CatBoost raw | 0.970155 | 0.823402 | 0.172186 | 0.999234 | 0.922361 | 0.003172 | 8.34 min |
| XGBoost raw | 0.952672 | 0.796771 | 0.183310 | 0.997763 | 0.905604 | 0.003951 | 0.82 min |
| Logistic raw | 0.882368 | 0.732823 | 0.205602 | 0.995375 | 0.850564 | 0.006094 | 9.65 min |
| Burden baseline | 0.602023 | 0.524635 | 0.248661 | 0.990693 | 0.647810 | — | — |

CatBoost raw captured 7,769 observed fraud labels in `K=8,008` OOF rows.

### 9.2 V1 calibrated decision table

| Candidate | NR@5% | AP | Brier | Protection | Technical | Selected? |
|---|---:|---:|---:|---:|---:|---|
| Cat/XGB blend, cross-fitted sigmoid | 0.971404 | 0.823904 | 0.171756 | 0.998893 | 0.923179 | No |
| CatBoost, cross-fitted sigmoid | 0.969905 | 0.823401 | 0.172180 | 0.999333 | 0.922234 | Yes |
| XGBoost calibrated | — | — | — | — | 0.905657 | No |
| Logistic calibrated | — | — | — | — | 0.850562 | No |

The blend gain over calibrated CatBoost was **0.000945**, only **0.000055** below the predeclared 0.001 threshold. This was too small to treat as robust evidence.

### 9.3 V1 audit allocation

CatBoost:

| Capacity | K | Fraud captured | Precision/normalized recall | Legitimate audits |
|---|---:|---:|---:|---:|
| 3% | 4,805 | 4,722 | 0.982726 | 83 |
| 5% | 8,008 | 7,767 after cross-fitted sigmoid | 0.969905 | 241 |
| 7% | 11,212 | 10,739 | 0.957813 | 473 |

The burden baseline's apparent 3% precision of 1.0 was later shown to be a tie/order artifact, not real predictive perfection.

### 9.4 V1 ablations

| Variant | Technical Score | Delta versus full CatBoost sigmoid | Interpretation |
|---|---:|---:|---|
| Full CatBoost sigmoid | 0.922234 | 0 | Reference |
| Protected attributes removed | 0.921837 | -0.000398 | Removing direct protected fields did not help |
| Location proxies removed | 0.826883 | -0.095351 | Location/facility signals carry very large in-distribution signal |
| Engineered features removed | 0.919728 | -0.002507 | V1 engineering adds modest useful signal |

Protected-field removal also slightly worsened Protection Score. This confirms that removing a protected column does not guarantee reduced disparity because proxy information and case mix remain.

### 9.5 V1 uncertainty

Conditional row-stratified fixed-prediction bootstrap intervals:

| Metric | Estimate | 95% interval |
|---|---:|---:|
| Technical Score | 0.922234 | [0.919977, 0.924381] |
| NR@5% | 0.969905 | [0.966409, 0.973277] |
| AP | 0.823401 | [0.820977, 0.825625] |
| Brier | 0.172180 | [0.171196, 0.173213] |
| Protection | 0.999333 | [0.998091, 0.999566] |

These intervals condition on fixed OOF predictions. They do not refit the learner and may understate model-selection uncertainty or row dependence.

### 9.6 V1 fairness

At 5%:

- gender FPR gap: approximately 0.000181;
- age FPR gap: approximately 0.000667;
- Protection Score: 0.999333.

Fairness was nearly saturated and was not the main score bottleneck.

### 9.7 V1 model diversity

OOF comparisons:

| Pair | Pearson | Spearman | Top-5 overlap |
|---|---:|---:|---:|
| CatBoost vs XGBoost | 0.89317 | 0.89306 | 0.67657 |
| CatBoost vs Logistic | 0.77437 | 0.77271 | 0.49963 |
| XGBoost vs Logistic | 0.78969 | 0.78390 | 0.52747 |

This proved that high-level model rankings were complementary enough to justify blend research, while not proving that blending would generalize.

---

## 10. What was wrong or suboptimal in V1

### 10.1 Source-row tie breaking was invalid

V1 used stable sorting for exact probability ties. Because all fraud labels were stored before legitimate labels, the original row order provided a favorable tie breaker.

Independent audit examples:

| Prediction | Source-order TP | Label-neutral/hash TP | Expected TP under random tie resolution |
|---|---:|---:|---:|
| Constant 0.5 | 8,008 | about 4,032–4,045 depending neutral key | 4,009.80 |
| V1 burden score | 4,821 | 4,264 | 4,261.21 |

The V1 fold-prevalence baseline therefore appeared to have NR@5%=1.0 and Technical Score around 0.862 despite containing no ranking information. That result was invalid as a predictive benchmark.

V2 fixes the shared selector: if a tie crosses the boundary, a label-neutral claim key is required. It also reports expected, optimistic, and pessimistic boundary capture.

### 10.2 Serious models were not materially driven by tie luck

The defect was severe for constant and low-resolution baselines, but small for serious models:

- V1 CatBoost raw boundary tie size was 3, with one remaining boundary slot and two fraud labels inside the tie.
- Pessimistic capture was 7,768, optimistic 7,769, expected 7,768.67.
- XGBoost and logistic had boundary tie size 1.
- Stable and neutral-key CatBoost selection both captured 7,769 in the audited run.

Therefore:

- the metric implementation required correction;
- the strong CatBoost result was not explained away by boundary ties;
- the trivial baseline narrative was materially contaminated.

### 10.3 Cross-fitted monotone calibration changed global OOF ranking

A single sigmoid map is monotone and should preserve ranking. V1, however, fitted a different sigmoid on each calibration fold, then concatenated the fold-specific outputs. Different maps rescaled rows differently across folds, so the global OOF order changed.

For V1 CatBoost:

- raw OOF captured 7,769 fraud labels at 5%;
- cross-fitted sigmoid captured 7,767;
- a single global sigmoid applied to all OOF scores preserved the raw rank and captured 7,769.

The cross-fitted sigmoid Brier gain was negligible. V1's selected calibrated Technical Score, 0.922234, was slightly below raw CatBoost's 0.922361.

V2 separates the quantities correctly:

- ranking terms are calculated from one common raw OOF score;
- calibration Brier is estimated cross-fitted;
- a single deployment map is fitted only after method selection;
- isotonic remains diagnostic because plateaus create boundary ambiguity.

### 10.4 Isotonic calibration created unsafe plateaus

In the V1 forensic audit, cross-fitted isotonic reduced probability uniqueness to 728 values and created a boundary tie of 360 rows. Depending on tie allocation, captured fraud could vary roughly from 7,755 to 7,775.

In V2, selected-model isotonic produced only 141 unique deployment probabilities and a boundary tie of 896. Its pessimistic NR@5% fell to 0.9683. It was therefore ineligible for final selection even when Brier looked competitive.

### 10.5 V1 blend evaluation and deployment formulas differed

V1 allowed fold-varying blend weights for OOF evaluation but deployed one median blend weight. The validation object and submission object were therefore not exactly the same function.

V2 evaluates fixed-form probability, logit, or rank blends using the same formula intended for test deployment.

### 10.6 XGBoost category encoding was arbitrary

V1 XGBoost represented categories as ordinal integers. Tree splits can treat those arbitrary codes as ordered quantities.

V2 replaces this with fold-fitted one-hot representation for both XGBoost and LightGBM. CatBoost retains native categorical treatment.

### 10.7 CatBoost capacity appeared saturated

All five V1 CatBoost folds stopped at roughly 2,195–2,200 of the 2,200-tree cap. This suggested that the cap, not a stable validation optimum, was constraining learning.

V2 therefore ran a small capacity/regularization screen and selected a 2,860-round depth-8 configuration. This was hypothesis-driven, not a massive parameter search.

### 10.8 Full-data refit and fold averaging produced different audit portfolios

V1's full-refit and fold-average CatBoost test vectors had:

- Pearson correlation: about 0.9940;
- Spearman correlation: about 0.9941;
- top-5 overlap: **0.9166**;
- 167 claims swapped on each side of the top-5 boundary.

High global correlation therefore did not imply operational equivalence. Because no test labels exist, the better test recipe is unknown. V2 uses fold averaging because that is the inference recipe directly represented by CV, while documenting that this remains an assumption.

### 10.9 The 0.001 blend gate was too brittle as a truth claim

V1's blend missed the gate by 0.000055. That does not make the blend scientifically worse; it means the observed gain was too small to distinguish from noise using that rule.

V2 replaces the one-number gate with fold wins, confirmation folds, tie safety, overlap, complexity, and paired uncertainty. It still rejects weak improvements.

### 10.10 Public leaderboard evidence could not identify the cause

The V1 public score of 0.95833 versus the prompt-time Top-1 score of 0.98000 is one sample-level comparison with a 0.02167 gap. It cannot reveal:

- which claims were wrong;
- whether full refit beat fold averaging;
- whether calibration matched hidden prevalence;
- whether public labels were easier;
- whether public/private distributions differ;
- whether the score gap is ranking, calibration, fairness, or sample variance.

Any claim assigning the leaderboard gap to one mechanism without hidden labels would be speculation.

---

## 11. What was not wrong in V1

- CatBoost was a genuinely strong model, not merely a tie artifact.
- Grouping exact repeated signatures was defensible and prevented verified cross-fold template overlap.
- The 5% metric, AP, Brier, and fairness components were implemented in the intended overall form.
- `claim_id` was excluded from prediction.
- OOF and fold-average test predictions were preserved, enabling cheap forensic re-analysis.
- Fairness was analyzed using the official eligibility rule and careful disparity language.
- Train/test covariate shift was small; there is no evidence that a large obvious feature shift explains V1.
- Engineered burden and interaction features added real, though modest, OOF value.
- The blend was correctly not promoted on weak evidence, even though the original cutoff was overly crisp.
- The solution respected the decision-support boundary and did not use external labels.

---

## 12. Main score bottlenecks after V1

Approximate remaining weighted headroom at the V1 CatBoost result:

| Component | Observed | Weight | Approx. weighted headroom to 1 |
|---|---:|---:|---:|
| NR@5% | 0.970 | 0.55 | 0.0166 |
| AP | 0.823 | 0.20 | 0.0353 |
| Calibration score | 0.828 | 0.15 | 0.0258 |
| Protection | 0.999 | 0.10 | negligible |

Interpretation:

- fairness was nearly saturated;
- AP and probability quality had more total theoretical headroom;
- the top-5 boundary remained operationally critical because it carries 55% of the official weight;
- the best next work was not demographic score manipulation, but better generalizable ranking/representation and a calibration protocol that did not corrupt ranking.

---

## 13. V2 hypotheses and final status

| ID | Hypothesis | Final status | Evidence |
|---|---|---|---|
| H01 | Label-neutral ties remove source-order optimism | **SUPPORTED** | Constant and burden tie sensitivity |
| H02 | Preserving semantic code strings and hierarchy improves representation | **INCONCLUSIVE** | Data-contract correction verified; isolated hierarchy contribution +0.00042 on screen |
| H03 | Longer regularized CatBoost improves boundary ranking | **SUPPORTED** | Depth-8 capacity screen won; 2,860 fixed rounds used |
| H04 | Density, concentration, rarity, and structured interactions recover near-boundary fraud | **INCONCLUSIVE** | F3/F4/F5 screen-positive; family evidence only two folds; full package improved corrected anchor |
| H05 | Two-seed CatBoost mean reduces ranking variance enough to select | **INCONCLUSIVE** | Gain over best seed only +0.00018 |
| H06 | A fixed cross-family blend contains complementary signal | **SUPPORTED**, but not selected | Best blend gain about +0.00081 in the screening definition; confirmation behavior was insufficient |
| H07 | Fold averaging is safer than an unvalidated full refit | **INCONCLUSIVE** | It is validation-aligned, but test labels cannot rank the two recipes |
| H08 | Location signal is useful in distribution but weakly transportable | **SUPPORTED** | Unseen-`kdkc` stress delta -0.19025; location removal also very costly |
| H09 | Monotone calibration materially improves selected CatBoost Brier | **REJECTED** | Raw won; improvement +0.000000 at displayed precision |
| H10 | Dedicated learning-to-rank is worth added complexity | **REJECTED** | No query groups; probability/fairness/deployment complexity outweighed evidence |

---

## 14. V2 analytical and modeling architecture

### 14.1 Runtime and reproducibility

Core configuration:

```text
seed = 2026
folds = 5
capacities = 3%, 5%, 7%
CatBoost screen cap = 1,400
CatBoost full cap = 3,200
selected CatBoost fixed rounds = 2,860
XGBoost estimators = 2,600
LightGBM estimators = 2,600
early-stopping control = 220
bootstrap repetitions = 300
SHAP sample = 700
minimum eligible non-fraud group = 100
```

Runtime versions:

- Python 3.12.13
- pandas 2.3.3
- scikit-learn 1.6.1
- CatBoost 1.2.10
- XGBoost 3.2.0
- LightGBM 4.6.0

### 14.2 Corrected top-k implementation

V2 requires a label-neutral tie key when an exact tie crosses the boundary. Selection uses descending score and claim-key secondary order. Diagnostics separately compute:

- strictly-above-boundary rows;
- tie size;
- remaining slots;
- fraud inside the tie;
- expected fraud captured under random allocation within the tie;
- optimistic and pessimistic capture.

This makes the official exact-K portfolio deterministic without pretending that the organizer's hidden tie policy is known.

### 14.3 Validation design

Primary validation is five-fold `StratifiedGroupKFold` by exact cleaned feature signature.

| Fold | Train rows | Validation rows | Validation fraud rate | Signature overlap | Role |
|---:|---:|---:|---:|---:|---|
| 1 | 128,102 | 32,072 | 0.5010 | 0 | selection |
| 2 | 128,179 | 31,995 | 0.5032 | 0 | selection |
| 3 | 128,176 | 31,998 | 0.5038 | 0 | selection |
| 4 | 128,207 | 31,967 | 0.5004 | 0 | confirmation |
| 5 | 128,032 | 32,142 | 0.4953 | 0 | confirmation |

The first three folds guide screens/meta-selection and the last two provide confirmation perspectives. This is not fully nested external validation; all base-model predictions remain standard five-fold OOF. The notebook says this explicitly.

A separate unseen-`kdkc` holdout of 32,877 rows is used only as a transport stress test.

### 14.4 Feature representation

The corrected V1-like set contains 63 features. V2 has 81 pre-frequency features and six possible fold-safe frequency features.

V2 adds or formalizes:

- semantic code strings and two-character code prefixes;
- zero-length-of-stay indicator;
- age × log-LOS interaction;
- diagnosis and procedure entropy;
- maximum burden share;
- total nonzero groups;
- procedure-to-diagnosis ratio;
- burden, procedure, and diagnosis per day;
- `kdkc__dati2`;
- diagnosis × CMG;
- diagnosis × service;
- diagnosis × facility;
- CMG × facility;
- fold-local frequency maps for `kdkc`, `dati2`, `geo__facility`, `kdkc__dati2`, `diagnosis__facility`, and `diagnosis__cmg`.

Every frequency map is learned on the outer training fold and applied to validation/test, preventing target or full-data leakage.

### 14.5 Why each model exists

**Burden logistic baseline.** A transparent low-dimensional benchmark for whether simple workload/count logic is enough. It is not eligible for final selection because coarse ties make its top-k result unstable.

**Full logistic regression.** Provides an interpretable regularized linear baseline after one-hot preprocessing. It shows whether nonlinearity and interactions add value.

**CatBoost.** Chosen as the anchor because the data contain meaningful heterogeneous categories and interactions. CatBoost handles native categorical variables without arbitrary numeric ordering and was strongest in V1.

**XGBoost.** Included as a complementary regularized tree family. V2 uses fold-fitted one-hot categories to remove V1's ordinal-code artifact.

**LightGBM.** Added only because it provides a genuinely different leaf-wise boosting bias at modest cost. It became the strongest ensemble complement, though not a robust enough winner.

**No deep neural network.** Two T4 GPUs are compute resources, not evidence that a neural network is appropriate. The dataset is medium-sized structured tabular data, and boosted trees already provide strong performance and clearer governance.

**No large Optuna search.** The experiment budget is hypothesis-led to reduce selection overfit.

**No SMOTE or automatic class weighting.** The competition sample is already nearly balanced.

**No direct ranking model.** There are no query groups; dedicated precision-at-top objectives would add calibration and deployment risk without existing evidence of superiority.

### 14.6 V2 model settings

Selected CatBoost depth-8 capacity configuration:

```text
iterations = 2860
learning_rate = 0.03
depth = 8
l2_leaf_reg = 8.0
random_strength = 0.35
bagging_temperature = 0.5
max_ctr_complexity = 2
border_count = 128
fixed outer-fold rounds = true
```

XGBoost one-hot:

```text
n_estimators = 2600
learning_rate = 0.035
max_depth = 7
min_child_weight = 6
subsample = 0.85
colsample_bytree = 0.82
reg_alpha = 0.08
reg_lambda = 7.0
fixed outer-fold rounds = true
```

LightGBM one-hot:

```text
n_estimators = 2600
learning_rate = 0.025
num_leaves = 48
max_depth = -1
min_child_samples = 60
subsample = 0.85
colsample_bytree = 0.82
reg_alpha = 0.10
reg_lambda = 7.0
fixed outer-fold rounds = true
```

### 14.7 Calibration design

Compared methods:

- raw;
- monotone sigmoid;
- monotone beta calibration;
- isotonic diagnostic.

Selection rules:

- Brier evidence is cross-fitted by outer fold;
- ranking metrics use the common raw order;
- sigmoid and beta are strictly monotone;
- isotonic is never eligible for final rank selection because of plateaus;
- one global deployment map is fitted on all OOF predictions only after method choice.

### 14.8 Ensemble design

V2 evaluates fixed probability, logit, and rank blends of the CatBoost seed mean with XGBoost or LightGBM across a compact CatBoost-weight grid from 0.4 to 0.9.

Selection safeguards include:

- maximum tie penalty 0.0005;
- minimum ensemble gain 0.00075;
- at least three total fold wins;
- at least one confirmation-fold win;
- component top-5 overlap below 0.985;
- positive paired cluster-bootstrap lower bound for a complex winner;
- within 0.0005, prefer the simpler candidate.

### 14.9 Final inference

V2's primary is the mean of five validated fold-model raw test predictions for CatBoost seed 2127. The selected calibration is raw, so no nontrivial transformation is applied.

The notebook validates:

- exact row count;
- exact column order;
- no missing or non-finite probabilities;
- probabilities inside `[0,1]`;
- no duplicate IDs;
- exact sample-submission ID order.

---

## 15. V2 executed experiment results

### 15.1 Fold results

Selected-model and main-family fold Technical Scores:

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---:|---:|---:|---:|---:|
| Corrected V1-like anchor | 0.90320 | 0.90827 | 0.90026 | 0.90287 | 0.90205 |
| Logistic V2 | 0.86262 | 0.86858 | 0.85643 | 0.85483 | 0.85936 |
| CatBoost V2 seed 2026 | 0.92045 | 0.92253 | 0.91967 | 0.92313 | 0.92070 |
| CatBoost V2 seed 2127 | 0.92115 | 0.92472 | 0.91980 | 0.92625 | 0.92113 |
| XGBoost V2 one-hot | 0.90260 | 0.90793 | 0.90012 | 0.90523 | 0.89718 |
| LightGBM V2 one-hot | 0.91195 | 0.91860 | 0.90866 | 0.91065 | 0.90832 |

The improvement is not concentrated in one isolated fold, although seed 2127's strongest fold is fold 4.

### 15.2 Corrected experiment registry

| Candidate | NR@5% | AP | Brier | Protection | Technical | Fold SD | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Cat mean + LightGBM logit, Cat weight 0.9 | 0.9712 | 0.8239 | 0.1719 | 0.9989 | 0.9230 | 0.0017 | Scalar best; failed confirmation gate |
| CatBoost two-seed mean | 0.9705 | 0.8239 | 0.1720 | 0.9987 | 0.9226 | 0.0020 | Inconclusive gain |
| Cat mean + XGBoost logit, Cat weight 0.9 | 0.9705 | 0.8237 | 0.1720 | 0.9990 | 0.9226 | 0.0015 | Failed gate |
| Cat mean + LightGBM probability, Cat weight 0.9 | 0.9704 | 0.8238 | 0.1719 | 0.9989 | 0.9226 | 0.0018 | Failed gate |
| Cat mean + LightGBM rank, Cat weight 0.9 | 0.9704 | 0.8238 | 0.1719 | 0.9990 | 0.9226 | 0.0017 | Failed gate |
| **CatBoost V2 seed 2127** | **0.9705** | **0.8232** | **0.1724** | **0.9988** | **0.9225** | **0.0027** | **Selected** |
| Cat mean + XGBoost probability, Cat weight 0.9 | 0.9698 | 0.8235 | 0.1720 | 0.9989 | 0.9222 | 0.0015 | Failed gate |
| Cat mean + XGBoost rank, Cat weight 0.9 | 0.9697 | 0.8235 | 0.1720 | 0.9989 | 0.9221 | 0.0015 | Failed gate |
| CatBoost V2 seed 2026 | 0.9680 | 0.8231 | 0.1722 | 0.9989 | 0.9211 | 0.0015 | Not selected |
| LightGBM V2 one-hot | 0.9587 | 0.8037 | 0.1798 | 0.9980 | 0.9108 | 0.0042 | Not selected |
| Corrected V1-like anchor | 0.9519 | 0.7899 | 0.1858 | 0.9973 | 0.9034 | 0.0030 | Comparison anchor |
| XGBoost V2 one-hot | 0.9484 | 0.7950 | 0.1832 | 0.9968 | 0.9028 | 0.0042 | Not selected |
| Logistic V2 | 0.8941 | 0.7460 | 0.2003 | 0.9967 | 0.8606 | 0.0055 | Baseline |
| Burden baseline | 0.6049 | 0.5408 | 0.2472 | 0.9755 | 0.6513 | 0.0078 | Diagnostic only |

### 15.3 Corrected V1-like anchor versus V2

| Metric | Corrected V1-like | Selected V2 | Delta |
|---|---:|---:|---:|
| NR@5% | 0.9519 | 0.9705 | +0.0186 |
| AP | 0.7899 | 0.8232 | +0.0333 |
| Brier | 0.1858 | 0.1724 | -0.0134, improvement |
| Protection | 0.9973 | 0.9988 | +0.0015 |
| Technical Score | 0.9034 | 0.9225 | +0.0191 |

Conditional exact-signature-cluster bootstrap for seed 2127 versus the corrected anchor:

- observed delta: 0.0191;
- bootstrap mean delta: 0.0193;
- 95% interval: [0.0168, 0.0221];
- estimated probability of gain: 1.000;
- classification: material.

This uncertainty is conditional on already-fitted OOF predictions. It does not include full model-refit or search uncertainty.

### 15.4 Critical comparison nuance

Historical V1 artifact metrics and corrected-anchor metrics answer different questions:

| Comparison | Technical Score | Valid interpretation |
|---|---:|---|
| Historical V1 CatBoost raw artifact | 0.922361 | What the original V1 OOF code reported |
| Historical V1 selected cross-fitted sigmoid | 0.922234 | What V1 selected offline |
| Corrected V1-like V2 anchor | 0.9034 | V1-like feature/model package under V2's corrected common framework |
| Selected V2 | 0.92246 | V2 under the corrected common framework |

The approximately +0.019 V2 claim is valid only for the corrected common-framework comparison. It is not valid to subtract 0.9034 from the V1 public leaderboard or to call the near-zero difference between historical V1 and V2 OOF a clean head-to-head result.

The large historical-to-corrected-anchor difference is produced by a bundle of changes—label-neutral selection, seeded row shuffling, semantic CSV typing, fixed tree rounds, and common V2 folds/implementation. The executed evidence does not isolate the contribution of each correction. A future audit should not attribute the entire drop to ties; serious-model tie sensitivity was tiny.

### 15.5 Calibration

For selected CatBoost seed 2127:

| Method | NR@5% | AP | Brier | Technical | Deployment unique values | Boundary tie | Eligible? |
|---|---:|---:|---:|---:|---:|---:|---|
| Raw | 0.9705 | 0.8232 | 0.1724 | 0.9225 | 149,014 | 1 | Yes |
| Sigmoid | 0.9705 | 0.8232 | 0.1724 | 0.9225 | 149,014 | 1 | Yes |
| Beta | 0.9705 | 0.8232 | 0.1724 | 0.9225 | 149,014 | 1 | Yes |
| Isotonic | 0.9705 nominal | 0.8232 | 0.1725 | 0.9224 | 141 | 896 | No |

Calibration intercept was 0.0138 and slope 1.0233, already close to ideal. Raw was selected because sigmoid and beta supplied no material Brier improvement, while isotonic introduced unstable plateaus.

### 15.6 Model diversity and blending

OOF diversity:

| Pair | Pearson | Spearman | Top-5 overlap |
|---|---:|---:|---:|
| Cat seed mean vs XGBoost | 0.9171 | 0.9183 | 0.6982 |
| Cat seed mean vs LightGBM | 0.9368 | 0.9368 | 0.7490 |
| XGBoost vs LightGBM | 0.9754 | 0.9757 | 0.8340 |

The best scalar blend was 90% Cat seed mean plus 10% LightGBM in logit space:

- NR@5% 0.9712;
- AP 0.8239;
- Brier 0.1719;
- Protection 0.9989;
- Technical Score 0.9230;
- fold SD 0.0017.

It was not selected because:

- gain over the best single was small;
- confirmation gain was negative, approximately -0.0007;
- it won only one confirmation fold;
- complexity was higher;
- the predeclared gate required repeatable evidence, not the largest aggregate score.

The two-seed CatBoost mean improved the best seed by only about 0.00018 and remained inconclusive.

### 15.7 Test prediction diversity

Selected seed 2127 versus nearby validated alternatives:

| Alternative | OOF Technical | Spearman with primary | Test top-5 overlap |
|---|---:|---:|---:|
| CatBoost seed 2026 | 0.9211 | 0.9983 | 0.9570 |
| CatBoost seed mean | 0.9226 | 0.9996 | 0.9785 |
| Cat/XGB probability 0.9 | 0.9222 | 0.9992 | 0.9510 |
| Cat/XGB logit 0.9 | 0.9226 | 0.9992 | 0.9695 |
| Cat/XGB rank 0.9 | 0.9221 | 0.9989 | 0.9486 |
| Cat/LightGBM probability 0.9 | 0.9226 | 0.9992 | 0.9555 |
| Cat/LightGBM logit 0.9 | 0.9230 | 0.9992 | 0.9690 |
| Cat/LightGBM rank 0.9 | 0.9226 | 0.9991 | 0.9565 |

None met both the OOF-proximity and diversity safeguards for a worthwhile second submission. V2 therefore did not create submission spam.

### 15.8 Fold-model inference stability

| Candidate | Mean pairwise fold top-5 overlap | Minimum | Mean-vs-median top-5 overlap | Mean row probability SD |
|---|---:|---:|---:|---:|
| Corrected V1-like anchor | 0.8175 | 0.8037 | 0.9560 | 0.0415 |
| CatBoost seed 2026 | 0.8534 | 0.8332 | 0.9595 | 0.0425 |
| CatBoost seed 2127 | 0.8473 | 0.8382 | 0.9595 | 0.0425 |
| XGBoost | 0.8488 | 0.8342 | 0.9705 | 0.0297 |
| LightGBM | 0.8611 | 0.8541 | 0.9680 | 0.0342 |

This is why high global correlation is insufficient for operational claims about the audit portfolio.

---

## 16. V2 audit allocation results

### 16.1 Selected V2

| Capacity | K | Fraud captured | Normalized recall / precision | Legitimate audits | Audit FPR | Lift over random |
|---|---:|---:|---:|---:|---:|---:|
| 3% | 4,805 | 4,718 | 0.9819 | 87 | 0.0011 | 1.9609× |
| 5% | 8,008 | 7,772 | 0.9705 | 236 | 0.0030 | 1.9383× |
| 7% | 11,212 | 10,761 | 0.9598 | 451 | 0.0056 | 1.9168× |

### 16.2 Corrected V1-like anchor

| Capacity | Fraud captured | Normalized recall / precision | Legitimate audits |
|---|---:|---:|---:|
| 3% | 4,674 | 0.9727 | 131 |
| 5% | 7,623 | 0.9519 | 385 |
| 7% | 10,464 | 0.9333 | 748 |

### 16.3 Random expectation

At 5%, random audit expectation is:

- fraud captured: 4,009.80;
- precision/normalized recall: 0.5007;
- legitimate audits: 3,998.20;
- lift: 1.0.

The audit table is a decision-capacity analysis, not a claim that 5% is an economically optimal universal policy.

---

## 17. Error and boundary analysis

### 17.1 Corrected V1-like error cohorts

- 385 legitimate claims were inside the corrected V1-like top 5%.
- 72,580 observed fraud labels were outside the top 5%; this large count is expected because the training sample is approximately 50% fraud but only 5% can be selected.
- 2,840 fraud claims lay in the ranks from `K` through `K + 2%N`, forming a near-boundary cohort.

CatBoost V2 seed 2127 and the corrected anchor shared 5,964 top-5 rows. Each selected 2,044 rows not selected by the other. Within those exclusive sets:

- V2 uniquely recovered 1,957 fraud labels;
- the corrected anchor uniquely recovered 1,808.

### 17.2 Numeric error patterns

Largest standardized contrasts:

- V1 false positives versus other legitimate claims: `burden_per_day` 0.3981, `procedure_per_day` 0.3597, and `log1p_los` 0.3089.
- V1 missed fraud versus captured fraud: `procedure_per_day` 0.6049, `burden_per_day` 0.5730, `procedure_nonzero_groups` 0.4924, and `procedure_max_share` 0.4358.
- Near-boundary fraud versus captured fraud: `procedure_per_day` 0.4689, `procedure_nonzero_groups` 0.4647, `procedure_max_share` 0.4320, and `burden_per_day` 0.4286.

These patterns motivated density and concentration features. They describe prediction cohorts, not fraudulent intent.

### 17.3 Categorical error patterns

Large cohort enrichments involved:

- `dati2_prefix2`;
- `typeppk`;
- `cmg` and `cmg__severity`;
- `service__facility`;
- `diagnosis__cmg`;
- `diagnosis__service`;
- `cmg__facility`;
- `kdkc_prefix2`.

This motivated limited structural interactions instead of hundreds of arbitrary crosses.

---

## 18. Feature experiments and ablations

Two-fold same-split screens:

| Variant removed | Technical | Delta from full V2 screen | Status |
|---|---:|---:|---|
| None, full screen | 0.9205 | 0 | Reference |
| Code hierarchy | 0.9201 | -0.0004 | Inconclusive |
| Burden density | 0.9198 | -0.0007 | Inconclusive |
| Code concentration | 0.9193 | -0.0011 | Screen-positive contribution |
| Structured interactions | 0.9192 | -0.0013 | Screen-positive contribution |
| Fold-safe rarity | 0.9196 | -0.0009 | Screen-positive contribution |
| Protected attributes | 0.9197 | -0.0008 | Sensitivity only |

What to retain:

- concentration, limited structural interactions, and fold-safe rarity had enough screening evidence to remain;
- code hierarchy remains for semantic correctness, but its isolated predictive gain is unproven;
- burden-density features are plausible and useful for error interpretation, but their isolated gain is not robustly established;
- protected-feature removal was not promoted;
- there was no extreme suspicious feature family requiring a removal experiment.

The family screens use only two folds. They support finalist pruning, not publication-level claims that every individual feature causes the full V2 gain.

---

## 19. Fairness and policyholder protection

### 19.1 V2 official gaps

| Capacity | Gender gap | Age gap | Protection |
|---|---:|---:|---:|
| 3% | approximately 0.0000 | 0.0010 | 0.9990 |
| 5% | 0.000106 | 0.001153 | 0.998847 |
| 7% | approximately 0.0000 | 0.0021 | 0.9979 |

All two gender groups and all four age groups met the minimum 100-legitimate-observation rule.

At 5%:

- gender `L`: 36,634 legitimate claims, 106 legitimate audits, FPR 0.0029;
- gender `P`: 43,337 legitimate claims, 130 legitimate audits, FPR 0.0030;
- age 0–17: 20,360 legitimate claims, 49 legitimate audits, FPR 0.0024;
- age 18–39: 20,786 legitimate claims, 74 legitimate audits, FPR 0.0036;
- age 40–59: 23,823 legitimate claims, 71 legitimate audits, FPR 0.0030;
- age 60+: 15,002 legitimate claims, 42 legitimate audits, FPR 0.0028.

Conditional exact-signature-cluster 95% intervals at 5%:

- gender gap: [0.0000, 0.0009];
- age gap: [0.0004, 0.0024];
- Protection Score: [0.9976, 0.9996].

### 19.2 Interpretation

- Observed disparities are small relative to the full score range.
- V2 Protection is slightly lower than V1's historical selected estimate because the V2 age gap is somewhat larger, but both are near 1.
- The fairness component is not the main performance limitation.
- Removing protected fields did not improve the two-fold screen.
- No demographic score manipulation was used.
- A gap is described as differential audit exposure, not proof of discrimination.

### 19.3 Operational fairness monitoring

Monitor not only score-based FPR gaps but also:

- group denominators;
- confirmed outcomes;
- investigation delays;
- evidence-request burden;
- overrides;
- escalations;
- appeals and successful appeals;
- data quality and missingness by group.

---

## 20. Explainability findings

### 20.1 Global importance

Top CatBoost native features:

| Feature | Mean native importance |
|---|---:|
| `geo__facility` | 8.8971 |
| `diagnosis__service` | 6.0357 |
| `diagnosis__cmg` | 5.3938 |
| `kdkc` | 4.7468 |
| `cmg__severity` | 4.7007 |
| `frequency__geo__facility` | 4.0760 |
| `kdkc_prefix2` | 3.9877 |
| `umur` | 3.8623 |
| `diagprimer` | 3.7285 |
| `cmg` | 3.6437 |

Cross-fold importance stability was high:

- mean Spearman: 0.9816;
- minimum Spearman: 0.9754;
- mean top-10 pairwise Jaccard: 1.0000.

Held-out permutation AP loss was dominated by:

- `geo__facility`: 0.1029;
- `severitylevel`: 0.0235;
- `diagnosis__cmg`: 0.0190;
- `diagnosis__service`: 0.0164;
- `cmg__severity`: 0.0085;
- `kdkc`: 0.0070.

### 20.2 Interpretation and warning

Location/facility interactions are the dominant predictive family. This may reflect legitimate institutional/case-mix patterns, data-generation artifacts, investigation history, or proxies. The data alone cannot distinguish those mechanisms.

SHAP, native importance, and permutation importance are predictive explanations. They are not causal evidence and do not prove intent.

### 20.3 Local explanation example

For one high-scored held-out claim, the largest positive SHAP contributions included:

- `diagnosis__cmg = h60_h95__J`: +1.1232;
- `geo__facility = 0075__C`: +1.0448;
- `kdkc = 0502`: +0.4205;
- `cmg__severity = J__0`: +0.4084;
- `diagnosis__service = h60_h95__2`: +0.3585.

The notebook also shows representative audited fraud, audited legitimate, and highest-scored non-audited fraud examples. Claim IDs remain internal dataset identifiers and were not externally searched.

### 20.4 Recommended investigator display

Show:

- calibrated fraud probability;
- audit rank and current capacity boundary;
- raw supporting claim facts;
- largest signed model drivers;
- missing, rare, or unseen-category warnings;
- relevant comparable patterns;
- uncertainty and data-context notes;
- an override/escalation/appeal route.

Never show the score as proof that fraud occurred.

---

## 21. Robustness and transport findings

### 21.1 Location stress

| Perspective | NR@5% | AP | Brier | Protection | Technical |
|---|---:|---:|---:|---:|---:|
| Primary grouped OOF, Cat seed 2026 raw | 0.9680 | 0.8231 | 0.1722 | 0.9989 | 0.9211 |
| Unseen-`kdkc` holdout | 0.7279 | 0.6097 | 0.2658 | 0.9840 | 0.7308 |

Technical Score stress delta: **-0.19025**.

This does not mean the Kaggle test is shifted to unseen locations; the train/test shift audit says otherwise. It means deployment to genuinely unseen locations is much weaker and must be validated separately.

### 21.2 Location removal

Historical V1 location-feature removal reduced Technical Score by about 0.095. Blindly deleting those fields is therefore not a high-value competition strategy.

Best current policy:

- retain location fields for the in-distribution competition submission;
- document them as proxy/governance risks;
- monitor by location;
- require shadow validation for new regions;
- consider hierarchical, regularized, or updated location representation for operational deployment.

### 21.3 Shift × importance

The largest importance-times-shift raw feature was `umur`, but the absolute measured shift remained small. `kdkc`, `dati2`, and LOS were important, while their train/test shift measures were also small.

### 21.4 What robustness evidence does and does not say

Supported:

- the corrected V2 package beats the corrected anchor;
- location transport is a real weakness;
- raw calibration is sufficient;
- fixed blends contain complementary signal.

Not established:

- that seed averaging is superior;
- that fold averaging beats full refit on hidden labels;
- that the scalar-best blend will improve private leaderboard performance;
- that each feature-family screen would survive full five-fold ablation.

---

## 22. Experiments that were ineffective, rejected, or not worth the cost

### 22.1 Ineffective or misleading

- Constant/prevalence top-k baseline under source-row stable ties: invalid because it exploited label storage order.
- Burden baseline's perfect 3% result: tie artifact.
- V1 cross-fold sigmoid as a combined ranking object: changed global OOF ordering and lost two top-5 fraud labels.
- Isotonic calibration: created large plateaus and unstable exact-K selection.
- V1 arbitrary ordinal XGBoost categories: methodologically weak representation.
- Removing protected variables: did not improve score or Protection.
- Removing location variables: caused severe performance loss.
- Selecting the scalar-highest V2 blend: not supported on confirmation folds.
- Two-seed mean: improvement too small to treat as real.

### 22.2 Explicitly rejected directions

- Dedicated learning-to-rank model: insufficient structural justification and worsens calibration/deployment complexity.
- Deep tabular neural network: no evidence that added cost and opacity would beat boosting.
- Massive hyperparameter optimization: high model-selection-overfit risk for marginal expected value.
- Demographic post-processing to game Protection: ethically and methodologically unjustified.
- Public leaderboard row probing: invalid competition behavior and poor private-LB strategy.
- Arbitrary business-cost optimization: no organizer-provided operational costs.

### 22.3 Inconclusive, not failed

- semantic code hierarchy's isolated predictive gain;
- burden-density feature family;
- two-seed averaging;
- fold-average versus full-refit hidden-test superiority;
- a cross-family blend's private-LB gain;
- exact source of the V1 historical-to-corrected-anchor score difference.

Inconclusive means the current evidence is too weak, not that the idea is false.

---

## 23. What should be applied now

### 23.1 Current strongest defensible submission

Use the V2 primary generated by:

- CatBoost V2 seed 2127;
- all five grouped-CV fold models;
- fold-average raw test probability;
- raw calibration choice;
- exact sample-submission ordering;
- no manual score edits.

Expected Kaggle output path:

`/kaggle/working/submission_primary.csv`

The executed run created this file, and the copied local submission is ready at:

`results/v2/submission_primary.csv`

Local validation performed on 30 August 2026 confirmed:

- header exactly `claim_id,fraud_probability`;
- 40,043 rows, matching test and sample submission;
- IDs complete, unique, and in exact sample-submission order;
- all probabilities finite and inside `[0,1]`;
- minimum probability `0.00612441572978`;
- maximum probability `0.996131995757`;
- SHA-256 `8cc89c67c689c6c965a7f40029b77bc226a9bdf38764c71de853e19f1282f72e`.

### 23.2 Secondary submission

Do not manufacture a second file. The executed rule withheld it because every OOF-near candidate was too similar to the primary or lacked robust enough evidence.

If competition submission slots allow two and a later experiment creates a materially different, evidence-qualified candidate, then a second file is justified. Current evidence does not.

### 23.3 Operational policy

- Use the model only to prioritize human review.
- Never auto-reject or auto-deny claims.
- Allow documented human override.
- Preserve investigator evidence and audit trails.
- Offer escalation and appeal.
- Maintain a small randomized audit stream so labels are not created only from high-scored claims.
- Monitor ranking, calibration, fairness, queue burden, input drift, prediction drift, location performance, label maturity, override outcomes, and appeals.
- Recalibrate when mature representative labels show reliability deterioration.
- Retrain when verified concept change, sustained performance decline, new fraud typologies, or data-contract changes justify it.
- Use policy-approved thresholds; notebook metrics alone should not invent deployment tolerances.

### 23.4 If a V3 is considered

The next highest-value research is not broad model shopping. Priorities should be:

1. isolate why the historical V1 artifact and corrected V1-like anchor differ so much;
2. obtain or reconstruct a clean apples-to-apples historical V1 comparison under the corrected selector, one fold definition, one data typing policy, and one inference recipe;
3. test location generalization with regularized/hierarchical frequency representations rather than deleting location signal;
4. promote only feature-family changes that survive all five folds;
5. compare the fixed LightGBM logit blend against seed 2127 with a fully paired confirmation design, not public-LB tuning;
6. preserve rank metrics on one common score and estimate calibration separately;
7. stop if gains remain below uncertainty or add disproportionate complexity.

---

## 24. Known, interpreted, plausible, and unknown

### 24.1 Known from executed evidence

- Train/test schemas and all dataset counts stated above.
- Labels are stored in one positive block then one negative block.
- V1 tie handling invalidated constant and coarse baseline top-k results.
- Serious CatBoost boundary ties had negligible effect.
- Cross-fold calibration maps can reorder global OOF rows.
- V1 full refit and fold-average test portfolios differ materially near top 5%.
- V2 ran successfully on two visible T4 GPUs.
- V2 selected CatBoost seed 2127 with raw calibration.
- V2 scalar-best blend failed confirmation safeguards.
- V2 did not produce a secondary.
- Location transport stress is much worse than primary OOF.
- Fairness gaps are small under the official metric.

### 24.2 Reasonable interpretations

- CatBoost is the best current anchor because it consistently dominates other single model families on common OOF evidence.
- Location/facility interactions encode substantial in-distribution signal but create deployment risk.
- V2's fold-average recipe is easier to defend offline than V1's unvalidated full refit.
- Raw selected CatBoost is already adequately calibrated for the competition sample.
- The best blend contains complementary information but its gain is too fragile for primary selection.

### 24.3 Plausible but unverified

- V1 public score may be higher than offline estimates because the public subset is easier, full-data fitting helps, prevalence differs, or public sample variance is favorable.
- Some location signal may reflect institutional process, prior investigation patterns, or label artifacts.
- A carefully regularized location hierarchy may improve unseen-location robustness.
- The LightGBM logit blend may help on private labels despite failing the current confirmation gate.

### 24.4 Unknown

- V2 public and private leaderboard scores.
- Hidden test prevalence and fairness labels.
- Whether V1 full refit or fold averaging is superior on hidden test labels.
- Organizer tie-breaking behavior.
- Provider, member, or temporal identities not supplied in the data.
- Label adjudication quality and operational base rate.
- Causal meaning of any feature association or SHAP attribution.

---

## 25. Research questions answered by V2

1. **Data quality:** 160,174 train and 40,043 test claims; no missing raw cells; repeated and conflicting signatures; unsafe stored-label order; codes require string preservation.
2. **Validation:** signature-grouped five-fold CV is the primary feasible selection view; unseen-location holdout is a distinct transport stress view.
3. **V1 diagnosis:** tie handling, fold-specific calibration ranking, blend deployment mismatch, XGBoost encoding, tree cap, and refit/foldbag divergence were verified.
4. **Features:** the overall V2 package improves the corrected anchor; concentration, rarity, and structured interactions are screen-positive, but family-specific evidence remains provisional.
5. **Model:** CatBoost seed 2127 is the selected robust single candidate.
6. **Calibration:** raw is sufficient; isotonic is unsafe for exact top-k ranking.
7. **Allocation:** at 5%, V2 captures 7,772 observed fraud labels and audits 236 legitimate claims in OOF.
8. **Errors:** procedure density and location/service/diagnosis interactions distinguish misses and recoveries.
9. **Protection:** 5% gender gap 0.000106, age gap 0.001153, Protection 0.998847.
10. **Operations:** use human review, appeal, randomized learning, monitoring, drift checks, versioning, and rollback.

Executed V2 completeness mapping:

| Competition area | Covered evidence |
|---|---|
| Task A | Missing/inconsistent values, duplicates, unusual values, cardinality, proxies, leakage, subgroup patterns, association-versus-causality limits |
| Task B | Feature representation, preprocessing, model choice, grouped validation, imbalance policy, leakage prevention, overfitting controls, calibration, prediction drivers |
| Task C | 3/5/7% budgets, fraud capture, legitimate audits, random and burden baselines, lift, capacity sensitivity |
| Task D | Gender, all four official age groups, 100-legitimate eligibility, FPR gaps, uncertainty, careful disparity interpretation |
| Task E | Investigator presentation, alert evidence, human override, escalation, appeal, performance/disparity monitoring, drift, recalibration, retraining, decision-support boundary |
| Technical | Exact NR@5%, AP, Brier, Calibration Score, Protection Score, Technical Score, tie and edge checks |
| Research/report | Literature-backed Chapters I–II, verified IEEE references, numbered tables/figures, dynamic results, explicit RQ conclusions |
| Reproducibility | Kaggle paths, T4 x2 detection, deterministic seeds, top-to-bottom execution, validated final submission |

---

## 26. Current filesystem state

### 26.1 Project is not a Git repository

Running `git status` from `/home/rian/Projects/prs` returns:

```text
fatal: not a git repository (or any parent up to mount point /)
```

There is therefore no trustworthy `git diff` or commit history in this directory. Filesystem hashes and notebook execution metadata are the current provenance mechanism.

### 26.2 Main inputs

```text
data/train.csv
data/test.csv
data/sample_submission.csv
docs/ANSWER SHEET PRELIMINARY ROUND.docx
ref/Notebook_InfoMagangDong (Penyisihan).ipynb
ref/Answer Sheets_sayonarasrama [-1].pdf
ref/Answer Sheet_InfoMagangDong.docx
```

### 26.3 Main notebooks

```text
notebooks/DAC2026_Ultimate_Baseline_T4x2.ipynb
notebooks/DAC2026_V2_Ultimate.ipynb
```

Both current files are executed Kaggle outputs with embedded results.

### 26.4 V1 persisted artifacts

```text
results/v1/ablation_results.csv
results/v1/audit_allocation.csv
results/v1/bootstrap_intervals.csv
results/v1/categorical_level_rates.csv
results/v1/experiment_registry.csv
results/v1/fairness_detail.csv
results/v1/fold_metrics.csv
results/v1/model_decision.csv
results/v1/monitoring_plan.csv
results/v1/primary_fold_assignments.csv
results/v1/quality_profile.csv
results/v1/submission.csv
results/v1/train_test_shift.csv
results/v1/predictions_burden_baseline.npz
results/v1/predictions_catboost.npz
results/v1/predictions_catboost_seed_2127.npz
results/v1/predictions_logistic_regression.npz
results/v1/predictions_xgboost.npz
results/v1/predictions_ablation_engineered_features_removed.npz
results/v1/predictions_ablation_location_proxies_removed.npz
results/v1/predictions_ablation_protected_attributes_removed.npz
results/v1/figure_3_1.png
results/v1/figure_4_1.png through figure_4_13.png
results/v1/__results___files/*.png
```

Selected immutable V1 hashes:

```text
experiment_registry.csv  d5158941b81b143d7336b344e79c82736fa3ca3ab0403d6ceb51a568ba80fd21
fold_metrics.csv          48e92076a8675ac5bd84f862e483a3668c4e2d7b9acc2bd4d2d20694bffb9728
model_decision.csv        130b15ff51ad0da088bbbad39614ebfe048f0e259e513315b725f0a62d68c886
predictions_catboost.npz  ffa23f887c1b3312ee066e15c03d868a8af85c53b25b06fc8a7f5c5c74509ffa
predictions_xgboost.npz   109980e0f56a0e75524147613edce29a160814d32a11fe11b1d8916df68d5408
submission.csv            634a0a47816382174b1d1b3d235406fb48ef53cf65560339bf44a57f3b6e95ea
```

### 26.5 V2 generated artifacts recovered locally

The V2 notebook's runtime manifest confirms these existed under `/kaggle/working/v2_artifacts/` during execution. The Kaggle output bundle has now been copied to `results/v2/v2_artifacts/`:

```text
selected_predictions.npz
preflight_checks.csv
experiment_registry_models_partial.csv
blend_grid_partial.csv
calibration_comparison_partial.csv
experiment_registry_partial.csv
eligibility_partial.csv
experiment_registry_selected.csv
experiment_registry.csv
fold_metrics.csv
model_decision.csv
hypothesis_registry.csv
v1_forensics.csv
v1_artifact_provenance.csv
calibration_comparison.csv
paired_uncertainty.csv
audit_allocation.csv
fairness_detail.csv
fairness_gaps.csv
fairness_gap_uncertainty.csv
error_contrasts.csv
categorical_error_cohorts.csv
prediction_comparison.csv
inference_stability.csv
ablation_screen.csv
feature_hypothesis_results.csv
feature_ablation_fold_detail.csv
location_stress.csv
train_test_shift.csv
shift_importance.csv
quality_profile.csv
categorical_level_rates.csv
monitoring_plan.csv
conclusions.csv
predictions_burden_baseline.npz
predictions_corrected_v1-like_anchor.npz
predictions_logistic_v2.npz
predictions_catboost_v2_seed_2026.npz
predictions_catboost_v2_seed_2127.npz
predictions_xgboost_v2_one-hot.npz
predictions_lightgbm_v2_one-hot.npz
primary_fold_assignments.csv
figure_3_1.png
figure_4_1.png through figure_4_12.png
```

The primary submission existed at:

`/kaggle/working/submission_primary.csv`

The runtime manifest recorded 40,043 rows and a file size of 1,531,177 bytes. The identical-size copied file is now at `results/v2/submission_primary.csv`. A secondary file did not exist because the evidence rule withheld it.

Selected local V2 hashes:

```text
submission_primary.csv     8cc89c67c689c6c965a7f40029b77bc226a9bdf38764c71de853e19f1282f72e
experiment_registry.csv    cafac25214d28f6907365b5fd8cde2ec0405e8fb7a517513d50071265dc2a0fc
model_decision.csv          5fa1b1849cbc32488af95c2b0e1762962a96d972f06e435c69af776035fe0ab0
selected_predictions.npz    6daf6d29f043e8e1f2310a9d68859c0b509b44f382468b3618237bc808c8972e
```

`results/v2/submission_primary.csv` is therefore the current upload-ready file. Preserve the entire `results/v2/` directory as executed evidence.

### 26.6 Tests and current artifact-state caveat

Test files:

```text
tests/test_dac2026_notebook.py
tests/test_dac2026_v2_notebook.py
```

Latest lightweight test observations:

- V1 official metric behavior test passes.
- V2 metric, label-neutral tie, registry, boolean eligibility, and prediction-shape contract tests pass.
- V2: 7 tests pass, one static-contract test fails, and one V1-preservation test errors.
- V1 static sequential-clean-state test fails.

These failures do **not** indicate that the executed notebooks failed on Kaggle. They arise because:

1. the tests expect clean authoring notebooks with `execution_count=None` and empty outputs, while both current notebook files are executed artifacts;
2. the clean authoring builders `notebooks/build_dac2026.py` and `notebooks/build_dac2026_v2.py` are currently absent; the V2 preservation test explicitly expects the former.

Every current notebook code cell compiles successfully:

- V1: 34/34 code cells compile;
- V2: 38/38 code cells compile.

The missing builders and clean-source-versus-executed-artifact distinction should be repaired only if further notebook development resumes. Preserve the executed evidence first; do not clear or overwrite either result-bearing notebook.

---

## 27. Complete next-chat continuation instructions

The next chat should do the following in order:

1. Read this file in full.
2. Inspect `notebooks/DAC2026_V2_Ultimate.ipynb` as an executed notebook; do not assume V2 is still only a plan.
3. Preserve both V1 and V2 executed notebooks before editing or clearing outputs.
4. Inspect `results/v1/` and verify the listed hashes before reusing predictions.
5. Inspect and preserve the already-recovered V2 bundle under `results/v2/`; verify the selected hashes before reusing predictions. Do not overwrite V1 or V2 evidence.
6. Record whether the V2 primary has already been submitted. No V2 leaderboard score is presently known in this handoff.
7. If the goal is simply the next Kaggle action, upload `results/v2/submission_primary.csv`. Do not create a secondary.
8. If the goal is further research, first isolate the historical V1-versus-corrected-anchor discrepancy. Do not run a broad search before this is understood.
9. Keep top-k ranking on a common raw score; use cross-fitting only to estimate calibration quality.
10. Retain claim-ID tie breaking only as a label-neutral deterministic policy and continue to report expected/optimistic/pessimistic tie sensitivity.
11. Keep grouped primary CV and separate location stress. Do not replace one with the other as if they answer the same question.
12. Treat location fields as useful but governance-sensitive. Do not blindly remove them or claim causality.
13. Do not use public-LB score movement to choose rows, weights, or calibration.
14. Run heavy experiments only on Kaggle T4 x2.
15. Preserve the official Chapters I–V, References, and Attachment report structure.
16. Any new empirical conclusion must be generated from executed values, not typed as an invented number.
17. If authoring tests are restored, keep separate files for a clean upload notebook and an executed evidence notebook, or update the test contract explicitly.

### Suggested opening prompt for the next chat

```text
Read docs/DAC2026_COMPLETE_CHAT_HANDOFF_2026-08-30.md in full and treat the filesystem as source of truth. V1 and V2 have both already been executed on Kaggle. Preserve their notebooks and `results/v1/` and `results/v2/` evidence. Verify the listed hashes and continue from the earliest genuinely incomplete objective. The validated upload-ready primary is `results/v2/submission_primary.csv`; first record whether it has already been submitted and whether any leaderboard score exists. Do not repeat reference inspection, literature research, V1 training, or V2 experiments unless the handoff identifies an unresolved verification need. No public-leaderboard probing and no hidden-label reconstruction.
```

---

## 28. Operational recommendation retained for the answer sheet

### Investigator interface

Display probability, rank, model drivers, source claim facts, data-quality warnings, comparable context, and reason codes. Avoid accusatory wording.

### Human override

Investigators may override priority when documented evidence indicates severe data error, urgent patient harm risk, credible contradictory documentation, or a novel fraud pattern outside model representation. Overrides must be logged with reason and outcome.

### Escalation and appeal

Disputed cases should move to an independent or senior review path. Policyholders/providers should have a clear correction and appeal mechanism. The model score cannot be the sole adverse-action basis.

### Feedback-loop protection

If only high-scored claims are investigated, future labels become selected by the model. Preserve randomized review and label-maturity metadata so monitoring does not confuse model-driven selection with population fraud risk.

### Monitoring

Track:

- NR@3/5/7%;
- AP;
- Brier, calibration intercept/slope, and reliability curves;
- gender and age FPR gaps with denominators and intervals;
- input contract, missingness, unseen levels, and code formatting;
- KS, standardized mean difference, JS divergence, and adversarial AUC;
- prediction distribution and top-boundary churn;
- performance by location and service pattern;
- label maturity and adjudicator agreement;
- override, escalation, appeal, and delay outcomes;
- feature-importance/explanation drift;
- system version, access logs, failures, and rollback readiness.

### Recalibration and retraining

Recalibrate only on representative, mature, recent labels when probability reliability deteriorates. Retrain when evidence shows sustained ranking loss, concept drift, new fraud typologies, material data-contract change, or validated transport failure. Thresholds should be policy-approved and based on monitoring uncertainty, not arbitrary numbers inserted into the report.

---

## 29. Limitations that must remain explicit

- The approximately balanced training prevalence may not represent operations.
- Label quality, maturity, and adjudication rules are unknown.
- No causal inference is supported.
- Exact signatures are grouped, but latent provider/member/time dependence may remain.
- Reduced clinical templates overlap heavily across train/test.
- The test labels and hidden subgroup labels are unavailable.
- Public leaderboard is a noisy partial sample.
- Fixed-prediction bootstrap does not include refitting or model-search uncertainty.
- Feature-family ablations use only two folds.
- Selection and confirmation folds are not a fully nested external experiment.
- Fold-average test inference differs from single-fold OOF inference.
- Location transport is weak.
- Official FPR gaps cover only a narrow part of fairness.
- SHAP and importance can be unstable with correlated variables.
- Fraud behavior can adapt after deployment.
- V2 outputs are local, but no V2 public/private leaderboard score has been supplied; offline evidence cannot substitute for that result.
- Current authoring tests are mismatched to executed notebook artifacts.

---

## 30. Verified bibliography used by V2

The following is the executed notebook's IEEE-style bibliography. It is included here so the next chat does not redo completed literature research.

[1] DAC 2026 Competition Committee, “Preliminary-Round Competition Specification and Evaluation Rules,” official competition material supplied to participants, 2026.

[2] World Health Organization, *Use of Machine Learning for Fraud Detection within the Claims Management Process in the Philippines: Insights from the National Health Insurance Scheme PhilHealth*. Geneva, Switzerland: WHO, 2024, ISBN 978-92-4-010133-3. https://www.who.int/publications/i/item/9789240101333

[3] World Health Organization, *The Implications of Artificial Intelligence and Machine Learning in Health Financing for Achieving Universal Health Coverage: Findings from a Rapid Literature Review*. Geneva, Switzerland: WHO, 2023, ISBN 978-92-4-006401-0. https://www.who.int/publications/i/item/9789240064010

[4] A. du Preez, S. Bhattacharya, P. A. Beling, and E. Bowen, “Fraud detection in healthcare claims using machine learning: A systematic review,” *Artificial Intelligence in Medicine*, vol. 160, Art. no. 103061, 2025, doi: 10.1016/j.artmed.2024.103061.

[5] H. Joudaki, A. Rashidian, B. Minaei-Bidgoli, M. Mahmoodi, B. Geraili, M. Nasiri, M. Arab, and E. Qorbani, “Using data mining to detect health care fraud and abuse: A review of literature,” *Global Journal of Health Science*, vol. 7, no. 1, pp. 194–202, 2015, doi: 10.5539/gjhs.v7n1p194.

[6] J. Davis and M. Goadrich, “The relationship between Precision-Recall and ROC curves,” in *Proc. 23rd International Conference on Machine Learning*, 2006, pp. 233–240, doi: 10.1145/1143844.1143874.

[7] T. Saito and M. Rehmsmeier, “The Precision-Recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,” *PLOS ONE*, vol. 10, no. 3, Art. no. e0118432, 2015, doi: 10.1371/journal.pone.0118432.

[8] G. W. Brier, “Verification of forecasts expressed in terms of probability,” *Monthly Weather Review*, vol. 78, no. 1, pp. 1–3, 1950, doi: `10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2`.

[9] B. Van Calster, D. J. McLernon, M. van Smeden, L. Wynants, and E. W. Steyerberg, “Calibration: The Achilles heel of predictive analytics,” *BMC Medicine*, vol. 17, Art. no. 230, 2019, doi: 10.1186/s12916-019-1466-7.

[10] M. Kull, T. M. Silva Filho, and P. Flach, “Beta calibration: A well-founded and easily implemented improvement on logistic calibration for binary classifiers,” in *Proc. 20th International Conference on Artificial Intelligence and Statistics*, PMLR, vol. 54, 2017, pp. 623–631. https://proceedings.mlr.press/v54/kull17a.html

[11] S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, “Leakage in data mining: Formulation, detection, and avoidance,” *ACM Transactions on Knowledge Discovery from Data*, vol. 6, no. 4, Art. no. 15, 2012, doi: 10.1145/2382577.2382579.

[12] D. R. Roberts et al., “Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure,” *Ecography*, vol. 40, no. 8, pp. 913–929, 2017, doi: 10.1111/ecog.02881.

[13] G. C. Cawley and N. L. C. Talbot, “On over-fitting in model selection and subsequent selection bias in performance evaluation,” *Journal of Machine Learning Research*, vol. 11, pp. 2079–2107, 2010. https://www.jmlr.org/papers/v11/cawley10a.html

[14] J. H. Friedman, “Greedy function approximation: A gradient boosting machine,” *The Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001, doi: 10.1214/aos/1013203451.

[15] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, “CatBoost: Unbiased boosting with categorical features,” in *Advances in Neural Information Processing Systems 31*, 2018. https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html

[16] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.

[17] G. Ke et al., “LightGBM: A highly efficient gradient boosting decision tree,” in *Advances in Neural Information Processing Systems 30*, 2017, pp. 3146–3154. https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

[18] P. Kar, H. Narasimhan, and P. Jain, “Surrogate functions for maximizing precision at the top,” in *Proc. 32nd International Conference on Machine Learning*, PMLR, vol. 37, 2015, pp. 189–198. https://proceedings.mlr.press/v37/kar15.html

[19] A. Niculescu-Mizil and R. Caruana, “Predicting good probabilities with supervised learning,” in *Proc. 22nd International Conference on Machine Learning*, 2005, pp. 625–632, doi: 10.1145/1102351.1102430.

[20] M. Hardt, E. Price, and N. Srebro, “Equality of opportunity in supervised learning,” in *Advances in Neural Information Processing Systems 29*, 2016. https://proceedings.neurips.cc/paper/2016/hash/9d2682367c3935defcb1f9e247a97c0d-Abstract.html

[21] A. Rajkomar, M. Hardt, M. D. Howell, G. Corrado, and M. H. Chin, “Ensuring fairness in machine learning to advance health equity,” *Annals of Internal Medicine*, vol. 169, no. 12, pp. 866–872, 2018, doi: 10.7326/M18-1990.

[22] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in *Advances in Neural Information Processing Systems 30*, 2017. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

[23] I. E. Kumar, S. Venkatasubramanian, C. Scheidegger, and S. Friedler, “Problems with Shapley-value-based explanations as feature importance measures,” in *Proc. 37th International Conference on Machine Learning*, PMLR, vol. 119, 2020, pp. 5491–5500. https://proceedings.mlr.press/v119/kumar20e.html

[24] J. Amann, A. Blasimme, E. Vayena, D. Frey, and V. I. Madai, “Explainability for artificial intelligence in healthcare: A multidisciplinary perspective,” *BMC Medical Informatics and Decision Making*, vol. 20, Art. no. 310, 2020, doi: 10.1186/s12911-020-01332-6.

[25] World Health Organization, *Ethics and Governance of Artificial Intelligence for Health*. Geneva, Switzerland: WHO, 2021, ISBN 978-92-4-002920-0. https://www.who.int/publications/i/item/9789240029200

[26] E. Tabassi, *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*, NIST AI 100-1. Gaithersburg, MD, USA: National Institute of Standards and Technology, 2023, doi: 10.6028/NIST.AI.100-1.

[27] J. Gama, I. Žliobaitė, A. Bifet, M. Pechenizkiy, and A. Bouchachia, “A survey on concept drift adaptation,” *ACM Computing Surveys*, vol. 46, no. 4, Art. no. 44, 2014, doi: 10.1145/2523813.

[28] B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: Chapman & Hall, 1993, doi: 10.1007/978-1-4899-4541-9.

[29] E. B. Wilson, “Probable inference, the law of succession, and statistical inference,” *Journal of the American Statistical Association*, vol. 22, no. 158, pp. 209–212, 1927, doi: 10.1080/01621459.1927.10502953.

[30] C. J. Kelly, A. Karthikesalingam, M. Suleyman, G. Corrado, and D. King, “Key challenges for delivering clinical impact with artificial intelligence,” *BMC Medicine*, vol. 17, Art. no. 195, 2019, doi: 10.1186/s12916-019-1426-2.

[31] CatBoost, “Training on GPU,” *CatBoost Documentation*. https://catboost.ai/docs/en/features/training-on-gpu. Accessed 28 August 2026.

[32] XGBoost Developers, “Using XGBoost with device CUDA,” *XGBoost Documentation*. https://xgboost.readthedocs.io/en/stable/gpu/index.html. Accessed 28 August 2026.

---

## 31. Final state summary for a new analyst

### Completed

- Official template and all three references inspected.
- Literature research and citation verification completed.
- Dataset forensics completed.
- V1 notebook authored and executed.
- V1 Kaggle submission generated and scored publicly at about 0.958.
- V1 artifacts recovered locally.
- Independent metric and tie audit completed.
- V1 calibration, blend, inference, fairness, shift, and ablation audit completed.
- V2 notebook authored and executed successfully on Kaggle T4 x2.
- V2 metrics, uncertainty, feature screens, calibration, blends, fairness, explanations, robustness, and submission selection completed.
- V2 primary generated during Kaggle execution.
- V2 primary, prediction NPZ files, registries, diagnostics, and figures recovered locally under `results/v2/`.
- Local V2 submission schema, ID order, uniqueness, finiteness, and probability bounds independently validated.
- No evidence-qualified V2 secondary generated.

### Partially complete or locally incomplete

- Clean authoring notebook copies are not separated from executed evidence copies.
- The clean V1/V2 builder scripts are absent; the preservation test still expects `notebooks/build_dac2026.py`.
- V2 has no known leaderboard result in this conversation.

### Remaining decision

The immediate evidence-based action is to submit `results/v2/submission_primary.csv` once if it has not already been submitted, then record the score as a coarse external signal. Further model development should wait until either:

- V2 submission feedback is available as a coarse external signal, without row-level probing; or
- the historical V1-versus-corrected-anchor discrepancy is isolated in a controlled offline comparison.

The project should not be restarted, the references should not be re-researched, and the completed V1/V2 runs should not be overwritten.

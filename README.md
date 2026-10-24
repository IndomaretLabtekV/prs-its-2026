# DAC PRS ITS 2026 — Team IndomaretLabtekV

🥉 **3rd Place, Final Round**, Data Analysis Competition (DAC) 2026, organized by HIMASTA-ITS (Institut Teknologi Sepuluh Nopember).

The repository covers both rounds of the competition:

| Round | Problem | Data | Metric | Deliverables |
|---|---|---|---|---|
| [Preliminary](#preliminary-round--healthcare-claim-fraud) | Rank healthcare insurance claims for a 5% audit budget | 160,174 train / 40,043 test claims | Normalized Recall@5% | [notebook](preliminary/notebook.ipynb), [answer sheet](preliminary/report.pdf) |
| [Final](#final-round--life-insurance-churn) (3rd place) | Predict policy churn for a life-insurance portfolio | 5,000 policies, 4,704 customers | ROC AUC | [notebook](final/notebook.ipynb), [report](final/report.pdf) |

The work focuses on sound analysis: leakage-free validation, calibrated probabilities, error and fairness analysis, and decisions tied to business constraints.

---

## Preliminary Round — Healthcare Claim Fraud

**Task.** Score every claim by fraud risk. Only the top 5% of claims (`K = ⌊0.05N⌋`) can be audited, so what matters is how many fraud cases end up in that portfolio. Leaderboard: [Kaggle — Preliminary Round DAC 2026](https://www.kaggle.com/competitions/preliminary-round-dac-2026/leaderboard).

**Submitted solution: V2** ([`preliminary/notebook.ipynb`](preliminary/notebook.ipynb)). Like the answer sheet ([`preliminary/report.pdf`](preliminary/report.pdf)), the notebook follows the competition's five-chapter format: Introduction, Theoretical Framework, Analytical Steps, Analysis of Results, and Conclusion.

**Approach**
- **Data forensics.** 0 missing cells, but 17,906 rows share an identical feature signature and 1,737 of those groups have conflicting labels. Source row order and `claim_id` carry label information, so neither is used as a feature.
- **Leakage-free validation.** `StratifiedGroupKFold` grouped by exact feature signature, with zero group overlap between folds. A plain K-fold split would have put 13,092 signature groups in both train and validation.
- **Models.** Logistic regression, LightGBM, XGBoost, and CatBoost (two seeds), compared on the same out-of-fold (OOF) evidence. We selected the model with a rule fixed before seeing the results, not by public leaderboard score.
- **Selected model.** A CatBoost fold ensemble: OOF NR@5% 0.9705, AP 0.8232, fold SD 0.0027. On the OOF scores, the raw output was already as well calibrated as the calibrated versions, so it is used unchanged.
- **Audit allocation.** At 5% (K = 8,008), the model captures 7,772 fraud claims and sends 236 legitimate ones to audit, 1.94× better than random selection. Workload curves for 3%, 5%, and 7% budgets are included.
- **Fairness and shift.** The gap in audit exposure is 0.0001 between genders and 0.0012 between age groups. Train and test barely differ (adversarial AUC 0.507).

**Post-submission research** ([`preliminary/experiments/`](preliminary/experiments)). After submitting, we ran more experiments to see how far the score could go:

| Version | Idea |
|---|---|
| V1 | First baseline: CatBoost with sigmoid calibration |
| V3 | Cross-fitted stacking plus a CatBoost refiner that re-ranks the top-10% boundary (strongest OOF result) |
| V4–V5 | Consensus re-ranking and a repaired supervised boundary stack |
| V7–V8 | Re-rankers trained on hard negatives near the audit cutoff |
| V9–V10 | New feature representations and categorical interactions |
| H1–H16 | Hypothesis tests on the refiner: ranking loss, capacity, class cost, boundary weighting |

None of the later versions beat V3 under our acceptance gate. The analysis points to a ceiling set by label noise: many identical claims carry conflicting labels. See [`findings.md`](preliminary/experiments/research/findings.md). These notebooks and scripts are kept as a record. They expect intermediate artifacts that are not in this repository, so they will not run as-is.

---

## Final Round — Life-Insurance Churn

🥉 **3rd place.**

**Task.** Find the policies most likely to churn so the insurer can target retention. The data comes as a workbook with policy, customer, and claim sheets. We merged the policy and customer sheets on `customer_id` into [`final/train/train.csv`](final/train/train.csv). The claim sheet has only 17 rows and is left out.

**Approach** ([`final/notebook.ipynb`](final/notebook.ipynb), report: [`final/report.pdf`](final/report.pdf))
- **Part 1: EDA.** Data profile, data quality, target analysis, univariate and multivariate structure, and a quick baseline signal check. The modelling decisions are written to `eda-output/eda_decisions.json`.
- **Part 2: Modelling pipeline.** Reads those decisions. Validation is group 5-fold by `customer_id`, plus a 20% holdout. Churned policies are rare (185, or 3.7%), so the models use class weights. A logistic regression baseline is compared with LightGBM, XGBoost, and CatBoost, and the three boosted models are combined in a weighted ensemble.

**Results (holdout)**

| Model | ROC AUC | PR AUC | Log loss |
|---|---|---|---|
| Ensemble (LightGBM 0.55, CatBoost 0.37, XGBoost 0.08) | **0.9911** (95% CI 0.986–0.996) | **0.649** | **0.055** |
| Logistic regression baseline | 0.9873 | 0.557 | 0.099 |

- The ensemble finds 92% of churners in the holdout (24 of 26), at 0.53 precision. For a retention campaign this trade is acceptable: contacting a loyal customer costs less than losing one.
- **Tenure** is by far the strongest driver (69% of model gain), followed by **payment frequency** and whether the customer reported an occupation.
- Main limitation: there are only 185 churn cases, so the confidence intervals stay wide.

---

## Repository Structure

```
.
├── preliminary/
│   ├── notebook.ipynb            # submitted solution (V2)
│   ├── report.pdf                # submitted answer sheet
│   ├── outputs/                  # V2 artifacts: figures, tables, OOF predictions, submission.csv
│   ├── experiments/
│   │   ├── notebooks/            # V1, V3–V10
│   │   └── research/             # hypothesis protocols, scripts, findings, literature notes
│   └── data/                     # Kaggle data goes here (not tracked)
├── final/
│   ├── notebook.ipynb            # EDA + modelling pipeline
│   ├── report.pdf                # final report
│   ├── data/                     # raw workbook, data dictionary, case brief
│   ├── train/train.csv           # merged modelling table
│   ├── eda-output/               # EDA figures, tables, decisions
│   └── pipeline-output/          # metrics, predictions, figures, run report
└── requirements.txt
```

## Reproducing

```bash
pip install -r requirements.txt
```

- **Preliminary.** Download `train.csv`, `test.csv`, and `sample_submission.csv` from the [competition page](https://www.kaggle.com/competitions/preliminary-round-dac-2026) into `preliminary/data/`, then run `preliminary/notebook.ipynb` from the `preliminary/` folder. The notebook is built for Kaggle (GPU T4 ×2) and also finds the data folder automatically when run locally.
- **Final.** Run `final/notebook.ipynb` from the `final/` folder. Everything runs on CPU in a few minutes. Outputs are written to `eda-output/` and `pipeline-output/`.

## Team

**IndomaretLabtekV**: Athilla Zaidan Zidna Fann, Riantama Putra

# Modelling Report: train.csv

## Setup
- Task: **classification** (2 classes), target `is_churn`, metric **roc_auc**
- Rows: 4,010 development, 990 holdout
- Validation: group 5-fold grouped by `customer_id`
- Target transform: None; class weights: True
- Features: 11 (7 numeric, 4 categorical)

## Decisions

| setting | value | source | why |
|---|---|---|---|
| TARGET | `is_churn` | user | target column analysed in the EDA, else sample_submission or the column missing from test |
| ID_COLS | `['customer_id', 'policy_id']` | eda | identifier columns (EDA, else recognised by name) |
| CV_GROUP | `customer_id` | eda | identifier repeated across rows, so copies must share a fold |
| TIME_COL | `None` | computed | no time column |
| DROP_COLS | `[]` | eda | constant or quasi-constant (one value >= 99%) |
| SENTINELS | `{}` | eda | placeholder values outside the normal range -> missing |
| COLUMN_ROLES | `2 num / 6 cat / 0 text` | eda | role inference shared with the EDA |
| TASK | `classification` | user | string or few-valued target |
| CLASS_WEIGHT | `True` | eda | imbalance ratio 26.0x |
| METRIC | `roc_auc` | eda | QWK for ordinal, F1 per label for multi-label, RMSE for regression, ROC AUC for binary, macro F1 for multiclass |
| DEDUP | `False` | eda | no identical rows in the EDA duplicate audit |
| CV_SCHEME | `group` | eda | repeated ids -> group, drift -> time |
| DROP_WEAK | `['age', 'marital_status', 'gender']` | eda | no mutual information and no significant association with the target |

## Leaderboard

| model | cv_roc_auc | cv_std | holdout_roc_auc |
|---|---|---|---|
| ensemble (selected) | 0.9872 | 0.0036 | 0.9911 |
| xgboost | 0.9870 | 0.0035 | 0.9888 |
| lightgbm | 0.9869 | 0.0031 | 0.9903 |
| catboost | 0.9859 | 0.0037 | 0.9912 |
| baseline | 0.9857 | 0.0043 | 0.9873 |

## Final result (holdout)
- ensemble: roc_auc = 0.9911 (95% CI 0.9861 to 0.9959)
- Baseline: 0.9873; naive: 0.5000
- Improvement over baseline: +0.4% (gain CI -0.0004 to +0.0090)
- All metrics: accuracy 0.9768, balanced_accuracy 0.9506, f1_macro 0.8320, f1_weighted 0.9798, logloss 0.0552, roc_auc 0.9911, pr_auc 0.6488, f1 0.6761
- Ensemble weights: baseline 0.00, lightgbm 0.55, xgboost 0.08, catboost 0.37

## Top features (xgboost, mean |SHAP|)
1. `tenure` (4.4489)
2. `payment_frequency` (1.0010)
3. `occupation_freq` (0.6726)
4. `occupation` (0.2934)
5. `payment_frequency_freq` (0.2554)
6. `insurance_type_freq` (0.2419)
7. `premium_freq` (0.1305)
8. `insurance_type` (0.1069)
9. `marital_status_freq` (0.0850)
10. `premium` (0.0743)

## Baseline top coefficients
- `tenure`: +4.4640
- `insurance_type_PA`: +4.1647
- `payment_frequency_SINGLEPREMIUM`: +4.1520
- `payment_frequency_MONTHLY`: +2.1294
- `payment_frequency_freq`: +1.5607
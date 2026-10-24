# EDA Summary: train.csv

## Dataset
- 5,000 rows x 11 columns, target `is_churn`, task **classification**
- Roles: 2 numeric, 6 categorical, 0 text, 2 id, 0 datetime, 0 constant

## Data quality
- Columns with NaN: 1 (worst: `occupation` 95.8%)
- Duplicates, identical ignoring IDs: 1,005 (20.1%)
- Duplicates, repeated customer_id: 296 (5.9%)
- Duplicates, same features, different target: 62 (1.2%)
- Heavily skewed numerics (|skew| > 1): ['tenure']

## Target
- 2 classes, imbalance ratio 26.0x

## Strongest feature-target associations (by mutual information)
- `tenure` (numeric): MI 0.062, Kruskal-Wallis stat 445.587, effect 0.089, p = 6.6e-99
- `payment_frequency` (categorical): MI 0.022, Cramer's V stat 0.198, effect 0.198, p = 2.7e-42
- `occupation` (categorical): MI 0.017, Cramer's V stat 0.248, effect 0.248, p = 1.7e-48
- `insurance_type` (categorical): MI 0.010, Cramer's V stat 0.171, effect 0.171, p = 3.6e-32
- `premium` (categorical): MI 0.007, Cramer's V stat 0.101, effect 0.101, p = 2.6e-11
- `age` (numeric): MI 0.003, Kruskal-Wallis stat 0.377, effect 0.000, p = 0.54
- `marital_status` (categorical): MI 0.000, Cramer's V stat 0.010, effect 0.010, p = 0.25
- `gender` (categorical): MI 0.000, Cramer's V stat 0.000, effect 0.000, p = 0.78

## Redundancy
- Pairs with |Spearman| >= 0.8: 0
- VIF > 10: [], max VIF 1.0 (`age`)

## Baseline (untuned GBM, out-of-fold)
- naive (majority class): macro_F1 0.491
- GBM random K-fold: macro_F1 0.763, ROC_AUC 0.979

## Exported files
- Figures (18): `eda-output\figures`
- Tables: `eda-output\tables`
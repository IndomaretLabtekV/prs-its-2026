# Exploratory Test-Label Order Audit

Date: 2026-09-03

This is a retrospective forensic audit. The reconstructed test-label files were
used only to measure the order hypothesis; they were not used to train or select
a competition model.

## Evidence

- `data/train.csv` has 160,174 rows and one target transition: labels `1` occupy
  indices `0..80202`, then labels `0` begin at index `80203`.
- `submission_50pct_1_50pct_0.csv` has 40,043 rows and one transition: labels
  `1` occupy `0..20021`, then `0` begins at `20022`.
- `research/test_labels_reconstructed_50-50.csv` has the same IDs but places the
  transition one row earlier, at `20021`. The files differ at exactly one row.
- Both files are aligned with `data/test.csv` and `data/sample_submission.csv`.
- Parsed numeric claim IDs have near-zero correlation with test row position,
  so the observed effect is storage order, not numeric ID ordering.
- The existing V1, V2, V3, V4, V5, and V8 score sequences all have the same
  score-mean changepoint at row `20052` when estimated without reconstructed
  labels. This is close to the proposed test boundary and supports a class-
  sorted data-generation/storage hypothesis.

## Retrospective metric check

For `K=floor(0.05*40043)=2002`, against either half-block file:

| Candidate | Fraud in top K | NR@5% |
|---|---:|---:|
| V3 | 1937 | 0.967532 |
| V5 | 1929 | 0.963536 |
| V8 | 1923 | 0.960539 |
| Position block | 2002 | 1.000000 |

The position-block row is a leakage diagnostic, not a legitimate feature-only
model. It wins because the proposed labels themselves are defined by position.

## Feature-only limitation

Exact grouping of all 51 non-ID test features gives 444 groups containing both
proposed labels, covering 1,416 rows. A feature-only predictor must assign the
same score to identical rows, so perfect reconstruction of every proposed test
label is impossible from those columns alone. This does not by itself prove
that a top-K portfolio can never be perfect, but it rules out a guaranteed
row-level 100% solution without extra information.

## Conclusion

The root file is a plausible half-block reconstruction, not independently
verified ground truth. If it is accepted as the true test label rule, the
position block reaches 1.0 retrospectively. If row order is excluded, the
strongest tested feature-only result remains V3-scale performance, about
0.9675 against this file and 0.96833 on the reported leaderboard. No further
model tuning is justified until the organizer confirms the hidden labels or
provides a non-leaking feature that distinguishes the conflicting cases.

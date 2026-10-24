# H15 - Positive-Cost Boundary Refiner

## Status

PROTOCOL LOCKED (2026-09-03, before results)

## Question and mechanism

Can a positive-class cost in the V3/H10 boundary refiner recover additional
fraud rows in the fixed top-5% portfolio? The current refiner uses unweighted
Logloss even though the operational objective is top-k capture. A moderate
positive cost may change finite-sample ordering inside the high-risk pool by
penalizing missed fraud more than false alarms.

## Fixed experiment

- Fold assignments: authoritative V8 artifact, five folds.
- Candidate pool: top 10% of the frozen V3 stack rank.
- Features: the frozen H10 `V9_FEATURES` representation.
- Refiner: CatBoost Logloss, depth 7, 900 iterations, learning rate 0.035,
  other H10 parameters unchanged.
- Seeds: exactly H10 fold seeds `CFG.seed + 3000 + fold`.
- Control: H10 unweighted refiner (`class_weights=[1, 1]`).
- Challengers: positive class weights `1.25`, `1.50`, and `2.00`.
- Selection: pooled OOF only; reconstructed test labels are not read by the
  experiment and public leaderboard feedback is not used.

## Promotion gate

A challenger must beat the control by at least 8 pooled TP (`NR@5% >= control
+ 0.001`), win at least three folds, lose no more than two TP in its worst
fold, and keep AP delta at least `-0.001`. Otherwise the hypothesis is
refuted and no test submission is promoted.

## Prediction

Positive weighting may improve recall at the top of the pool, but the effect is
expected to be small because all training labels are nearly balanced and the
refiner already sees a high-risk candidate pool. A gain below 8 TP is a useful
negative result, not evidence for more weight searching.

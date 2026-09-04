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
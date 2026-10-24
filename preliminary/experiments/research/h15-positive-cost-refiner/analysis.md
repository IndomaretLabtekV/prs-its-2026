# H15 Analysis - Positive-Cost Boundary Refiner

## Result

H15 is REFUTED. The experiment ran the locked three-arm protocol on the
authoritative five-fold OOF setup. The H10 V9 unweighted refiner remained the
control; no reconstructed test labels or public leaderboard feedback were
used.

| positive weight | NR@5% | TP | TP delta | NR delta | fold TP delta |
|---:|---:|---:|---:|---:|:---|
| 1.25 | 0.9724026 | 7787 | -9 | -0.0011239 | [-2, -3, 0, -1, -1] |
| 1.50 | 0.9720280 | 7784 | -12 | -0.0014985 | [-1, 0, -5, 0, 0] |
| 2.00 | 0.9721528 | 7785 | -11 | -0.0013736 | [0, -2, 1, -3, -4] |

All arms fail the pooled TP and NR gates. They also fail the fold-win and
worst-fold gates. AP stays effectively unchanged, so the degradation is a
top-k ordering effect rather than a broad ranking collapse.

## Interpretation

Positive class weighting does not recover the missing boundary positives in
this high-risk candidate pool. It changes the local ordering in a harmful,
non-robust direction. This rules out the pre-registered cost-sensitive variant
and does not justify a wider weight sweep.

## Artifacts

- `h15.log`: runtime log and completion marker.
- `h15_registry.csv`: pooled and fold-level metrics.
- `h15_scores.npz`: finite OOF score arrays for all three arms.

## Decision

Do not promote an H15 submission. Retain the V3/V8 feature-only result as the
best defensible result under the current data and leakage constraints.

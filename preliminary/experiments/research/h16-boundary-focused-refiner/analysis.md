# H16 Analysis - Boundary-Focused Refiner

## Result

H16 is REFUTED. The locked experiment completed on the authoritative V8
five-fold OOF setup using the frozen H10 V9 control. All score arrays are finite
and contain 160174 rows.

| alpha | NR@5% | TP | TP delta | NR delta | AP delta | fold wins | worst fold | promote |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 0.9720280 | 7784 | -12 | -0.0014985 | -0.0001535 | 1 | -6 | false |
| 3 | 0.9701548 | 7769 | -27 | -0.0033716 | -0.0003112 | 2 | -14 | false |
| 7 | 0.9696553 | 7765 | -31 | -0.0038711 | -0.0005893 | 0 | -12 | false |

All arms fail the pooled TP, NR, fold-win, and worst-fold gates. AP remains
inside the allowed degradation limit for each arm, so the loss is concentrated
in top-k ordering rather than a complete ranking collapse.

## Interpretation

Boundary-focused sample weighting is worse than the unweighted H10 V9
refiner. Increasing emphasis on rows near the nominal 5% boundary makes the
local ordering less stable, with degradation growing as alpha increases. This
also rules out the idea that H15 failed only because its global class weight
was too broad.

## Artifacts

- `h16.log`: runtime log with data, fold, pool, and completion checks.
- `h16_registry.csv`: pooled metrics and persisted gate fields.
- `h16_scores.npz`: finite OOF score arrays for alpha 1, 3, and 7.
- `protocol.md`: pre-run locked protocol.

## Decision

Do not promote an H16 submission. Do not expand the alpha or boundary-band
sweep. Retain V3/V8 as the best defensible feature-only result under the
current data and leakage constraints.

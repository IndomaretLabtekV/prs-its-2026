# H13 — Kapasitas & seed refiner-v2 (fitur V9)

## Status: PROTOCOL LOCKED (2026-09-03, sebelum hasil)

## Pertanyaan + mekanisme
Refiner boundary (V3) = CatBoost depth 7, 900 iter, lr 0.035 — tidak pernah
di-tune sejak konsepnya. H10 membuktikan features+v9 memberi +2 TP pada config
yang sama; hipotesis berikutnya: refiner UNDERFIT pada pool 16k, sehingga
menambah iterasi memperbaiki ordering boundary.

## Eksperimen (one lever at a time, eval identik H10)
- Control: refiner-v2 900 iter (0.97353 / 7796 TP, hasil H10 sudah ada).
- H13a: refiner-v2 1800 iter (lr 0.035).
- H13b: refiner-v2 2700 iter (lr 0.035).
- H13c: seed-ensemble refiner-v2 (3 seed @900 iter, rerata prob) — varian non-capacity.
Semua: pool top-10% dari STACK_RANK, 5-fold, GPU single, fitur V9_FEATURES.

## Prediksi
0 <= delta <= +0.0006 (2-5 TP) -> catat, tapi gate +0.001 tetap tak tercapai →
CONCLUDE final. Jika delta >= +0.001 (kejutan positif, > 8 TP) -> promosi V10.

## File
- code/run_h13.py (reuse run_h10, parameter iter & seed count)
- results disimpan: h13_scores.npz + table print OOF

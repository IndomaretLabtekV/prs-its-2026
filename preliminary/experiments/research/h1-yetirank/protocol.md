# H1 — YetiRank mengalahkan Logloss pada ordering top-5% OOF

## Status: PROTOCOL LOCKED (belum ada hasil saat ditulis)

## Pertanyaan
Apakah mengganti `loss_function='Logloss'` → ranking loss (YetiRank) pada
CatBoost anchor yang identik (fitur, fold, seed, iterasi) menaikkan pooled OOF
NormalizedRecall@5%?

## Mekanisme
Logloss mengoptimasi probabilitas global; YetiRank mengoptimasi ordering
(NDCG-like). Metrik kompetisi murni ranking top-K → objektif yang selaras
seharusnya memindahkan TP melintasi boundary K. 205 FP "tenggelam dalam"
tak tersentuh rerank, tapi boundary ±250 rank (spread skor ~0.003) masih
menyimpan 471 fraud near-miss — di situlah YetiRank bekerja.

## Prediksi terkunci
- Reproduksi baseline (Logloss, seed 2127, 5 fold, 2860 iter, depth 8, lr 0.03):
  NR ≈ 0.9705 (±0.001 toleransi env: versi catboost/CUDA boleh menggeser).
- Treatment (YetiRank, `CatBoostRanker`, `group_id` = fold asal baris):
  **NR ≥ baseline + 0.001** → LANJUT ke submission via sigmoid monotonik.
- Jika 0 < delta < 0.001 → netral, arsipkan, lanjut H2.
- Jika delta ≤ 0 → H1 GUGUR, catat sebagai negatif, lanjut H2.

## Desain (apple-to-apple)
- Fitur: FULL_FEATURES verbatim dari sel 05 notebook V8 (clean + engineer sama).
- Fold: `primary_fold_assignments.csv` artefak V8 (5 fold, overlap 0).
- Kedua arms pooled-OOF tanpa kalibrasi silang (ranking only; kalibrasi monotonik
  hanya di tahap submission dan order-preserving → tak memengaruhi NR).
- Metrik: pooled OOF NR@5% (K=8008) + TP + AP diagnostik. Tanpa menyentuh test
  untuk seleksi (test dipakai hanya untuk submission final bila lolos).

## Tahap
1. SMOKE: 1 fold (valid fold 0), 300 iter, kedua arms. Syarat lanjut: jalan di
   GPU 6GB, waktu/iter terukur, tidak NaN.
2. FULL: 5 fold × 2860 iter × 2 arms di background, estimasi dari smoke.
3. Keputusan gate di atas. Semua angka OOF tercatat sebelum submission apa pun.

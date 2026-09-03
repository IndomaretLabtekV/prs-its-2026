# Research Log — Road to 0.98+ (LB saat ini 0.96833, OOF 0.97303)

## 2026-09-03 — Bootstrap + error analysis OOF (temuan kunci)
- Bongkar `results/v8/results.zip` (17 file, termasuk `submission_v8.csv` + `selected_config.json`).
- Seleksi V8 = **V3 control fallback** (0.97303, 7792 TP). 12/12 challenger gagal gate_1+gate_2.
- Error analysis OOF pooled (reproduksi 7791/7792 TP — selisih 1 = tie-break boundary, tak material):
  1. **205/217 FP "tenggelam dalam"** (skor > skor rank-7758). Bukan masalah boundary → menjelaskan kegagalan V8.
  2. **Penalti post-hoc subgroup KM/kdkc-2101-2103/dati2-309 GAGAL** (+1 TP lalu negatif). Sebab: 1418 fraud top-K se-subgroup — subgroup itu high-risk (fraud rate 0.567 vs 0.498), penalti menghukum fraud juga. Eksperimen negatif, tercatat.
  3. Interaksi umur/los × subgroup searah base-rate (bukan bias koreksibel post-hoc).
  4. **Label noise struktural**: ~3800 grup fitur-identik berlabel campur (analisis eksploratori, fitur mentah) — menekan ceiling di bawah 1.0.
- Target 0.98 = +56 TP; 0.985 = +98 TP. Ceiling oracle +216 TP.

## Keputusan arah (outer loop #1): DEEPEN via fungsi objektif ranking
- Jalur rerank/post-hoc: JENUH (bukti V8 + poin 1-2). Jalur fitur ad-hoc: ditunda.
- H1 (prioritas 1): CatBoostRanker + YetiRank — satu-satunya perubahan yang menyentuh
  ordering top-K secara langsung. Prediksi: +0.001–0.004 OOF NR pooled.
- H2 (cadangan): fitur interaksi eksplisit subgroup×clinical + retrain logloss.
- Batas: tanpa LB probing; seleksi murni OOF; submit hanya jika H1 lolos kriteria.

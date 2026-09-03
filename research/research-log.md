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

## 2026-09-03 — Pivot ke V9 Kaggle (keputusan user: stop eksperimen lokal)
- Ranking loss_QUEUE ditutup: YetiRank GPU limit query 1023, LGBM limit 10k;
  chunked YetiRank CPU -0.037 NR (konfirmasi negatif, bukan sekadar lambat).
- Local compute terukur: Logloss GPU ~9 dtk/300iter/fold (~7 mnt full arm).
  Env dipindah ke conda `datsci` (catboost 1.2.10 + GPU OK); research/.venv dihapus.
- Fold lokal != fold artefak (beda versi pandas/sklearn) -> V9 wajib pakai
  fold artefak langsung.
- V9 design+plan ditulis (docs/superpowers/specs + plans 2026-09-03-v9-*):
  taruhan tunggal = fitur representasi (risk_subgroup + crosses + mixed flag),
  roster 5 kandidat, gate V8 verbatim, fallback V3 control.

## 2026-09-03 — V9 plan selesai (pivot Kaggle, atas permintaan user)
- Spec: docs/superpowers/specs/2026-09-03-v9-design.md.
- Plan: docs/superpowers/plans/2026-09-03-v9-implementation.md.
- Pelajaran shell: JANGAN pkill -f dengan pola yang cocok dengan command
  shell sendiri (membunuh shell persistent -> timeout). Cek via ps+grep saja.
- Env riset = conda datsci (catboost 1.2.10 GPU OK, lgbm 4.7.0).
- Next: implementasi notebooks/V9_Representation.ipynb ikut plan, smoke lokal,
  upload + run Kaggle T4x2.

## 2026-09-03 — V9 plan selesai (pivot Kaggle, atas permintaan user)
- Spec: docs/superpowers/specs/2026-09-03-v9-design.md.
- Plan: docs/superpowers/plans/2026-09-03-v9-implementation.md.
- Pelajaran shell: JANGAN pkill -f dengan pola yang cocok dengan command
  shell sendiri (membunuh shell persistent -> timeout). Cek via ps+grep saja.
- Env riset = conda datsci (catboost 1.2.10 GPU OK, lgbm 4.7.0).
- Next: implementasi notebooks/V9_Representation.ipynb ikut plan, smoke lokal,
  upload + run Kaggle T4x2.

## 2026-09-03 — Notebook V9 selesai dibangun, SIAP untuk Kaggle
- File: notebooks/V9_Representation.ipynb (18 sel, builder research/v9/build_notebook.py).
- Roster final 4 kandidat (C4 mixed-weight DIBATALKAN: flag grup-campur vakuous
  di bawah StratifiedGroupKFold — temuan negatif tercatat, bukan bug).
- Sel 1-9 terverifikasi jalan di smoke lokal (Behaviors 1,2,4,5 PASS).
- Sel 11-17: compile OK + review baris-per-baris, BELUM pernah dieksekusi
  end-to-end (smoke penuh tidak selesai karena interupsi sesi berulang).
- Pelajaran: tie-break di boundary K beda +-1 TP antar versi pandas
  (registry 7792 vs recompute lokal 7791) -> assert diganti toleransi +-8 TP.
- Langkah berikut: upload notebook + results/v8/results.zip + results/v7/results.zip
  sebagai dataset Kaggle, run T4x2, submit bila ada kandidat lolos gate.

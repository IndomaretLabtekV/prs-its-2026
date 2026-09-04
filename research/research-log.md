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

## 2026-09-03 — OUTER LOOP: H10/H11/H12 selesai (resume autoresearch)
- H10 refiner-v2 (fitur V9 di refiner V3): control rebuild NR=0.97328 TP=7794,
  refiner-v2 NR=0.97353 TP=7796 -> DELTA +0.00025 / +2 TP. ARAH BENAR tapi
  < gate 0.001. Menarik: rebuild bersih (5-base stack, iter 900) menghasilkan
  7794 vs registry 7792 (tie-break rebuild) — konsistensi toleransi OK.
- H11 fold 2/4: distribusi kovariat antar fold IDENTIK (top kdkc/dati2/typeppk/
  cmg/umur/los sama) -> lemahnya fold 2/4 SIFAT SUBSET, bukan bias penyebaran.
  Tidak ada perbaikan fold yang sah.
- H12 boundary-rank blend V3 x refiner-v2: w terbaik di selection=0.25
  (selNR 0.97500) GAGAL transfer ke confirmation (confNR 0.97070 < ctrl 0.97194).
  REFUTED.
- Ceiling kuantitatif: grup identik-berulang berlabel campur = satu-satunya TP
  yang mustahil dipisahkan; dengan feature-ceiling, headroom oracle +216 TP TAPI
  oracle = memorisasi (hampir semua grup singleton -> tidak actionable).
- VERDICT: 5 tembakan independen (V8 surgical, V9 global, V9 mixed-canceled,
  H10 refiner-v2, H12 blend) SEMUA memuncak di +2 TP vs kontrol. 0.96833 = hasil
  final yang layak; 0.98+ bukan risiko yang bisa dibeli dengan pendekatan ini.

## DIRECTION: CONCLUDE (negative-result matrix lengkap) — open sisa H13 (refiner
## seed-ensemble) hanya jika user ingin 1 percobaan terakhir berbasis bukti;
## rekomendasi gue: tidak.

## 2026-09-03 — accuracy-improvement-loop: diagnosa ulang (Step 1)
- Re-audit leakage: 0 (17,906 identik = label noise, split bergrup benar).
- FP top-K: umur 41.3 vs TP 39.7 (kecil); LOS FP 0.63 vs TP 0.41 (muncul sinyal
  LOS panjang pada FP — tapi rentang korelasi tidak besar; tanpa konfirmasi).
- Gap yang belum pernah disentuh: V3 refiner 900 iter TIDAK PERNAH di-tune
  (kapasitas) sejak konsepnya di V3; H10 membuktikan fitur memberi +2 TP,
  berarti ruang perbaikan masih ada di sisi kapasitas/seed refiner.

## 2026-09-03 — H13 REFUTED: kapasitas refiner = 900 adalah optimum
- it1800: 0.97253 (-8 TP) | it2700: 0.97228 (-10 TP) | seed-ens: 0.97315 (-3 TP).
- Intepretasi: refiner @pool 16k OVERFIT dengan iterasi lanjutan; 900 ITER = optimum.
- Final standings (semua levers diuji): fitur +2 (H10) > ctrl rebuild 0 TP >
  seed-ens -3 > fitur-global (V9) -3..-4 > blend -2..-4 > iter-lain -8..-10.
- MATRIX LENGKAP dari semua tembakan: V8 surgical (12 challenger 0 lolos gate),
  V9 global (0 lolos), V9 mixed (dibatalkan, vakuous), H10 refiner-v2 (+2 TP,
  di bawah gate), H12 blend (refuted), H13 kapasitas+seed (refuted).
- KESIMPULAN FINAL: 0.96833 (V3 control) = ceiling; 0.98+ tidak terjangkau
  dengan representasi & data yang ada (official_data_only, no external).

## 2026-09-03 — Exploratory retrospective test-order audit
- Train target has one storage-order transition at index 80203. The supplied
  root test file has one transition at index 20022; the prior reconstructed file
  transitions at 20021 and differs at exactly one row.
- Exact non-ID test-feature grouping under the proposed labels gives 444
  mixed-label groups covering 1,416 rows. This is evidence against perfect
  row-level feature-only reconstruction, not a formal top-K impossibility proof.
- A changepoint estimated from each existing test score sequence alone is row
  20052. It is close to the proposed boundary and supports the class-sorted
  storage hypothesis.
- Against either half-block file, V3 gets 1937/2002 top-K fraud rows (NR
  0.967532); the position-block diagnostic gets 2002/2002 (NR 1.0). This is
  leakage confirmation, not a promotion candidate or verified ground truth.
- Decision: keep the competition result at V3/V8 and stop model compute unless
  the label source is independently confirmed or new non-leaking information is
  supplied.

## 2026-09-04 - H15 positive-cost boundary refiner REFUTED
- Locked protocol ran successfully on the authoritative V8 five-fold splits,
  using the frozen H10 V9 candidate pool and no reconstructed test labels.
- Positive weights 1.25, 1.50, and 2.00 produced NR@5% 0.9724026,
  0.9720280, and 0.9721528, with TP deltas -9, -12, and -11 versus H10 V9.
- Fold TP deltas were [-2,-3,0,-1,-1], [-1,0,-5,0,0], and [0,-2,1,-3,-4].
  Every arm failed the pooled, NR, fold-win, and worst-fold promotion gates.
- Interpretation: class weighting changes the high-risk-pool ordering in a
  harmful direction; it does not recover the missing top-k positives.
- Decision: no H15 submission and no wider weight sweep. V3/V8 remains the
  best defensible official-data-only result; model compute is concluded unless
  organizer-confirmed labels or new non-leaking information arrives.

## 2026-09-04 - H16 boundary-focused refiner REFUTED
- Locked H16 ran successfully on the authoritative V8 five-fold splits with
  frozen H10 V9 features, stack, candidate pool, and seeds. No test labels,
  row-order signal, or test submission were used.
- Boundary sample weights with alpha 1, 3, and 7 produced NR@5% 0.9720280,
  0.9701548, and 0.9696553, with TP deltas -12, -27, and -31 versus H10 V9.
- Fold TP deltas were [-3,-3,-4,1,-6], [1,2,-9,-6,-14], and
  [-4,-1,-12,-10,-6]. All arms failed pooled TP, NR, fold-win, and worst-fold
  gates; AP stayed within the allowed degradation limit.
- Interpretation: emphasizing the nominal boundary destabilizes local
  ordering, and the effect worsens with alpha. This closes the boundary-focused
  weighting branch and makes a wider alpha/band sweep unjustified.
- Decision: no H16 submission. V3/V8 remains the best defensible
  official-data-only result; model compute is concluded unless organizer-
  confirmed labels or new non-leaking information arrives.

# H10 — Refiner-v2: fitur V9 di dalam refiner boundary V3

## Status: PROTOCOL LOCKED (2026-09-03, sebelum hasil)

## Pertanyaan
V3 control = CatBoost refiner (depth 7, 900 iter, top-10% pool berdasarkan
STACK_RANK) yang rerank stack score. Fitur refiner = FULL_FEATURES (81 kolom).
Hipotesis: refiner dengan V9_FEATURES (85 kolom = 81 + risk_subgroup + 3
interaksi klinis) menghasilkan NR pooled lebih tinggi, karena:
1. V9 melakukan topping +0.002 vs anchor GLOBAL — sinyal subgroup nyata.
2. Refiner memandang SEMUA kolom -- fitur interaksi memberi konteks yang
   CTR CatBoost global depth-8 mungkin tidak eksploitasi di pool top-10%.

## Mekanisme & prediksi terkunci
- Baseline kontrol: V3 control rebuild (stack all-5 base preds → logit/rank
  meta → logistic stack → boundary refiner FULL_FEATURES) ≈ 0.9730 (±0.0003).
- Treatment: identik, hanya features=FULL_FEATURES+V9_NEW.
- Prediksi: **NR_treatment >= NR_control + 0.001** → submis prematur
  (membuat V10 = V9 + refiner-v2, run Kaggle). Jika 0 <= delta < 0.001 →
  netral, arsipkan; H12 (blend rank boundary) berikutnya. Jika delta < 0 →
  H10 GUGUR; kesimpulan "boundary memang jenuh" (CONCLUDE-lane).

## Desain (apple-to-apple)
- Data & fold: artefak V8 (primary_fold_assignments owner, overlap 0).
- Base preds: semua 5 prediksi dari results/v7/results.zip (2127, 2026,
  xgb, lgbm, logistic) — replikasi persis stack V8.
- STACK: logistic logit-blend (P = Pipeline scale + LogReg C=1, max_iter 700),
  crossfit 5-fold; STACK_RANK_OOF = fold-local percentile.
- Refiner: CatBoost Logloss depth 7, lr 0.035, l2 10.0, 900 iter, seed 3000+fold,
  fit per-fold di pool top-10% train; OOF score = boundary_rank_score(
  STACK_RANK_OOF, ref_oof, mask, rate=0.10, weight=1.0) — persis V3.
- Treatment: sama persis tapi features=V9_FEATURES (train_v9).
- Metric: pooled NR@5% K=8008 (metric_row/top_k_mask netral), TP, AP.
- Toleransi tie: bandingkan relativ (kontrol rebuild), bukan vs registry.
- Env: conda datsci, CUDA_VISIBLE_DEVICES=0.

## Tahap
1. Salin skrip eksperimen research/experiments/h10-refiner-v2/run_h10.py
   (jangan eksekusi sebelum commit protokol ini).
2. Full 5-fold x 2 arms; durasi estimasi < 15 menit (refiner 900 itr @16k pool).
3. Output: tabel NR/TP/AP per arm + delta. Simpan juga
   h10_refiner_v2_oof.npz untuk H12 (blend cadangan).

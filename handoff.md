# Handoff — DAC 2026 PRS-ITS (Fraud Detection)

> Dibuat 2026-09-03. Untuk agent baru / sesi berikutnya. Detail desain & plan lengkap ada di:
> - Spec: `docs/superpowers/specs/2026-09-02-v8-robust-boundary-design.md`, `docs/superpowers/specs/2026-09-03-v9-design.md`
> - Plan: `docs/superpowers/plans/2026-09-02-v8-robust-boundary-implementation.md`, `docs/superpowers/plans/2026-09-03-v9-implementation.md`
> - Riset: `research/research-log.md`, `research/findings.md`, `research/research-state.yaml`
> - Graph: `graphify-out/` (jalankan `graphify query "<q>"` — graph 321 node, build tanggal 2026-09-03)

## Satu Paragraf
Kompetisi DAC 2026 (fraud claim BPJS): 160.174 baris train / 40.043 test.
Celine: audit 5% teratas → metrik `NormalizedRecall@5%`. Kandidat final **V3 control**:
CatBoost refiner boundary top-10% di atas logistic stack dari 5 base model (CatBoost 2127/2026,
XGBoost, Logistic, LightGBM), validasi `StratifiedGroupKFold` per signature fitur eksak
(overlap 0), **OOF 0.97303 (7792 TP) / LB 0.96833**. Seluruh riset kenaikan skor ke 0.98+
**selesai dan GAGAL mencapai gate** — hasil final diyakini ceiling.

## Kebenaran Penting (jangan dilanggar)
- `claim_id` = alignment only, **bukan fitur**. Row position bukan fitur. `ground_truth` di-reject.
- Seleksi kandidat MURNI OOF, tidak ada probing public LB.
- Fold otoritatif = `results/v8/results.zip` (primary_fold_assignments.csv). Recompute lokal
  match dengan toleransi ±1 TP di boundary (tie-break versi pandas) → jangan assert bitwise NR.
- KV V9: mixed-signature flag is VAKUOUS di bawah StratifiedGroupKFold (tiap signature 1 fold).
- Ranking loss (YetiRank/LambdaRank) infeasible: limit query GPU 1023 / 10000 vs grup 32k;
  chunked YetiRank = -0.037 NR.
- Jangan pkill -f dengan pola yang cocok command shell sendiri (pernah bunuh shell).
- Env riset lokal = `conda run -n datsci python ...` (catboost 1.2.10, GPU RTX 4050 6GB,
  lgbm 4.7.0, sklearn 1.7.2, pandas 2.3.3).

## Semua yang Sudah dicoba (matriks eksperimen)

### Baseline & validasi (fondasi)
| Hal | Detail | Hasil |
|---|---|---|
| Data forensics | row-order artifact (desil 0-4 = 100% fraud), 17.906 baris duplikat signature (6.747 grup), claim_id 160k cardinality | N/A |
| Fraud rate | 80.203 fraud / 79.971 legit (50.07%) | N/A |
| Train-test shift | maks univariate 0.014 (proc39_45 0.0133); adversarial AUC 0.507 | N/A |
| Grouped validation | signature-grup, overlap 0; naif = 13.100 overlap | dipakai |
| V1 | notebook 72 cell, CatBoost + kalibrasi sigmoid, baseline LB | LB 0.95833 |
| V2 | 79 cell, V1-reconstruction, corrected ordinal encodings, CatBoost 2 seeds, stack | (V2 ancestor ke V3) |
| V3 | boundary refiner (top-10%) + stack; **= V3 control final** | OOF 0.97303 / LB 0.96833 |
| V4 | boundary_bands config, cat_mean 0.55/0.45 | +3 TP (7795) |

### Eksperimen kenaikan skor (semua di bawah gate /0.001)
| ID | Hipotesis | Hasil | Δ TP | Verdict |
|---|---|---|---|---|
| V8 (12 challenger) | HN full/proxy-light specialist + surgical rerank (2 band × 3 weight) | 0/12 lolos gate_1+2 (terbaik +3 TP, +0.00037) | +3 | REFUTED |
| V9 global interact | risk_subgroup + 3 interaksi klinis (2 seed, 85 fitur) | Interact-2127 0.97265, 2026 0.97178; fold 2 -11 TP | -3/-4 | REFUTED |
| V9 seed-mean3 | rata 3 seed (2127+2026+3407) | 0.9715, gagal gate_4 (worst -8) | -1..-3 | REFUTED |
| V9 mixed-weight | bobot 0.5 pada grup campur | **DIBATALKAN saat smoke**: flag vakuous | - | CANCELLED |
| H1 YetiRank | ranking loss (CatBoostRanker/LambdaRank) pada grup 32k | chunked -0.037 NR; GPU limit query | -30.. | REFUTED |
| H10 refiner-v2 | V9 features DI DALAM refiner (900 it, bukan model global) | 0.97353 / 7796 vs ctrl-rebuild 0.97328 / 7794 | +2 | paling dekat, di bawah gate |
| H11 fold 2/4 | diagnosa kovariat | semua fold IDENTIK (kdkc/dati2/typeppk/cmg/umur/los) — lemah = sifat subset | - | NO-FIX |
| H12 boundary blend | rank-blend V3 × refiner-v2 (w=0/0.25/0.5/0.75/1) | best-selection w=0.25 gagal transfer conf-fold (0.9707 < 0.9719) | -2..-4 | REFUTED |
| H13 kapasitas+seed | iterasi refiner 1800/2700; seed-ensemble 3× | it1800 0.97253 (-8), it2700 0.97228 (-10), seed-ens 0.97315 (-3) | -8..-10 | REFUTED (900 = optimum, overfit) |
| Ceiling | oracle bound | FP confident 205/217; FP umur 41.3 vs TP 39.7, LOS 0.63 vs 0.41 (tidak bisa bedakan) | - | CEILING |

### Penemuan penting selamat riset
1. **V3 control rebuild beda-versi pandas**: registry 7792 vs recompute 7791 (tie order).
   Fix: gate toleransi (±8) + rebuild boleh beda ±1, array checkpoint jadi acuan.
2. **Refiner lebih kuat dari global model + fitur baru** — jalan terakhir harus ke REFINER.
   (V9 global gagal; H10 refiner-v2 paling dekat.)
3. **Boundary jenuh**: semua challenger mati di gate_1 (gain ≥ +0.001 = 8 TP) dan gate_2
   (fold-wins ≥ 3).
4. **Label noise struktural**: 3.800 grup signature identik campur (12.867 baris) → ceiling riil
   di bawah 1.0.
5. Output jujur: V9 di-run di Kaggle T4x2 (~32 menit) — discovery dataset flat user
   (`v8-artifacts-results/`) adalah fix yang valid (mock + test OK).

## Artifacts & file penting
- **Final submission**: `results/v1/submission.csv` TIDAK final. FINAL = `results/v8/results.zip`
  → `v8_artifacts/submission_v8.csv` (V3 control, LB 0.96833). V9 = `submission_v9.csv` (PASS
  tapi identik fallback V3).
- **Checkpoint**: `results/v8/results.zip` (17 file: v8_scores.npz, registry, fold assign,
  stress, churn...) dan `results/v7/results.zip` (base preds 5 model) → di-upload user sebagai
  dataset Kaggle `v8-artifacts-results/` + `v7-checkpoint/v7_artifacts/`.
- **Notebook**: `notebooks/V8_Robust_Boundary.ipynb` (18 sel, final, 25-30 menit T4x2),
  `notebooks/V9_Representation.ipynb` (18 sel, 4 kandidat, 3 arm) — V9 paling baru & bersih utk
  base masa depan.
- **Builder**: `research/v9/build_notebook.py` (regenerate V9; patchnya sudah benerin:
  discovery flat, assert Series, probability_logit, roster 4, 8 gates).
- **H10/H12/H13 runner**: `research/experiments/h10-refiner-v2/run_h10.py`,
  `h12-rank-blend/run_h12.py`, `h13-refiner-capacity/run_h13.py` (scores di NPZ + logs).
- **RIWAYAT EKSPERIMEN FULL**: `research/research-log.md`, `research/findings.md`,
  `research/research-state.yaml` (semua commit penelitian ada di git log).

## Status & Rekomendasi
- **Submisi final**: `submission_v8.csv` sudah disubmit → LB 0.96833.
- **Eksperimen scoring SELESAI (CONCLUDE)**: semua 6 levers diuji, tidak ada yang lolos gate.
  JANGAN ulangi. Sisa: H13-v2 (bukan bagus), dsb.
- **Rekomendasi**: (1) stop, tulis laporan kompetisi (V8 jadi basis + tabel eksperimen),
  (2) kalau mau eksplorasi lebih lanjut: data eksternal/teks DILARANG (`official_data_only`),
  (3) untuk laporan: gunakan `submission_v8.csv` + `selected_config.json` + registry.

## Suggested Skills
- `rigorous-evaluation` — jika menganalisis metrik/angka baru (umum penting).
- `ml-research-methodology` — jika meneruskan riset kenaikan skor (harus mulai dari diagnostik,
  bukan trik).
- `data-rigor-and-leakage` — jika ada dataset/versi baru (group split + fold-safe TE).
- `graphify` — jika ingin query graph pengetahuan yang sudah dibangun (`graphify-out/`).
- `notebook-delivery` — jika membuat/mengubah notebook Kaggle baru (style + env).

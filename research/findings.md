# Findings — DAC 2026 PRS-ITS

## Current Understanding
Competisi = ranking fraud BPJS @5% audit capacity (NR@5%).
V3 control (CatBoost refiner atas stack top-10%) adalah kandidat terkuat yang
pernah ada: OOF NR 0.97303 (7792 TP, 216 legit audits, AP 0.82512) -> LB 0.96833.
V8 (12 surgical rerank challengers) dan V9 (interaksi global 2 seed + seed-mean3)
GAGAL lolos gate: keduanya menempel kuat ke ide "boundary jenuh".

## Patterns and Insights
- 205/217 FP di top-K adalah FP CONFIDENT (skor di atas level rank-7758):
  bukan masalah ordering — rerank apa pun tidak bisa menyentuh mereka.
- FP menumpuk di subgroup high-risk (KM, kdkc 2101/2103, dati2 309, umur tua,
  LOS pendek) yang fraud rate-nya PUN tinggi (0.567 vs 0.498): model tak salah.
- Fitur interaksi global MEMANG terlihat menambah sinyal vs anchor mentah
  (+0.002, 0.97053->0.97265) tapi masih di bawah V3 refiner (0.97303).
- Fold 2 & 4 loss TP terparah di SEMUA kandidat (-8..-11): homogenitas
  validasi, bukan artefak model. Perlu diagnosa kovariat.
- Label noise struktural: ~3800 grup signature identik berlabel campur
  (~12.8k baris) = ceiling riil di bawah 1.0.

## Lessons and Constraints
- Jangan pernah memilih lewat public LB: gap OOF->LB 0.0047 wajar (tie-order
  beda versi pandas pun menggeser +-1 TP -> pakai toleransi gate, bukan
  assert bitwise).
- Ranking loss di grup 32k infeasible (limit query GPU) & chunked = buruk.
- StratifiedGroupKFold membuat feature-flag per level-signature VAKUOUS
  (group selalu utuh dalam satu fold) -> jangan ulangi ide itu.
- Semua hyperprameter = warisan V7/V8; hyperparameter sweep bukan tuas di sini.
- Refiner boundary (V3) lebih kuat dari model global + fitur baru: target
  berikutnya harus menyentuh REFINER, bukan model global.

## Open Questions
- H10, H11, H12, dan H13 sudah selesai; hasilnya tercatat di
  `research/research-log.md`.
- Tidak ada jalur feature-only yang belum diuji dan punya bukti cukup untuk
  target public LB 0.985 di bawah constraint saat ini.

## Retrospective Test-Order Audit (2026-09-03)
- `data/train.csv` has one label transition: fraud indices `0..80202`, then
  legitimate rows. This confirms a storage-order defect in the released train
  file.
- The root proposed test file is exactly `1` for indices `0..20021` and `0`
  thereafter. The earlier reconstructed file differs at one odd-row boundary
  only (`20021`). Neither is independently verified private ground truth.
- Exact grouping of the 51 non-ID test features finds 444 mixed-label groups
  covering 1,416 rows under the proposed labels. This blocks perfect
  row-level feature-only reconstruction, although it is not by itself a proof
  that NR@5% cannot equal 1.
- A score-sequence-only mean changepoint detector estimates row `20052` for
  every existing submission. This is independent support for class-sorted test
  storage, but it is still a leakage diagnostic.
- Retrospective position-block selection captures `2002/2002` top-K rows
  (`NR@5%=1.0`); V3 captures `1937/2002` (`0.967532`). The former must not be
  presented as a legitimate model improvement.

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
- Apakah menambah fitur V9 ke refiner boundary (bukan ke model global)
  menembus +0.001? (H10, sedang diuji)
- Mengapa fold 2/4 lemah — distribusi kovariat, atau isi fold yang memang
  tersulit? (H11)

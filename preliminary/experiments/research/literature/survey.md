# Literature Survey - Fraud Risk Ranking at Fixed Audit Capacity

Date: 2026-09-03

## Question

What defensible methods can improve a tabular fraud-risk ranking at a fixed
top-k audit budget, when the local DAC data contain repeated conflicting
feature signatures and no verified additional labels?

There is no directly comparable published benchmark using the DAC schema,
50-percent sampled prevalence, exact-signature grouping, and the same
NormalizedRecall@5% definition. The sources below therefore inform method
choice, not a claimed DAC state of the art.

## Evidence Summary

| Source | Setting | Result or method | Relevance and caveat |
|---|---|---|---|
| Fraud Detection Handbook (2022) | ULB simulator, chronological validation | Fixed-budget P@k and card-level metrics | Supports evaluating the audit portfolio directly; toy data are not DAC-comparable. |
| Dal Pozzolo et al. (2018) | Worldline stream, delayed labels and drift | P@k, cumulative precision, AUC | Supports time-aware feedback handling; proprietary card data differ from DAC. |
| Carcillo et al. (2018) | Worldline stream, active learning | Querying delayed feedback improves top-100 precision | Requires feedback and temporal labels not present here. |
| Paldino et al. (2022/2024) | Large Worldline streams | Diversity-based ensembles for P@100 | Supports small complementary ensembles; prior DAC ensemble tests plateaued. |
| Tittelfitz (2023) | IEEE-CIS and fraud datasets with class-conditional noise | Micro-model label-noise estimation | Suggests independent label auditing; no verified clean-label sample exists here. |
| Zhang et al. (2021) | Alipay CCT and PAF with multi-source noisy labels | Label aggregation/correction | Requires multiple label sources; the DAC project has one observed target. |

## Practical Conclusions

1. Use fixed-k capture, AP/PR-AUC, and calibration together; ordinary accuracy
   is not the objective.
2. Use chronological or group-safe validation when the data-generating process
   has time or repeated entities. Random row splits are unsafe here.
3. Label correction can help only with independent evidence. It cannot separate
   rows whose observable features are identical but whose labels conflict.
4. Feedback, active learning, or external labels are plausible future levers,
   but they are unavailable under the current `official_data_only` constraint.
5. A dedicated ranking loss is low priority after the local YetiRank/LambdaRank
   experiment was infeasible or degraded.

## Sources

- [Fraud Detection Handbook: top-k metrics](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_4_PerformanceMetrics/TopKBased.html)
- [Fraud Detection Handbook: validation strategies](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_5_ModelValidationAndSelection/ValidationStrategies.html)
- [Dal Pozzolo et al. (2018), DOI](https://doi.org/10.1109/TNNLS.2017.2736643)
- [Carcillo et al. (2018), DOI](https://doi.org/10.1007/s41060-018-0116-z)
- [Carcillo et al. (2018), arXiv](https://arxiv.org/abs/1804.07481)
- [Paldino et al., PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9516537/)
- [Paldino et al., DOI](https://doi.org/10.1007/s11634-022-00515-5)
- [Tittelfitz (2023), arXiv](https://arxiv.org/html/2308.02695)
- [Zhang et al. (2021), DOI](https://doi.org/10.1145/3459637.3482433)

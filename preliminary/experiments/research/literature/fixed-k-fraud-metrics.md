# Fixed-k Fraud Metrics

- **Source:** Fraud Detection Handbook, Le Borgne et al., 2022.
- **URL:** https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_4_PerformanceMetrics/TopKBased.html
- **Method:** Precision-at-k and cumulative/card-level fixed-budget evaluation.
- **Finding:** A fraud system should be evaluated by the cases selected for a
  constrained investigation budget, not only by a threshold metric.
- **DAC relevance:** Supports `NormalizedRecall@5%`, TP@K, and explicit tie
  handling as the primary local evaluation.
- **Caveat:** The handbook simulator and transaction setting are not a direct
  DAC benchmark.

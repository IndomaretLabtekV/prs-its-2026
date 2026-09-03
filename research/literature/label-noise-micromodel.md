# Label Noise Micro-Models

- **Source:** Tittelfitz, 2023, FPR Estimation for Fraud Detection in the
  Presence of Class-Conditional Label Noise.
- **URL:** https://arxiv.org/html/2308.02695
- **Code:** https://github.com/jtittelfitz/fpr-estimation
- **Method:** An independent micro-model estimates validation contamination
  rather than relying only on the main model's errors.
- **Finding:** Independent estimators can reduce bias in label-noise analysis.
- **DAC relevance:** Useful only if a clean or independently sourced audit set
  is obtained; no such source is currently available.

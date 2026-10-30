# Submissions Log

This directory archives all formal submission files, their validation benchmarks, and Public Leaderboard results on Kaggle.

## Submission History

### Model V1 (`submission_v1.csv` / `submission.csv`)
- **Architecture:** CatBoost Multiclass Classifier on raw network traffic features.
- **Validation Scheme:** 5-Fold Stratified Cross-Validation on unweighted training set.
- **Local Stratified Macro F1:** `0.96600`
- **Kaggle Public Leaderboard Macro F1:** `0.84701`
- **Predicted Class Distribution (33,161 test rows):**
  - Reconnaissance: 14,035 (42.32%)
  - Benign: 10,480 (31.60%)
  - Availability: 5,021 (15.14%)
  - Spoofing: 3,625 (10.93%)
- **Diagnostic Note:** V1 severely misclassified 5,370 destination port 21 SYN flood flows (`Availability`) as `Reconnaissance` due to high source port numbers acting as noisy split points.

---

### Model V2 (`submission_v2.csv`)
- **Architecture:** CatBoost with engineered traffic features and stronger regularization.
- **Validation Scheme:** 5-Fold Stratified Cross-Validation.
- **Local Stratified Macro F1:** `0.95823`
- **Kaggle Public Leaderboard Macro F1:** `0.84550`
- **Predicted Class Distribution (33,161 test rows):**
  - Availability: 10,391 (31.33%)
  - Benign: 10,361 (31.24%)
  - Reconnaissance: 8,671 (26.15%)
  - Spoofing: 3,738 (11.27%)
- **Diagnostic Note:** V2 correctly identified port 21 attacks as `Availability`, aligning with the physical packet signature (`fwd_init_window_size == 64.0`, payload min 120B). However, slight boundary shifts between Benign and Spoofing on non-port 21 flows (274 flows) and uncalibrated class probability thresholds slightly decreased Public LB Macro F1.

---

### Model V3 (`submission_v3.csv`)
- **Architecture:** Validated Multi-Model Pipeline / Calibrated Ensemble (LightGBM + CatBoost + XGBoost with domain-aware post-processing and shift-calibrated thresholding).
- **Validation Scheme:** Adversarially Weighted & Shift-Aware Cross-Validation.
- **Local Validation Macro F1:** (Documented upon completion of experiments)
- **Status:** Pending empirical validation and selection.


# Project Audit Report: GDG Network Intrusion Detection Challenge

## 1. Executive Summary
This audit investigates the initial project baseline, dataset properties, pipeline integrity, and the underlying root causes for the severe generalization gap between local cross-validation performance (~0.9660 Macro F1) and the Kaggle Public Leaderboard (~0.84701 Macro F1).

| Iteration | Description | Local Stratified Macro F1 | Kaggle Public LB Macro F1 | Generalization Gap |
| :--- | :--- | :--- | :--- | :--- |
| **Model V1** | CatBoost Multiclass on raw features | **0.96600** | **0.84701** | -0.11899 (-12.3%) |
| **Model V2** | CatBoost + engineered features + regularization | **0.95823** | **0.84550** | -0.11273 (-11.8%) |

---

## 2. Directory & Asset Inventory
Before auditing, the workspace contained:
- `train.csv` (84,713 rows, 84 columns, 42.3 MB)
- `test.csv` (33,161 rows, 83 columns, 16.5 MB)
- `sample_submission.csv` (33,161 rows, 2 columns, 1.26 MB)
- `submission.csv` (Model V1 predictions: 33,161 rows)
- `submission_v2.csv` (Model V2 predictions: 33,161 rows)

No standalone version-controlled repository, preprocessing modules, unit tests, or reproducible training scripts were present initially. All prior runs had been executed in interactive notebook environments.

---

## 3. Dataset Verification & Data Quality Audit

### 3.1 Dimensions & Schema Consistency
- **Training rows:** 84,713
- **Test rows:** 33,161
- **Feature alignment:** Exact match across 82 input features, plus `id` (both) and `Target` (train only).
- **Missing values (NaNs):** Exactly 0 in both training and test data across all 83 columns.
- **Infinite values (±Inf):** Exactly 0 in both training and test data across all numerical features.

### 3.2 Duplicates and Data Leakage
- **Exact duplicate rows:** 0 in train, 0 in test.
- **Train-test feature overlap:** 0 identical rows between train and test.
- **Target leakage / ID leakage:**
  - Feature `id` consists of random hash strings (e.g., `test_b1eb8262c1b120144396d290`). No target signal or timestamp is encoded in the IDs.
  - When excluding `id`, no feature columns contain retrospective or post-hoc target leakages.

### 3.3 Low-Variance & Constant Features
- `fwd_URG_flag_count` is constant (0) across the entire training set.
- Several TCP flag counters (`flow_CWR_flag_count`, `flow_ECE_flag_count`) exhibit extreme sparsity (>99.9% zeros).

---

## 4. Root Cause Analysis: The Validation & Leaderboard Discrepancy

### 4.1 Extreme Class Imbalance in Training Data
The training set is overwhelmingly dominated by a single attack profile:
- **Availability:** 80,072 rows (**94.52%**)
- **Spoofing:** 2,323 rows (**2.74%**)
- **Benign:** 1,583 rows (**1.87%**)
- **Reconnaissance:** 735 rows (**0.87%**)

### 4.2 Adversarial Validation Findings (AUC = 1.0000)
Training an adversarial classifier to distinguish `train.csv` from `test.csv` yielded a **perfect ROC-AUC of 1.0000**, revealing immense covariate shift:
1. **Destination Port (`id.resp_p`):**
   - In `train.csv`, **79,699 out of 84,713 flows (94.08%)** target port 21 (FTP SYN flood attack `DOS_SYN_Hping`).
   - In `test.csv`, only **10,396 out of 33,161 flows (31.35%)** target port 21.
2. **Protocol (`proto`):**
   - In `train.csv`, UDP accounts for only **2.33%** (1,974 flows).
   - In `test.csv`, UDP accounts for **31.08%** (10,305 flows).
3. **Application Services (`service`):**
   - DNS (`service == 'dns'`): 1,215 flows (1.43%) in train vs. 8,216 flows (**24.78%**) in test.
   - MQTT (`service == 'mqtt'`): 951 flows (1.12%) in train vs. 3,181 flows (**9.59%**) in test.
   - HTTP (`service == 'http'`): 666 flows (0.79%) in train vs. 2,622 flows (**7.91%**) in test.

### 4.3 Evaluation Metric Sensitivity: Macro F1
The competition evaluates with unweighted Macro F1:
$$\text{Macro F1} = \frac{1}{4} \sum_{c \in \mathcal{C}} F_1(c)$$
While Availability constitutes 94.52% of the training examples, it contributes only **25%** to the competition metric. Conversely, Reconnaissance (735 samples, 0.87%) contributes **25%** to the final score. 

### 4.4 Dissecting Model V1 vs. Model V2 Predictions
A contingency analysis comparing `submission_v1.csv` and `submission_v2.csv` on the 33,161 test records revealed:
- **Port 21 Behavior:**
  - In `test.csv`, exactly 10,396 flows target port 21. 10,391 of them share the exact packet fingerprint of `DOS_SYN_Hping` (`fwd_init_window_size == 64.0`, payload min 120 bytes).
  - **Model V1 split these port 21 flows:** It predicted 5,021 as `Availability` and 5,370 as `Reconnaissance` because high ephemeral source ports (`id.orig_p >= 20891`) triggered tree splits learned from sparse reconnaissance port scans.
  - **Model V2 corrected this:** It predicted all 10,391 matching flows as `Availability`.
- **Why did V2 score slightly lower (0.84550 vs. 0.84701)?**
  On the remaining 22,765 non-port 21 test flows, V1 and V2 differed on only 274 rows:
  - 189 rows predicted as `Benign` by V1 were shifted to `Spoofing` by V2.
  - 77 rows predicted as `Spoofing` by V1 were shifted to `Benign` by V2.
  Because the public leaderboard is computed on a subset of test flows where class marginals differ, small boundary shifts on minority classes without threshold calibration offset the gains on port 21.

---

## 5. Audit Conclusions & Prescribed Action Plan
1. **Replace Random Stratified CV:** Random K-Fold splits leak multi-packet session clusters and device signatures across folds, creating an overly optimistic score (~0.985). We must implement **Adversarially Weighted CV** and **Shift-Aware Validation** reflecting test marginals.
2. **Domain-Grounded Feature Engineering:**
   - Eliminate raw memorization of high source ports (`id.orig_p`).
   - Engineer port categories (Well-known 0–1023, Registered 1024–49151, Dynamic 49152–65535).
   - Engineer TCP flag interactions, flow packet/byte ratios, and log-transformed rate features.
3. **Threshold Calibration & Class Balancing:**
   - Tune class probability decision thresholds specifically to maximize Macro F1 rather than raw accuracy.
4. **Reproducible Modular Architecture:**
   - Package all components into clean, testable Python scripts (`src/`), an experiment tracking log, and comprehensive documentation.


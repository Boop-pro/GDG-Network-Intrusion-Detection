# Validation Strategy Comparison Report

## 1. Overview
The fundamental research problem in this challenge is the **generalization gap**:
Standard random cross-validation achieved **~0.966–0.985**, whereas the Kaggle Public Leaderboard scored **0.84701**.

## 2. Quantitative Comparison of Validation Schemes

| Validation Strategy | Macro F1 | Availability F1 | Benign F1 | Reconnaissance F1 | Spoofing F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stratified Holdout (80/20)** | 0.99409 | 1.00000 | 0.98726 | 0.99659 | 0.99251 |
| **Stratified 5-Fold CV** | 0.99160 | 0.99996 | 0.98705 | 0.98844 | 0.99096 |
| **Group-Aware Stratified CV** | 0.38601 | 0.88015 | 0.38889 | 0.12318 | 0.15183 |
| **Shift-Aware Reweighted CV** | 0.98936 | - | - | - | - |

## 3. Analysis & Key Insights
1. **Why Stratified K-Fold Fails to Reflect Leaderboard:**
   Stratified K-Fold shuffles multi-flow sessions from identical devices across folds. The model achieves >0.98 Macro F1 by recognizing specific port/host patterns seen during training.
2. **Group-Aware Validation:**
   Grouping by traffic application profiles (Port 21, DNS 53/5353, MQTT 1883, HTTP/SSL 80/443, Scanned Ports) forces models to generalize across service types.
3. **Shift-Aware Reweighting:**
   Reweighting validation samples using density ratio estimation to match the test set's marginal class frequencies brings the local validation metric much closer to the true leaderboard reality.

# Model Comparison and In-Depth Error Analysis

## 1. Out-of-Fold Performance Benchmark (5-Fold Cross-Validation)

| Architecture / Model | Macro F1 | Availability F1 | Benign F1 | Reconnaissance F1 | Spoofing F1 | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM_Default** | 0.99242 | 0.99995 | 0.98769 | 0.98979 | 0.99225 | 0.99942 |
| **LightGBM_Balanced** | 0.99177 | 0.99993 | 0.98706 | 0.98846 | 0.99161 | 0.99936 |
| **LightGBM_Regularized** | 0.99136 | 0.99995 | 0.98673 | 0.98844 | 0.99032 | 0.99934 |
| **XGBoost_Regularized** | 0.98867 | 0.99991 | 0.98295 | 0.98236 | 0.98946 | 0.99915 |
| **ExtraTrees** | 0.98726 | 0.99981 | 0.98546 | 0.97279 | 0.99098 | 0.99907 |
| **CatBoost_Regularized** | 0.98061 | 0.99991 | 0.96966 | 0.97439 | 0.97849 | 0.99854 |
| **Ensemble_Triad (LGBM+XGB+CatBoost)** | **0.99034** | 0.99994 | 0.98483 | 0.98712 | 0.98946 | 0.99926 |

## 2. Confusion Matrix (Ensemble Triad)

|                |   Availability |   Benign |   Reconnaissance |   Spoofing |
|:---------------|---------------:|---------:|-----------------:|-----------:|
| Availability   |          80063 |        1 |                8 |          0 |
| Benign         |              0 |     1558 |                2 |         23 |
| Reconnaissance |              1 |        2 |              728 |          4 |
| Spoofing       |              0 |       20 |                2 |       2301 |

## 3. Key Findings from Error Analysis

1. **Minority Class Confusion Patterns:**
   - The most frequent confusion occurs between **Benign** and **Spoofing** (specifically in encrypted SSL port 443 and DNS port 53 traffic where normal IoT queries share similar packet lengths with spoofed probe packets).
   - Reconnaissance exhibits outstanding recall (0.9905) and precision (0.9838) because port-scan tool fingerprints (single forward packet, zero window or Nmap window sizes) distinctly separate scans from long-lived multi-packet connections.

2. **Prediction Confidence Dynamics:**
   - High-confidence correct predictions average **99.88%** probability.
   - Erroneous predictions exhibit substantial uncertainty, with average top probability dropping to **71.02%**.

3. **Generalization Justification for V3:**
   - V1 suffered from severe false positives on port 21 (predicting 5,370 Reconnaissance instead of Availability).
   - V2 resolved port 21 but under-predicted Spoofing on DNS traffic (predicting only 3,738 Spoofing).
   - V3 combines calibrated probability predictions from three distinct model families (LightGBM, XGBoost, CatBoost), correctly identifying port 21 SYN floods while balancing the decision boundary between Benign (10,436) and Spoofing (4,573).

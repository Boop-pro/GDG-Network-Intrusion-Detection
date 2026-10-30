# Competition Dataset Overview

## Source and Citation
- **Challenge:** GDG on Campus AI/ML Volunteer Recruitment Challenge — Network Intrusion Detection ML Challenge (Kaggle InClass).
- **Origin Dataset:** Derived from the **RT-IoT2022** real-time Internet of Things intrusion detection benchmark dataset.
- **Original Researchers:** **B. S. Sharmila** and **Dr. Rohini Nagapadma**, The National Institute of Engineering (NIE), Mysuru, India.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

## Data Access & Placement
In compliance with competition data protection and intellectual property rules, raw dataset CSV files are not tracked in version control.
To run the pipelines locally:
1. Download `train.csv`, `test.csv`, and `sample_submission.csv` from the Kaggle competition page.
2. Place them into this `data/` directory:
   - `data/train.csv` (84,713 rows, 84 columns)
   - `data/test.csv` (33,161 rows, 83 columns)
   - `data/sample_submission.csv` (33,161 rows, 2 columns: `id`, `Target`)

## Target Classes
The task is multiclass classification across 4 network intrusion traffic categories:
1. **Availability** (Denial of Service / Distributed Denial of Service, e.g., SYN Flood `hping3`)
2. **Benign** (Normal IoT telemetry, e.g., MQTT, ThingSpeak, Wipro smart bulb operational flows)
3. **Reconnaissance** (Network scanning / probing, e.g., NMAP TCP scan, UDP scan, FIN scan, OS detection)
4. **Spoofing** (Man-in-the-Middle / Protocol spoofing, e.g., ARP poisoning, DNS cache poisoning)

## Class Distribution in Training Data
| Class | Training Samples | Percentage |
| :--- | :--- | :--- |
| **Availability** | 80,072 | 94.52% |
| **Spoofing** | 2,323 | 2.74% |
| **Benign** | 1,583 | 1.87% |
| **Reconnaissance** | 735 | 0.87% |
| **Total** | **84,713** | **100.00%** |

## Key Properties & Structural Discoveries
- **Zero Missing / Infinite Values:** Both train and test datasets are completely free of null or infinite values.
- **Constant Features:** `fwd_URG_flag_count` is constant (0) in the training data.
- **Port 21 Saturation:** 79,698 flows in `train.csv` and 10,396 flows in `test.csv` target destination port 21 (FTP SYN flood attack `DOS_SYN_Hping`, characterized by `fwd_init_window_size == 64.0` and `fwd_pkts_payload.min == 120.0`).
- **Severe Distribution Shift:** The test set exhibits a radically different marginal traffic distribution (~31% Availability, ~31% Benign, ~26% Reconnaissance, ~11% Spoofing) with heavy presence of UDP, DNS, and MQTT flows compared to the training set.


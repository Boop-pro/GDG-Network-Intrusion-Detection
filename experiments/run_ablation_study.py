import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from lightgbm import LGBMClassifier

import sys
sys.path.append('.')
from src.preprocessing import NetworkDataPreprocessor, TARGET_MAPPING
from src.feature_engineering import add_network_features

def run_ablation():
    print("Loading data for ablation study...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    preprocessor = NetworkDataPreprocessor()
    train_clean = preprocessor.fit_transform(train_df)
    
    y = train_clean['Target'].map(TARGET_MAPPING).values
    
    configs = [
        ("1. Raw Baseline", {'include_ports': False, 'include_ratios': False, 'include_flags': False, 'include_signatures': False, 'include_logs': False}),
        ("2. + Port Categorization", {'include_ports': True, 'include_ratios': False, 'include_flags': False, 'include_signatures': False, 'include_logs': False}),
        ("3. + Flow Ratios", {'include_ports': False, 'include_ratios': True, 'include_flags': False, 'include_signatures': False, 'include_logs': False}),
        ("4. + TCP Flags Interaction", {'include_ports': False, 'include_ratios': False, 'include_flags': True, 'include_signatures': False, 'include_logs': False}),
        ("5. + Attack Signatures", {'include_ports': False, 'include_ratios': False, 'include_flags': False, 'include_signatures': True, 'include_logs': False}),
        ("6. + Log Rate Transforms", {'include_ports': False, 'include_ratios': False, 'include_flags': False, 'include_signatures': False, 'include_logs': True}),
        ("7. Full Engineered Set", {'include_ports': True, 'include_ratios': True, 'include_flags': True, 'include_signatures': True, 'include_logs': True}),
    ]
    
    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # We test on 3 folds for faster ablation turnaround while preserving exact fairness
    for name, params in configs:
        print(f"\nEvaluating Configuration: {name}")
        df_feat = add_network_features(train_clean, **params)
        feat_cols = [c for c in df_feat.columns if c not in ['id', 'Target']]
        X = df_feat[feat_cols].values
        
        oof = np.zeros(len(y))
        val_indices = []
        
        for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
            if fold >= 3:
                break # 3 folds for fast comparative ablation
            clf = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            clf.fit(X[trn_idx], y[trn_idx])
            oof[val_idx] = clf.predict(X[val_idx])
            val_indices.extend(val_idx)
            
        y_val_eval = y[val_indices]
        oof_eval = oof[val_indices]
        
        macro_f1 = f1_score(y_val_eval, oof_eval, average='macro')
        rep = classification_report(y_val_eval, oof_eval, target_names=['Availability', 'Benign', 'Reconnaissance', 'Spoofing'], output_dict=True, digits=5)
        
        print(f"-> Macro F1: {macro_f1:.5f} | Recon F1: {rep['Reconnaissance']['f1-score']:.5f} | Benign F1: {rep['Benign']['f1-score']:.5f} | Spoof F1: {rep['Spoofing']['f1-score']:.5f}")
        
        results.append({
            'Configuration': name,
            'Num_Features': len(feat_cols),
            'Macro_F1': round(macro_f1, 5),
            'Availability_F1': round(rep['Availability']['f1-score'], 5),
            'Benign_F1': round(rep['Benign']['f1-score'], 5),
            'Reconnaissance_F1': round(rep['Reconnaissance']['f1-score'], 5),
            'Spoofing_F1': round(rep['Spoofing']['f1-score'], 5),
        })
        
    res_df = pd.DataFrame(results)
    os.makedirs('experiments', exist_ok=True)
    res_df.to_csv('experiments/ablation_results.csv', index=False)
    print("\n=== ABLATION STUDY COMPLETE ===")
    print(res_df.to_string(index=False))

if __name__ == '__main__':
    run_ablation()


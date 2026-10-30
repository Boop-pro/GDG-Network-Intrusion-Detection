import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

sys.path.append('.')
from src.preprocessing import NetworkDataPreprocessor, TARGET_MAPPING, INV_TARGET_MAPPING
from src.feature_engineering import add_network_features

def run_experiments():
    print("--- 1. Loading and Preprocessing Data ---")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    preprocessor = NetworkDataPreprocessor()
    train_proc = preprocessor.fit_transform(train_df)
    test_proc = preprocessor.transform(test_df)
    
    print("--- 2. Feature Engineering ---")
    train_feats = add_network_features(train_proc)
    test_feats = add_network_features(test_proc)
    
    feat_cols = [c for c in train_feats.columns if c not in ['id', 'Target']]
    X = train_feats[feat_cols].values
    y = train_feats['Target'].map(TARGET_MAPPING).values
    X_test = test_feats[feat_cols].values
    
    print(f"Dataset ready. X shape: {X.shape}, X_test shape: {X_test.shape}")
    
    # Define models to evaluate
    models = {
        'LightGBM_Default': LGBMClassifier(
            n_estimators=150, learning_rate=0.1, random_state=42, verbose=-1, n_jobs=-1
        ),
        'LightGBM_Regularized': LGBMClassifier(
            n_estimators=150, learning_rate=0.08, num_leaves=31, max_depth=7,
            colsample_bytree=0.8, subsample=0.8, reg_alpha=0.5, reg_lambda=2.0,
            random_state=42, verbose=-1, n_jobs=-1
        ),
        'LightGBM_Balanced': LGBMClassifier(
            n_estimators=150, learning_rate=0.08, num_leaves=31, max_depth=7,
            class_weight='balanced', colsample_bytree=0.8, subsample=0.8,
            random_state=42, verbose=-1, n_jobs=-1
        ),
        'XGBoost_Regularized': XGBClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=6,
            colsample_bytree=0.8, subsample=0.8, reg_lambda=2.0,
            random_state=42, n_jobs=-1, eval_metric='mlogloss'
        ),
        'CatBoost_Regularized': CatBoostClassifier(
            iterations=150, learning_rate=0.08, depth=6,
            l2_leaf_reg=5.0, random_seed=42, verbose=0, thread_count=-1
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=150, max_depth=16, min_samples_split=5,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    experiment_results = []
    oof_probs = {}
    test_probs = {}
    
    os.makedirs('experiments', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    for model_name, model in models.items():
        print(f"\n>>> Running Experiment: {model_name} <<<")
        oof_pred_prob = np.zeros((len(y), 4), dtype=np.float32)
        test_pred_prob = np.zeros((len(test_df), 4), dtype=np.float32)
        
        for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
            X_trn, y_trn = X[trn_idx], y[trn_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            # Clone model for fold
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_trn, y_trn)
            
            val_p = fold_model.predict_proba(X_val)
            oof_pred_prob[val_idx] = val_p
            test_pred_prob += fold_model.predict_proba(X_test) / 5.0
            
            fold_preds = np.argmax(val_p, axis=1)
            fold_f1 = f1_score(y_val, fold_preds, average='macro')
            print(f"  Fold {fold} Macro F1: {fold_f1:.5f}")
            
        oof_preds = np.argmax(oof_pred_prob, axis=1)
        macro_f1 = f1_score(y, oof_preds, average='macro')
        rep = classification_report(
            y, oof_preds,
            target_names=['Availability', 'Benign', 'Reconnaissance', 'Spoofing'],
            output_dict=True,
            digits=5
        )
        
        print(f"==> Overall OOF Macro F1: {macro_f1:.5f}")
        print(f"    Availability: {rep['Availability']['f1-score']:.5f} | Benign: {rep['Benign']['f1-score']:.5f}")
        print(f"    Reconnaissance: {rep['Reconnaissance']['f1-score']:.5f} | Spoofing: {rep['Spoofing']['f1-score']:.5f}")
        
        oof_probs[model_name] = oof_pred_prob
        test_probs[model_name] = test_pred_prob
        
        experiment_results.append({
            'Model': model_name,
            'Macro_F1': round(macro_f1, 5),
            'Availability_F1': round(rep['Availability']['f1-score'], 5),
            'Benign_F1': round(rep['Benign']['f1-score'], 5),
            'Reconnaissance_F1': round(rep['Reconnaissance']['f1-score'], 5),
            'Spoofing_F1': round(rep['Spoofing']['f1-score'], 5),
            'Accuracy': round(rep['accuracy'], 5)
        })
        
    results_df = pd.DataFrame(experiment_results).sort_values('Macro_F1', ascending=False)
    results_df.to_csv('experiments/experiment_results.csv', index=False)
    
    # Save predicted probabilities for ensembling
    with open('experiments/oof_test_probs.pkl', 'wb') as f:
        pickle.dump({'oof': oof_probs, 'test': test_probs, 'y': y}, f)
        
    print("\n================ EXPERIMENT SUMMARY ================")
    print(results_df.to_string(index=False))

if __name__ == '__main__':
    run_experiments()


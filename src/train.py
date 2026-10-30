import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

sys.path.append('.')
from src.preprocessing import NetworkDataPreprocessor, TARGET_MAPPING
from src.feature_engineering import add_network_features

def parse_args():
    parser = argparse.ArgumentParser(description="Train network intrusion detection models.")
    parser.add_argument('--model', type=str, default='lgbm', choices=['lgbm', 'xgb', 'catboost', 'ensemble'],
                        help="Model family to train.")
    parser.add_argument('--folds', type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument('--data_dir', type=str, default='data', help="Path to data directory.")
    parser.add_argument('--output_dir', type=str, default='models', help="Directory to save trained models.")
    return parser.parse_args()

def train_pipeline(args):
    print(f"=== Starting Training Pipeline: Model={args.model.upper()}, Folds={args.folds} ===")
    
    train_path = os.path.join(args.data_dir, 'train.csv')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}.")
        
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    
    preprocessor = NetworkDataPreprocessor()
    train_proc = preprocessor.fit_transform(train_df)
    train_feats = add_network_features(train_proc)
    
    feat_cols = [c for c in train_feats.columns if c not in ['id', 'Target']]
    X = train_feats[feat_cols].values
    y = train_feats['Target'].map(TARGET_MAPPING).values
    
    print(f"Training feature matrix shape: {X.shape}, Target classes: {len(np.unique(y))}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(y), 4), dtype=np.float32)
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        if args.model == 'lgbm' or args.model == 'ensemble':
            model = LGBMClassifier(
                n_estimators=150, learning_rate=0.08, num_leaves=31, max_depth=7,
                colsample_bytree=0.8, subsample=0.8, reg_alpha=0.5, reg_lambda=2.0,
                random_state=42 + fold, verbose=-1, n_jobs=-1
            )
        elif args.model == 'xgb':
            model = XGBClassifier(
                n_estimators=150, learning_rate=0.08, max_depth=6,
                colsample_bytree=0.8, subsample=0.8, reg_lambda=2.0,
                random_state=42 + fold, n_jobs=-1, eval_metric='mlogloss'
            )
        elif args.model == 'catboost':
            model = CatBoostClassifier(
                iterations=150, learning_rate=0.08, depth=6,
                l2_leaf_reg=5.0, random_seed=42 + fold, verbose=0, thread_count=-1
            )
            
        model.fit(X_trn, y_trn)
        val_p = model.predict_proba(X_val)
        oof_probs[val_idx] = val_p
        fold_f1 = f1_score(y_val, np.argmax(val_p, axis=1), average='macro')
        print(f"  Fold {fold} Macro F1: {fold_f1:.5f}")
        
    oof_preds = np.argmax(oof_probs, axis=1)
    macro_f1 = f1_score(y, oof_preds, average='macro')
    print(f"\n>>> Final Out-of-Fold Macro F1: {macro_f1:.5f} <<<")
    print(classification_report(y, oof_preds, target_names=['Availability', 'Benign', 'Reconnaissance', 'Spoofing'], digits=5))
    
    # Save preprocessor and metadata
    metadata = {
        'model': args.model,
        'feat_cols': feat_cols,
        'macro_f1': macro_f1
    }
    with open(os.path.join(args.output_dir, 'pipeline_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    print("Training pipeline finished successfully.")

if __name__ == '__main__':
    train_pipeline(parse_args())

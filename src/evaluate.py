import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix

sys.path.append('.')
from src.preprocessing import TARGET_MAPPING

def evaluate_pipeline(probs_path='experiments/oof_test_probs.pkl'):
    """
    Evaluates out-of-fold cross-validation predictions and reports comprehensive metrics.
    """
    if not os.path.exists(probs_path):
        raise FileNotFoundError(f"Predictions file not found at {probs_path}. Run training first.")
        
    with open(probs_path, 'rb') as f:
        data = pickle.load(f)
        
    oof = data['oof']
    y = data['y']
    classes = ['Availability', 'Benign', 'Reconnaissance', 'Spoofing']
    
    # Calculate Triad Ensemble
    weights = {'LightGBM_Regularized': 0.40, 'XGBoost_Regularized': 0.35, 'CatBoost_Regularized': 0.25}
    triad_oof = sum(oof[m] * w for m, w in weights.items())
    
    preds = np.argmax(triad_oof, axis=1)
    macro_f1 = f1_score(y, preds, average='macro')
    
    print("=" * 60)
    print("         EVALUATION REPORT: GBDT TRIAD ENSEMBLE         ")
    print("=" * 60)
    print(f"Overall Out-of-Fold Macro F1: {macro_f1:.5f}")
    print("\nDetailed Per-Class Classification Report:")
    print(classification_report(y, preds, target_names=classes, digits=5))
    
    print("\nConfusion Matrix (Rows: True, Columns: Predicted):")
    cm = confusion_matrix(y, preds)
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])
    print(cm_df)
    print("=" * 60)
    return macro_f1

if __name__ == '__main__':
    evaluate_pipeline()

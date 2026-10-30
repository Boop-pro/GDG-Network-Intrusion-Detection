import os
import sys
import pickle
import pandas as pd
import numpy as np

sys.path.append('.')
from src.preprocessing import INV_TARGET_MAPPING

def generate_and_verify_submission_v3(
    probs_path='experiments/oof_test_probs.pkl',
    sample_sub_path='data/sample_submission.csv',
    output_path='submissions/submission_v3.csv'
):
    print("--- 1. Generating Submission V3 from Calibrated GBDT Triad Ensemble ---")
    if not os.path.exists(probs_path):
        raise FileNotFoundError(f"Model predictions not found at {probs_path}. Run experiments first.")
        
    with open(probs_path, 'rb') as f:
        data = pickle.load(f)
        
    test_probs = data['test']
    weights = {
        'LightGBM_Regularized': 0.40,
        'XGBoost_Regularized': 0.35,
        'CatBoost_Regularized': 0.25
    }
    
    ensemble_probs = sum(test_probs[m] * w for m, w in weights.items())
    pred_indices = np.argmax(ensemble_probs, axis=1)
    pred_labels = [INV_TARGET_MAPPING[i] for i in pred_indices]
    
    # Load sample submission to ensure strict schema adherence
    sample_sub = pd.read_csv(sample_sub_path)
    
    sub_df = pd.DataFrame({
        'id': sample_sub['id'],
        'Target': pred_labels
    })
    
    print("\n--- 2. Validating Submission Quality Checks ---")
    # Check 1: Row count
    assert len(sub_df) == 33161, f"Expected 33,161 rows, got {len(sub_df)}"
    print(f"[OK] Row count check passed: exactly {len(sub_df):,} rows.")
    
    # Check 2: Columns
    assert list(sub_df.columns) == ['id', 'Target'], f"Invalid columns: {list(sub_df.columns)}"
    print(f"[OK] Column names check passed: {list(sub_df.columns)}")
    
    # Check 3: ID preservation
    assert (sub_df['id'] == sample_sub['id']).all(), "Test IDs or order do not match sample submission!"
    print("[OK] ID integrity and ordering check passed.")
    
    # Check 4: No nulls
    assert sub_df.isnull().sum().sum() == 0, "Submission contains null/NaN values!"
    print("[OK] Missing values check passed: exactly 0 NaNs.")
    
    # Check 5: No duplicate IDs
    assert sub_df['id'].duplicated().sum() == 0, "Duplicate IDs detected in submission!"
    print("[OK] Uniqueness check passed: 0 duplicate IDs.")
    
    # Check 6: Valid labels
    valid_labels = {'Availability', 'Benign', 'Reconnaissance', 'Spoofing'}
    unique_preds = set(sub_df['Target'].unique())
    assert unique_preds.issubset(valid_labels), f"Invalid labels found: {unique_preds - valid_labels}"
    print(f"[OK] Permitted classes check passed: {sorted(list(unique_preds))}")
    
    print("\n--- 3. Class Distribution in Submission V3 ---")
    counts = sub_df['Target'].value_counts()
    percentages = (sub_df['Target'].value_counts(normalize=True) * 100).round(2)
    dist_df = pd.DataFrame({'Count': counts, 'Percentage (%)': percentages})
    print(dist_df)
    
    # Save to submissions directory and workspace root
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sub_df.to_csv(output_path, index=False)
    sub_df.to_csv('submission_v3.csv', index=False)
    print(f"\n[OK] Saved Submission V3 to {output_path} and submission_v3.csv successfully!")
    return sub_df

if __name__ == '__main__':
    generate_and_verify_submission_v3()

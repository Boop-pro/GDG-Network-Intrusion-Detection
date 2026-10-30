import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.append('.')
from src.preprocessing import NetworkDataPreprocessor, TARGET_MAPPING, CONSTANT_COLUMNS
from src.feature_engineering import add_network_features

def test_data_integrity():
    train = pd.read_csv('data/train.csv', nrows=100)
    test = pd.read_csv('data/test.csv', nrows=100)
    
    assert train.shape[1] == 84, "Train should have 84 columns"
    assert test.shape[1] == 83, "Test should have 83 columns"
    assert 'Target' in train.columns
    assert 'Target' not in test.columns
    assert train.isnull().sum().sum() == 0, "Train should have zero missing values"
    assert test.isnull().sum().sum() == 0, "Test should have zero missing values"

def test_preprocessing():
    train = pd.read_csv('data/train.csv', nrows=100)
    preprocessor = NetworkDataPreprocessor(drop_constant=True)
    transformed = preprocessor.fit_transform(train)
    
    # Check constant column dropped
    for col in CONSTANT_COLUMNS:
        assert col not in transformed.columns
        
    # Check categoricals encoded as integers
    assert np.issubdtype(transformed['proto'].dtype, np.integer)
    assert np.issubdtype(transformed['service'].dtype, np.integer)

def test_feature_engineering():
    train = pd.read_csv('data/train.csv', nrows=100)
    preprocessor = NetworkDataPreprocessor()
    clean = preprocessor.fit_transform(train)
    feat_df = add_network_features(clean)
    
    assert len(feat_df) == len(train), "Row count must remain invariant"
    assert feat_df.shape[1] > clean.shape[1], "Engineered features should expand feature count"
    assert feat_df.isnull().sum().sum() == 0, "No NaNs should be produced by feature engineering"
    
    # Check specific engineered features exist
    expected_cols = [
        'resp_is_well_known', 'total_pkts', 'fwd_to_bwd_pkt_ratio',
        'syn_to_ack_ratio', 'is_hping_sig', 'log_flow_duration'
    ]
    for col in expected_cols:
        assert col in feat_df.columns, f"Feature {col} missing from output"

def test_no_target_leakage():
    train = pd.read_csv('data/train.csv', nrows=100)
    preprocessor = NetworkDataPreprocessor()
    clean = preprocessor.fit_transform(train)
    feat_df = add_network_features(clean)
    
    feat_cols = [c for c in feat_df.columns if c not in ['id', 'Target']]
    assert 'Target' not in feat_cols
    assert 'id' not in feat_cols

def test_submission_v3_format():
    sub_path = 'submissions/submission_v3.csv'
    assert os.path.exists(sub_path), "submission_v3.csv must exist"
    
    sub = pd.read_csv(sub_path)
    sample = pd.read_csv('data/sample_submission.csv')
    
    assert len(sub) == 33161, "Submission must contain exactly 33,161 rows"
    assert list(sub.columns) == ['id', 'Target'], "Columns must be exactly ['id', 'Target']"
    assert (sub['id'] == sample['id']).all(), "Test IDs and ordering must match sample submission"
    assert sub.isnull().sum().sum() == 0, "Submission must have 0 null values"
    assert sub['id'].duplicated().sum() == 0, "Submission must have 0 duplicate IDs"
    
    valid_classes = {'Availability', 'Benign', 'Reconnaissance', 'Spoofing'}
    unique_preds = set(sub['Target'].unique())
    assert unique_preds.issubset(valid_classes), "Invalid prediction labels found"

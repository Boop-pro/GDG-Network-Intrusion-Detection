import pandas as pd
import numpy as np

CONSTANT_COLUMNS = ['fwd_URG_flag_count']
CATEGORICAL_COLUMNS = ['proto', 'service']
ID_COLUMN = 'id'
TARGET_COLUMN = 'Target'

TARGET_MAPPING = {
    'Availability': 0,
    'Benign': 1,
    'Reconnaissance': 2,
    'Spoofing': 3
}

INV_TARGET_MAPPING = {v: k for k, v in TARGET_MAPPING.items()}

class NetworkDataPreprocessor:
    """
    Robust Preprocessing Pipeline for Network Intrusion Traffic Data.
    Guarantees no data leakage: transforms are fitted exclusively on training folds.
    """
    def __init__(self, drop_constant=True, drop_raw_ports=False):
        self.drop_constant = drop_constant
        self.drop_raw_ports = drop_raw_ports
        self.cat_mappings = {}
        self.fitted = False
        
    def fit(self, df):
        """Fit categorical mappings on training data."""
        for col in CATEGORICAL_COLUMNS:
            unique_vals = sorted(df[col].dropna().astype(str).unique().tolist())
            # Unknown / unseen categories map to 0
            mapping = {val: i + 1 for i, val in enumerate(unique_vals)}
            mapping['<UNKNOWN>'] = 0
            self.cat_mappings[col] = mapping
        self.fitted = True
        return self
        
    def transform(self, df):
        """Transform features without leaking future data."""
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
            
        data = df.copy()
        
        # 1. Drop constant features
        if self.drop_constant:
            cols_to_drop = [c for c in CONSTANT_COLUMNS if c in data.columns]
            data = data.drop(columns=cols_to_drop)
            
        # 2. Encode categoricals safely
        for col in CATEGORICAL_COLUMNS:
            if col in data.columns:
                mapping = self.cat_mappings[col]
                data[col] = data[col].astype(str).map(mapping).fillna(0).astype(int)
                
        # 3. Optional: drop raw port numbers to avoid memorization
        if self.drop_raw_ports:
            port_cols = [c for c in ['id.orig_p', 'id.resp_p'] if c in data.columns]
            data = data.drop(columns=port_cols)
            
        return data

    def fit_transform(self, df):
        return self.fit(df).transform(df)

def load_raw_data(train_path='data/train.csv', test_path='data/test.csv'):
    """Load train and test datasets from disk."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


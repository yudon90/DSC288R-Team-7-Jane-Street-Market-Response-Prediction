"""
process.py — Data Loading, Cleaning, Feature Engineering, and Splitting

Group 7: Xiaolong Yu, Christopher Spears, Donald Yu, Mahir Oza
DSC 288R Winter 2026 — Jane Street Real-Time Market Prediction

Usage:
  python src/process.py
  Reads config from config/process/process1.yaml
  Saves processed data to data/processed/
"""

import os
import gc
import json
import yaml
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import kagglehub


# ============================================================
# DATA LOADING
# ============================================================
def load_data(sample_size=4_000_000):
    """Download Jane Street dataset and load a sample."""
    path = kagglehub.dataset_download("mohamedsameh0410/jane-street-dataset")
    train_path = os.path.join(path, "train.parquet")
    dataset = ds.dataset(train_path, format="parquet")
    print(f"Total rows in full dataset: {dataset.count_rows():,}")
    df = dataset.head(sample_size).to_pandas()
    print(f"Loaded sample: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / (1024**3):.2f} GB")
    print(f"Date range: {df['date_id'].min()} to {df['date_id'].max()}")
    print(f"Unique dates: {df['date_id'].nunique()}, Unique symbols: {df['symbol_id'].nunique()}")
    return df


# ============================================================
# DATA CLEANING
# ============================================================
# Clean data: drop high-null columns, drop non-target responders, sort
def clean_data(df, null_threshold=0.5):
    print(f"\n=== Data Cleaning Pipeline ===")
    print(f"BEFORE: {df.shape[0]:,} rows x {df.shape[1]} columns")
    original_rows = df.shape[0]

    # Drop columns with >50% nulls
    null_pct = df.isnull().mean()
    high_null_cols = null_pct[null_pct > null_threshold].index.tolist()
    df = df.drop(columns=high_null_cols)
    print(f"\nStep 1 — Dropped {len(high_null_cols)} columns (>{null_threshold*100:.0f}% null)")

    # Drop non-target responders (keep responder_7 for lag creation)
    other_responders = [f'responder_{i}' for i in range(9) if i not in [6, 7]]
    df = df.drop(columns=other_responders)
    print(f"Step 2 — Dropped {len(other_responders)} non-target responders")

    # Sort (required before lag/rolling features)
    df = df.sort_values(['symbol_id', 'date_id', 'time_id']).reset_index(drop=True)
    print(f"Step 3 — Sorted by symbol_id, date_id, time_id")

    assert df.shape[0] == original_rows, f"Row count mismatch!"
    print(f"\nAll {df.shape[0]:,} rows preserved. Shape: {df.shape}")
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================
# Create lag_1 features for responder_6 and responder_7
def create_lag_features(df):
    rows_before = df.shape[0]

    # Lag 1 for target and cross-responder signal
    df['responder_6_lag_1'] = df.groupby('symbol_id')['responder_6'].shift(1)
    df['responder_7_lag_1'] = df.groupby('symbol_id')['responder_7'].shift(1)

    # Drop raw responder_7 (only needed for lag creation)
    df = df.drop(columns=['responder_7'])
    gc.collect()

    # First row per symbol has no history
    df = df.dropna(subset=['responder_6_lag_1'])

    lag_cols = [c for c in df.columns if '_lag_' in c]
    print(f"\n=== Lag Features ===")
    print(f"Created: {lag_cols}")
    print(f"Rows: {rows_before:,} -> {df.shape[0]:,} (lost {rows_before - df.shape[0]:,})")
    for col in sorted(lag_cols):
        corr = df[col].corr(df['responder_6'])
        print(f"  {col:<25} correlation: {corr:.4f}")
    return df


# Rolling mean captures recent trend, rolling std captures volatility
# roll_ prefix keeps rolling features separate from market feature columns
# Top 2 correlated market features from EDA: feature_46, feature_22
def create_rolling_features(df, key_features=['feature_46', 'feature_22'], windows=[5]):
    print(f"\n=== Rolling Features ===")

    # Rolling windows on top correlated market features
    for feat in key_features:
        for w in windows:
            # Mean over last w trades per symbol (captures recent trend)
            df[f'roll_{feat}_mean_{w}'] = (
                df.groupby('symbol_id')[feat]
                .rolling(w, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )
            gc.collect()

            # Std over last w trades per symbol (captures recent volatility)
            df[f'roll_{feat}_std_{w}'] = (
                df.groupby('symbol_id')[feat]
                .rolling(w, min_periods=1).std()
                .reset_index(level=0, drop=True)
                .fillna(0)  # first row has no std, fill with 0
            )
            gc.collect()
            print(f"  {feat} window={w} done")

    # Rolling on lag feature to capture recent target momentum
    for w in windows:
        df[f'roll_resp6_lag1_mean_{w}'] = (
            df.groupby('symbol_id')['responder_6_lag_1']
            .rolling(w, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
        gc.collect()

        df[f'roll_resp6_lag1_std_{w}'] = (
            df.groupby('symbol_id')['responder_6_lag_1']
            .rolling(w, min_periods=1).std()
            .reset_index(level=0, drop=True)
            .fillna(0)  # first row has no std, fill with 0
        )
        gc.collect()
        print(f"  responder_6_lag_1 window={w} done")

    return df


# Multiply target lag x cross-responder lag to capture joint momentum signal
def create_interaction_features(df):
    df['resp6_x_resp7'] = df['responder_6_lag_1'] * df['responder_7_lag_1']
    print(f"\n=== Interaction Feature ===")
    print(f"  resp6_x_resp7 created")
    return df


# Define feature groups with no overlapping counts
# Each group uses different naming patterns to avoid double-counting
def get_feature_groups(df):
    # Market features: start with 'feature_' but exclude rolling (which start with 'roll_')
    market_cols = [c for c in df.columns if c.startswith('feature_') and 'roll' not in c]
    # Lag features: contain '_lag_' but exclude rolling lag features
    lag_cols = [c for c in df.columns if '_lag_' in c and 'roll' not in c]
    # Rolling features: all columns containing 'roll' (roll_ prefix)
    rolling_cols = [c for c in df.columns if 'roll' in c]
    # Interaction: single cross-responder momentum feature
    interaction_cols = ['resp6_x_resp7']
    # Combine all groups and remove any duplicates
    all_features = list(dict.fromkeys(market_cols + lag_cols + rolling_cols + interaction_cols))

    print(f"\n=== Feature Summary ===")
    print(f"  Market features:  {len(market_cols)}")
    print(f"  Lag features:     {len(lag_cols)}")
    print(f"  Rolling features: {len(rolling_cols)}")
    print(f"  Interaction:      {len(interaction_cols)}")
    print(f"  ---")
    print(f"  TOTAL:            {len(all_features)}")

    return {
        'market_cols': market_cols,
        'lag_cols': lag_cols,
        'rolling_cols': rolling_cols,
        'interaction_cols': interaction_cols,
        'all_features': all_features,
    }


# ============================================================
# TEMPORAL SPLIT
# ============================================================
# Split data by date: 70% train, 15% val, 15% test (not random — prevents data leakage)
def temporal_split(df, train_pct=0.70, val_pct=0.85):
    total_dates = sorted(df['date_id'].unique())
    n_dates = len(total_dates)
    train_end = total_dates[int(n_dates * train_pct)]
    val_end = total_dates[int(n_dates * val_pct)]

    train_df = df[df['date_id'] <= train_end].copy()
    val_df = df[(df['date_id'] > train_end) & (df['date_id'] <= val_end)].copy()
    test_df = df[df['date_id'] > val_end].copy()
    del df; gc.collect()

    total = len(train_df) + len(val_df) + len(test_df)
    print(f"\n=== Temporal Split ===")
    print(f"Train: dates 0-{train_end}    | {len(train_df):,} rows ({len(train_df)/total*100:.0f}%)")
    print(f"Val:   dates {train_end+1}-{val_end}   | {len(val_df):,} rows ({len(val_df)/total*100:.0f}%)")
    print(f"Test:  dates {val_end+1}-{total_dates[-1]}  | {len(test_df):,} rows ({len(test_df)/total*100:.0f}%)")
    return train_df, val_df, test_df


# ============================================================
# IMPUTATION
# ============================================================
# Impute with training median only (no leakage from val/test)
def impute_features(train_df, val_df, test_df, all_features):
    for col in all_features:
        med = train_df[col].median()
        if hasattr(med, 'iloc'):
            med = med.iloc[0]
        train_df[col] = train_df[col].fillna(float(med))
        val_df[col] = val_df[col].fillna(float(med))
        test_df[col] = test_df[col].fillna(float(med))
    gc.collect()
    print(f"\n=== Imputation ===")
    print(f"  Train nulls: {train_df[all_features].isnull().sum().sum()}")
    print(f"  Val nulls:   {val_df[all_features].isnull().sum().sum()}")
    print(f"  Test nulls:  {test_df[all_features].isnull().sum().sum()}")
    return train_df, val_df, test_df


# ============================================================
# PREPARE FINAL ARRAYS
# ============================================================
# Extract X(features), y(target), w(weights) for each split
def prepare_arrays(train_df, val_df, test_df, all_features, target_col='responder_6'):
    X_train, y_train = train_df[all_features], train_df[target_col]
    X_val, y_val = val_df[all_features], val_df[target_col]
    X_test, y_test = test_df[all_features], test_df[target_col]
    w_train, w_val, w_test = train_df['weight'], val_df['weight'], test_df['weight']
    print(f"\n=== Final Arrays ===")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    return (X_train, y_train, w_train, X_val, y_val, w_val, X_test, y_test, w_test)


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline(config_path='config/process/process1.yaml'):
    """Run the full data processing pipeline."""
    # Load config if available, otherwise use defaults
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    sample_size = config.get('sample_size', 4_000_000)
    null_threshold = config.get('null_threshold', 0.5)
    key_features = config.get('key_features', ['feature_46', 'feature_22'])
    windows = config.get('windows', [5])
    train_pct = config.get('train_pct', 0.70)
    val_pct = config.get('val_pct', 0.85)
    output_dir = config.get('output_dir', 'data/processed')

    # Run pipeline
    df = load_data(sample_size)
    df = clean_data(df, null_threshold)
    df = create_lag_features(df)
    df = create_rolling_features(df, key_features, windows)
    df = create_interaction_features(df)
    feature_info = get_feature_groups(df)
    all_features = feature_info['all_features']
    train_df, val_df, test_df = temporal_split(df, train_pct, val_pct)
    train_df, val_df, test_df = impute_features(train_df, val_df, test_df, all_features)

    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(output_dir, 'train.parquet'), index=False)
    val_df.to_parquet(os.path.join(output_dir, 'val.parquet'), index=False)
    test_df.to_parquet(os.path.join(output_dir, 'test.parquet'), index=False)
    with open(os.path.join(output_dir, 'feature_info.json'), 'w') as f:
        json.dump(feature_info, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    return train_df, val_df, test_df, feature_info


if __name__ == '__main__':
    run_pipeline()

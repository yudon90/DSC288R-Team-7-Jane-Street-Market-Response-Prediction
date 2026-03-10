"""
test_process.py — Unit tests for src/process.py
Run: pytest tests/test_process.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.process import (
    clean_data, create_lag_features, create_rolling_features,
    create_interaction_features, get_feature_groups,
    temporal_split, impute_features, prepare_arrays,
)


@pytest.fixture
def sample_df():
    """Create a small synthetic dataset mimicking real data structure."""
    np.random.seed(42)
    n_rows = 200
    n_symbols = 4
    rows_per_symbol = n_rows // n_symbols
    records = []
    for sym in range(n_symbols):
        for i in range(rows_per_symbol):
            row = {
                'date_id': i // 5, 'time_id': i % 5, 'symbol_id': sym,
                'weight': np.random.uniform(0.5, 2.0),
                'responder_6': np.random.normal(0, 1),
                'responder_7': np.random.normal(0, 1),
            }
            for r in range(9):
                if r not in [6, 7]:
                    row[f'responder_{r}'] = np.random.normal(0, 1)
            for f in range(10):
                row[f'feature_{f:02d}'] = np.random.normal(0, 1)
            records.append(row)
    return pd.DataFrame(records)


@pytest.fixture
def sample_df_with_nulls(sample_df):
    df = sample_df.copy()
    df['feature_bad'] = np.nan
    df.loc[:10, 'feature_bad'] = 1.0
    return df


def test_clean_data_drops_nulls(sample_df_with_nulls):
    """Verify columns with >50% nulls are dropped."""
    cleaned = clean_data(sample_df_with_nulls)
    assert 'feature_bad' not in cleaned.columns

def test_clean_data_drops_responders(sample_df):
    """Verify non-target responders are dropped."""
    cleaned = clean_data(sample_df)
    assert 'responder_6' in cleaned.columns
    assert 'responder_7' in cleaned.columns
    assert 'responder_0' not in cleaned.columns

def test_clean_data_preserves_rows(sample_df):
    """Verify no rows are lost during cleaning."""
    cleaned = clean_data(sample_df)
    assert len(cleaned) == len(sample_df)

def test_lag_features_created(sample_df):
    """Verify both lag columns are created."""
    cleaned = clean_data(sample_df)
    result = create_lag_features(cleaned)
    assert 'responder_6_lag_1' in result.columns
    assert 'responder_7_lag_1' in result.columns
    assert 'responder_7' not in result.columns

def test_lag_drops_first_row_per_symbol(sample_df):
    """Verify first row per symbol is dropped."""
    cleaned = clean_data(sample_df)
    n_symbols = cleaned['symbol_id'].nunique()
    rows_before = len(cleaned)
    result = create_lag_features(cleaned)
    assert rows_before - len(result) == n_symbols

def test_rolling_features_created(sample_df):
    """Verify rolling columns with roll_ prefix are created."""
    cleaned = clean_data(sample_df)
    result = create_lag_features(cleaned)
    result = create_rolling_features(result, key_features=['feature_00'], windows=[3])
    assert 'roll_feature_00_mean_3' in result.columns
    assert 'roll_feature_00_std_3' in result.columns

def test_rolling_no_rows_lost(sample_df):
    """Verify no rows lost during rolling feature creation."""
    cleaned = clean_data(sample_df)
    result = create_lag_features(cleaned)
    rows_before = len(result)
    result = create_rolling_features(result, key_features=['feature_00'], windows=[3])
    assert len(result) == rows_before

def test_interaction_feature(sample_df):
    """Verify interaction = lag_6 * lag_7."""
    cleaned = clean_data(sample_df)
    result = create_lag_features(cleaned)
    result = create_interaction_features(result)
    expected = result['responder_6_lag_1'] * result['responder_7_lag_1']
    pd.testing.assert_series_equal(result['resp6_x_resp7'], expected, check_names=False)

def test_feature_groups_no_overlap(sample_df):
    """Verify feature groups don't double-count."""
    cleaned = clean_data(sample_df)
    result = create_lag_features(cleaned)
    result = create_rolling_features(result, key_features=['feature_00'], windows=[3])
    result = create_interaction_features(result)
    groups = get_feature_groups(result)
    total = (len(groups['market_cols']) + len(groups['lag_cols']) +
             len(groups['rolling_cols']) + len(groups['interaction_cols']))
    assert total == len(groups['all_features'])

def test_temporal_split_no_overlap(sample_df):
    """Verify no dates appear in multiple splits."""
    cleaned = clean_data(sample_df)
    result = create_lag_features(cleaned)
    train_df, val_df, test_df = temporal_split(result)
    train_dates = set(train_df['date_id'].unique())
    val_dates = set(val_df['date_id'].unique())
    test_dates = set(test_df['date_id'].unique())
    assert len(train_dates & val_dates) == 0
    assert len(val_dates & test_dates) == 0

def test_temporal_split_chronological(sample_df):
    """Verify train < val < test in date order."""
    cleaned = clean_data(sample_df)
    result = create_lag_features(cleaned)
    train_df, val_df, test_df = temporal_split(result)
    assert train_df['date_id'].max() < val_df['date_id'].min()
    assert val_df['date_id'].max() < test_df['date_id'].min()

def test_impute_removes_nulls(sample_df):
    """Verify no nulls remain after imputation."""
    cleaned = clean_data(sample_df)
    result = create_lag_features(cleaned)
    result = create_rolling_features(result, key_features=['feature_00'], windows=[3])
    result = create_interaction_features(result)
    groups = get_feature_groups(result)
    train_df, val_df, test_df = temporal_split(result)
    train_df, val_df, test_df = impute_features(train_df, val_df, test_df, groups['all_features'])
    assert train_df[groups['all_features']].isnull().sum().sum() == 0

def test_prepare_arrays_shapes(sample_df):
    """Verify X, y, w arrays have correct shapes."""
    cleaned = clean_data(sample_df)
    result = create_lag_features(cleaned)
    result = create_rolling_features(result, key_features=['feature_00'], windows=[3])
    result = create_interaction_features(result)
    groups = get_feature_groups(result)
    train_df, val_df, test_df = temporal_split(result)
    train_df, val_df, test_df = impute_features(train_df, val_df, test_df, groups['all_features'])
    arrays = prepare_arrays(train_df, val_df, test_df, groups['all_features'])
    X_train, y_train, w_train = arrays[0], arrays[1], arrays[2]
    assert X_train.shape[1] == len(groups['all_features'])
    assert len(y_train) == X_train.shape[0]

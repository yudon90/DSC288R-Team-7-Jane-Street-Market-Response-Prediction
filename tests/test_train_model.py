"""
test_train_model.py — Unit tests for src/train_model.py

Run: pytest tests/test_train_model.py -v
"""

# NOTE: If dependencies are missing, run one of these in a Jupyter cell:
#   !pip install joblib scikit-learn xgboost
#   !pip3 install joblib scikit-learn xgboost
#   import sys; !{sys.executable} -m pip install joblib scikit-learn xgboost

import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import with aliases to avoid name collision with test functions
from src.train_model import (
    train_trivial as _train_trivial,
    test_trivial as _eval_trivial,
    train_ridge as _train_ridge,
    test_ridge as _eval_ridge,
    train_rf as _train_rf,
    test_rf as _eval_rf,
    train_xgboost_default as _train_xgb_default,
    test_xgboost_default as _eval_xgb_default,
    train_xgboost_tuned as _train_xgb_tuned,
    test_xgboost_tuned as _eval_xgb_tuned,
    train_stacking as _train_stacking,
    test_stacking as _eval_stacking,
    run_ablation as _run_ablation,
)


# ============================================================
# FIXTURES — Small synthetic data for testing
# ============================================================
@pytest.fixture
def model_data():
    """Create small synthetic train/val/test data for model testing."""
    np.random.seed(42)
    n_features = 10
    feature_names = [f'feature_{i:02d}' for i in range(n_features)]

    def make_df(n_rows):
        X = pd.DataFrame(np.random.randn(n_rows, n_features), columns=feature_names)
        y = pd.Series(np.random.randn(n_rows), name='responder_6')
        w = pd.Series(np.random.uniform(0.5, 2.0, n_rows), name='weight')
        return X, y, w

    # 1000 train, 200 val, 200 test — small enough to run in seconds
    X_train, y_train, w_train = make_df(1000)
    X_val, y_val, w_val = make_df(200)
    X_test, y_test, w_test = make_df(200)

    return {
        'X_train': X_train, 'y_train': y_train, 'w_train': w_train,
        'X_val': X_val, 'y_val': y_val, 'w_val': w_val,
        'X_test': X_test, 'y_test': y_test, 'w_test': w_test,
        'feature_names': feature_names,
    }


# ============================================================
# TESTS: Trivial Baseline
# ============================================================
# Verify trivial baseline trains and produces predictions with R^2 near 0
def test_trivial(model_data):
    train_mean, val_rmse, val_r2 = _train_trivial(
        model_data['y_train'], model_data['y_val'], model_data['w_val'])
    test_rmse, test_r2 = _eval_trivial(
        train_mean, model_data['y_test'], model_data['w_test'])
    assert isinstance(train_mean, float)
    assert test_rmse > 0
    assert abs(val_r2) < 0.1  # predicting mean explains nothing


# ============================================================
# TESTS: Ridge Regression
# ============================================================
# Verify Ridge trains and produces correct number of predictions
def test_ridge(model_data):
    model, scaler, val_rmse, val_r2 = _train_ridge(
        model_data['X_train'], model_data['y_train'], model_data['w_train'],
        model_data['X_val'], model_data['y_val'], model_data['w_val'])
    preds, test_rmse, test_r2 = _eval_ridge(
        model, scaler, model_data['X_test'], model_data['y_test'], model_data['w_test'])
    assert len(preds) == len(model_data['y_test'])
    assert test_rmse > 0


# ============================================================
# TESTS: Random Forest
# ============================================================
# Verify Random Forest trains and produces correct number of predictions
def test_random_forest(model_data):
    model, val_rmse, val_r2 = _train_rf(
        model_data['X_train'], model_data['y_train'], model_data['w_train'],
        model_data['X_val'], model_data['y_val'], model_data['w_val'])
    preds, test_rmse, test_r2 = _eval_rf(
        model, model_data['X_test'], model_data['y_test'], model_data['w_test'])
    assert len(preds) == len(model_data['y_test'])
    assert test_rmse > 0


# ============================================================
# TESTS: XGBoost Default
# ============================================================
# Verify XGBoost Default trains and produces correct number of predictions
def test_xgboost_default(model_data):
    model, val_rmse, val_r2 = _train_xgb_default(
        model_data['X_train'], model_data['y_train'], model_data['w_train'],
        model_data['X_val'], model_data['y_val'], model_data['w_val'])
    preds, test_rmse, test_r2 = _eval_xgb_default(
        model, model_data['X_test'], model_data['y_test'], model_data['w_test'])
    assert len(preds) == len(model_data['y_test'])
    assert test_rmse > 0


# ============================================================
# TESTS: XGBoost Tuned
# ============================================================
# Verify XGBoost Tuned trains with early stopping kicking in before max iterations
def test_xgboost_tuned(model_data):
    model, val_rmse, val_r2 = _train_xgb_tuned(
        model_data['X_train'], model_data['y_train'], model_data['w_train'],
        model_data['X_val'], model_data['y_val'], model_data['w_val'])
    preds, test_rmse, test_r2 = _eval_xgb_tuned(
        model, model_data['X_test'], model_data['y_test'], model_data['w_test'])
    assert len(preds) == len(model_data['y_test'])
    assert model.best_iteration < 2000  # early stopping should kick in


# ============================================================
# TESTS: Stacking Ensemble
# ============================================================
# Verify Stacking trains base models, blends via meta-model with 2 coefficients
def test_stacking(model_data):
    # Train base models first
    ridge_model, ridge_scaler, _, _ = _train_ridge(
        model_data['X_train'], model_data['y_train'], model_data['w_train'],
        model_data['X_val'], model_data['y_val'], model_data['w_val'])
    xgbt_model, _, _ = _train_xgb_tuned(
        model_data['X_train'], model_data['y_train'], model_data['w_train'],
        model_data['X_val'], model_data['y_val'], model_data['w_val'])
    # Train and test stacking
    meta_model, meta_scaler, val_rmse, val_r2 = _train_stacking(
        ridge_model, ridge_scaler, xgbt_model,
        model_data['X_train'], model_data['y_train'], model_data['w_train'],
        model_data['X_val'], model_data['y_val'], model_data['w_val'])
    preds, test_rmse, test_r2 = _eval_stacking(
        meta_model, meta_scaler, ridge_model, ridge_scaler, xgbt_model,
        model_data['X_test'], model_data['y_test'], model_data['w_test'])
    assert len(preds) == len(model_data['y_test'])
    assert len(meta_model.coef_) == 2  # one coefficient per base model


# ============================================================
# TESTS: Ablation Study
# ============================================================
# Verify ablation returns DataFrame with correct structure and number of rows
def test_ablation(model_data):
    feature_groups = {
        'Group A': model_data['feature_names'][:5],
        'Group B': model_data['feature_names'],
    }
    result = _run_ablation(
        feature_groups,
        model_data['X_train'], model_data['y_train'], model_data['w_train'],
        model_data['X_val'], model_data['y_val'], model_data['w_val'],
        model_data['X_test'], model_data['y_test'], model_data['w_test'])
    assert len(result) == 2  # two feature groups tested
    assert 'Test RMSE' in result.columns
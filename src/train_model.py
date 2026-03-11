"""
train_model.py — Model Training, Testing, and Evaluation

Group 7: Xiaolong Yu, Christopher Spears, Donald Yu, Mahir Oza
DSC 288R Winter 2026 — Jane Street Real-Time Market Prediction

Models: Trivial, Ridge, Random Forest, XGBoost Default, XGBoost Tuned, Stacking

Usage:
  python src/train_model.py
  Reads processed data from data/processed/
  Saves results to data/final/ and models to models/
"""

import os
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# ============================================================
# EVALUATION HELPERS
# ============================================================
# Weighted metrics — trade weights emphasize higher-importance trades
def weighted_rmse(y_true, y_pred, w):
    return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=w))

def weighted_r2(y_true, y_pred, w):
    return r2_score(y_true, y_pred, sample_weight=w)


# ============================================================
# MODEL 1: TRIVIAL BASELINE
# ============================================================
# Predict training mean for every trade — simplest possible baseline
def train_trivial(y_train, y_val, w_val):
    train_mean = y_train.mean()

    # Use same prediction (mean) for every row in validation
    val_pred = np.full(len(y_val), train_mean)
    val_rmse = weighted_rmse(y_val, val_pred, w_val)
    val_r2 = weighted_r2(y_val, val_pred, w_val)
    print(f"=== Trivial Baseline ===")
    print(f"Training mean: {train_mean:.6f}")
    print(f"[Validate] RMSE: {val_rmse:.4f} | R^2: {val_r2:.4f}")
    return train_mean, val_rmse, val_r2


# Evaluate trivial baseline on held-out test set
def test_trivial(train_mean, y_test, w_test):
    test_pred = np.full(len(y_test), train_mean)
    test_rmse = weighted_rmse(y_test, test_pred, w_test)
    test_r2 = weighted_r2(y_test, test_pred, w_test)
    print(f"[Test]     RMSE: {test_rmse:.4f} | R^2: {test_r2:.4f}\n")
    return test_rmse, test_r2


# ============================================================
# MODEL 2: RIDGE REGRESSION
# ============================================================
# Ridge: linear model with L2 regularization — requires scaling
def train_ridge(X_train, y_train, w_train, X_val, y_val, w_val, alpha=1.0):
    start = time.time()

    # Scale features (Ridge is sensitive to feature magnitude)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_vl_sc = scaler.transform(X_val)

    model = Ridge(alpha=alpha)  # alpha controls L2 regularization strength
    model.fit(X_tr_sc, y_train, sample_weight=w_train)
    val_pred = model.predict(X_vl_sc)
    val_rmse = weighted_rmse(y_val, val_pred, w_val)
    val_r2 = weighted_r2(y_val, val_pred, w_val)
    print(f"=== Ridge Regression ===")
    print(f"[Train]    Time: {time.time()-start:.1f}s")
    print(f"[Validate] RMSE: {val_rmse:.4f} | R^2: {val_r2:.4f}")
    # Returns model + scaler (needed for test scaling later)
    return model, scaler, val_rmse, val_r2


# Evaluate Ridge on held-out test set (scale with same scaler from training)
def test_ridge(model, scaler, X_test, y_test, w_test):
    X_te_sc = scaler.transform(X_test)
    preds = model.predict(X_te_sc)
    test_rmse = weighted_rmse(y_test, preds, w_test)
    test_r2 = weighted_r2(y_test, preds, w_test)
    print(f"[Test]     RMSE: {test_rmse:.4f} | R^2: {test_r2:.4f}\n")
    # preds saved for stacking ensemble later
    return preds, test_rmse, test_r2


# ============================================================
# MODEL 3: RANDOM FOREST
# ============================================================
# Random Forest: ensemble of independent trees, each trained on random data subset
def train_rf(X_train, y_train, w_train, X_val, y_val, w_val,
             n_estimators=50, max_depth=15, min_samples_leaf=50):
    start = time.time()
    model = RandomForestRegressor(
        n_estimators=n_estimators,  # number of trees in the forest
        max_depth=max_depth,         # max depth per tree
        min_samples_leaf=min_samples_leaf,  # minimum samples per leaf
        n_jobs=-1, random_state=42
    )
    # No scaling needed — tree models are not sensitive to feature magnitude
    model.fit(X_train, y_train, sample_weight=w_train)
    val_pred = model.predict(X_val)
    val_rmse = weighted_rmse(y_val, val_pred, w_val)
    val_r2 = weighted_r2(y_val, val_pred, w_val)
    print(f"=== Random Forest ===")
    print(f"[Train]    Time: {time.time()-start:.1f}s")
    print(f"[Validate] RMSE: {val_rmse:.4f} | R^2: {val_r2:.4f}")
    return model, val_rmse, val_r2


# Evaluate Random Forest on test set
def test_rf(model, X_test, y_test, w_test):
    preds = model.predict(X_test)
    test_rmse = weighted_rmse(y_test, preds, w_test)
    test_r2 = weighted_r2(y_test, preds, w_test)
    print(f"[Test]     RMSE: {test_rmse:.4f} | R^2: {test_r2:.4f}\n")
    return preds, test_rmse, test_r2


# ============================================================
# MODEL 4: XGBOOST (DEFAULT)
# ============================================================
# XGBoost: gradient boosted trees — each tree corrects errors of previous ones
def train_xgboost_default(X_train, y_train, w_train, X_val, y_val, w_val,
                          n_estimators=300, max_depth=4, learning_rate=0.1,
                          subsample=0.9, colsample_bytree=0.9,
                          reg_alpha=0.01, reg_lambda=0.5):
    start = time.time()
    # Safety check: remove duplicate columns (XGBoost crashes on duplicates)
    X_tr = X_train.loc[:, ~X_train.columns.duplicated()]
    X_vl = X_val.loc[:, ~X_val.columns.duplicated()]
    model = XGBRegressor(
        n_estimators=n_estimators,   # number of boosting rounds
        max_depth=max_depth,          # shallow trees to prevent overfitting
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,          # L1 regularization
        reg_lambda=reg_lambda,        # L2 regularization
        tree_method='hist', n_jobs=-1, random_state=42, verbosity=0
    )
    # eval_set monitors validation loss during training
    model.fit(X_tr.values, y_train.values, sample_weight=w_train.values,
              eval_set=[(X_vl.values, y_val.values)],
              sample_weight_eval_set=[w_val.values], verbose=False)
    val_pred = model.predict(X_vl.values)
    val_rmse = weighted_rmse(y_val, val_pred, w_val)
    val_r2 = weighted_r2(y_val, val_pred, w_val)
    print(f"=== XGBoost (Default) ===")
    print(f"[Train]    Time: {time.time()-start:.1f}s")
    print(f"[Validate] RMSE: {val_rmse:.4f} | R^2: {val_r2:.4f}")
    return model, val_rmse, val_r2


# Evaluate XGBoost Default on held-out test set
def test_xgboost_default(model, X_test, y_test, w_test):
    X_te = X_test.loc[:, ~X_test.columns.duplicated()]
    preds = model.predict(X_te.values)
    test_rmse = weighted_rmse(y_test, preds, w_test)
    test_r2 = weighted_r2(y_test, preds, w_test)
    print(f"[Test]     RMSE: {test_rmse:.4f} | R^2: {test_r2:.4f}\n")
    return preds, test_rmse, test_r2


# ============================================================
# MODEL 5: XGBOOST (TUNED)
# ============================================================
# XGBoost Tuned: more trees, deeper, slower learning, finds optimal stopping point
def train_xgboost_tuned(X_train, y_train, w_train, X_val, y_val, w_val,
                        n_estimators=2000, max_depth=8, learning_rate=0.03,
                        subsample=0.7, colsample_bytree=0.7,
                        reg_alpha=0.1, reg_lambda=2.0, min_child_weight=10,
                        early_stopping_rounds=100):
    start = time.time()
    # Safety check: remove duplicate columns (XGBoost crashes on duplicates)
    X_tr = X_train.loc[:, ~X_train.columns.duplicated()]
    X_vl = X_val.loc[:, ~X_val.columns.duplicated()]
    model = XGBRegressor(
        n_estimators=n_estimators,    # max boosting rounds (early stopping will cut short)
        max_depth=max_depth,           # deeper trees to capture complex patterns
        learning_rate=learning_rate,   # smaller steps for finer optimization
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,           # L1 regularization (stronger than default)
        reg_lambda=reg_lambda,         # L2 regularization (stronger than default)
        min_child_weight=min_child_weight,
        tree_method='hist', n_jobs=-1, random_state=42,
        verbosity=0, early_stopping_rounds=early_stopping_rounds
    )
    # eval_set required for early stopping — monitors validation loss
    model.fit(X_tr.values, y_train.values, sample_weight=w_train.values,
              eval_set=[(X_vl.values, y_val.values)],
              sample_weight_eval_set=[w_val.values], verbose=False)
    val_pred = model.predict(X_vl.values)
    val_rmse = weighted_rmse(y_val, val_pred, w_val)
    val_r2 = weighted_r2(y_val, val_pred, w_val)
    print(f"=== XGBoost (Tuned) ===")
    print(f"Best iteration: {model.best_iteration}")  # actual trees used before stopping
    print(f"[Train]    Time: {time.time()-start:.1f}s")
    print(f"[Validate] RMSE: {val_rmse:.4f} | R^2: {val_r2:.4f}")
    return model, val_rmse, val_r2


# Evaluate XGBoost Tuned on test set
def test_xgboost_tuned(model, X_test, y_test, w_test):
    X_te = X_test.loc[:, ~X_test.columns.duplicated()]
    preds = model.predict(X_te.values)
    test_rmse = weighted_rmse(y_test, preds, w_test)
    test_r2 = weighted_r2(y_test, preds, w_test)
    print(f"[Test]     RMSE: {test_rmse:.4f} | R^2: {test_r2:.4f}\n")
    # preds saved for stacking ensemble later
    return preds, test_rmse, test_r2


# ============================================================
# MODEL 6: STACKING ENSEMBLE
# ============================================================
# Stacking: blend Ridge (linear) + XGBoost (non-linear) via a meta-learner
def train_stacking(ridge_model, ridge_scaler, xgbt_model,
                   X_train, y_train, w_train, X_val, y_val, w_val):
    start = time.time()

    # Step 1: Get predictions from both base models on validation set
    X_val_sc = ridge_scaler.transform(X_val)
    X_val_dedup = X_val.loc[:, ~X_val.columns.duplicated()]
    ridge_val_pred = ridge_model.predict(X_val_sc)
    xgbt_val_pred = xgbt_model.predict(X_val_dedup.values)

    # Step 2: Stack predictions as meta-features
    meta_val = np.column_stack([ridge_val_pred, xgbt_val_pred])

    # Step 3: Train meta-model to learn optimal blend weights
    meta_scaler = StandardScaler()
    meta_val_sc = meta_scaler.fit_transform(meta_val)
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_val_sc, y_val, sample_weight=w_val)

    meta_val_pred = meta_model.predict(meta_val_sc)
    val_rmse = weighted_rmse(y_val, meta_val_pred, w_val)
    val_r2 = weighted_r2(y_val, meta_val_pred, w_val)
    print(f"=== Stacking Ensemble (Ridge + XGBoost Tuned -> Meta Ridge) ===")
    print(f"[Train]    Time: {time.time()-start:.1f}s")
    print(f"[Validate] RMSE: {val_rmse:.4f} | R^2: {val_r2:.4f}")
    # Coefficients show how much the meta-model trusts each base model
    print(f"  Meta-model coefficients: Ridge={meta_model.coef_[0]:.4f}, XGBoost={meta_model.coef_[1]:.4f}")
    return meta_model, meta_scaler, val_rmse, val_r2


# Evaluate stacking ensemble on test set
def test_stacking(meta_model, meta_scaler, ridge_model, ridge_scaler, xgbt_model,
                  X_test, y_test, w_test):
    # Get base model predictions on test set
    X_te_sc = ridge_scaler.transform(X_test)
    X_te_dedup = X_test.loc[:, ~X_test.columns.duplicated()]
    ridge_test_pred = ridge_model.predict(X_te_sc)
    xgbt_test_pred = xgbt_model.predict(X_te_dedup.values)

    # Stack and blend through meta-model
    meta_test = np.column_stack([ridge_test_pred, xgbt_test_pred])
    meta_test_sc = meta_scaler.transform(meta_test)
    preds = meta_model.predict(meta_test_sc)

    test_rmse = weighted_rmse(y_test, preds, w_test)
    test_r2 = weighted_r2(y_test, preds, w_test)
    print(f"[Test]   RMSE: {test_rmse:.4f} | R^2: {test_r2:.4f}\n")
    return preds, test_rmse, test_r2


# ============================================================
# ABLATION STUDY
# ============================================================
# Run XGBoost Tuned with different feature subsets to measure each group's contribution
def run_ablation(feature_groups, X_train, y_train, w_train,
                 X_val, y_val, w_val, X_test, y_test, w_test):
    ablation_results = []
    for name, features in feature_groups.items():
        features = list(dict.fromkeys(features))

        # Select only this subset of features
        X_tr = X_train[features].loc[:, ~X_train[features].columns.duplicated()]
        X_vl = X_val[features].loc[:, ~X_val[features].columns.duplicated()]
        X_te = X_test[features].loc[:, ~X_test[features].columns.duplicated()]

        # Train XGBoost Tuned with same hyperparameters each time
        start = time.time()
        model = XGBRegressor(
            n_estimators=2000, max_depth=8, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=2.0, min_child_weight=10,
            tree_method='hist', n_jobs=-1, random_state=42,
            verbosity=0, early_stopping_rounds=100
        )
        model.fit(X_tr.values, y_train.values, sample_weight=w_train.values,
                  eval_set=[(X_vl.values, y_val.values)],
                  sample_weight_eval_set=[w_val.values], verbose=False)

        # Evaluate this feature subset on val and test
        val_rmse = weighted_rmse(y_val, model.predict(X_vl.values), w_val)
        test_rmse = weighted_rmse(y_test, model.predict(X_te.values), w_test)
        test_r2 = weighted_r2(y_test, model.predict(X_te.values), w_test)
        ablation_results.append({
            'Feature Set': name, 'Num Features': len(features),
            'Val RMSE': val_rmse, 'Test RMSE': test_rmse, 'Test R^2': test_r2,
        })

        print(f"=== Ablation: {name} ({len(features)} features) ===")
        print(f"[Validate] RMSE: {val_rmse:.4f}")
        print(f"[Test]     RMSE: {test_rmse:.4f} | R^2: {test_r2:.4f}")
        print(f"Time: {time.time()-start:.1f}s\n")
        del model; gc.collect()
    return pd.DataFrame(ablation_results)


# ============================================================
# VISUALIZATION
# ============================================================
# RMSE and R^2 side by side
def plot_model_comparison(results_df, best_model):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    results_sorted = results_df.sort_values('Test RMSE', ascending=True)
    colors = ['red' if 'Trivial' in m else 'green' if m == best_model else 'steelblue'
              for m in results_sorted['Model']]

    axes[0].barh(results_sorted['Model'], results_sorted['Test RMSE'], color=colors)
    axes[0].set_xlabel('Test RMSE (lower is better)')
    axes[0].set_title('Model Comparison - Test RMSE')
    for i, rmse in enumerate(results_sorted['Test RMSE']):
        axes[0].text(rmse + 0.005, i, f'{rmse:.4f}', va='center', fontsize=10)

    axes[1].barh(results_sorted['Model'], results_sorted['Test R^2'], color=colors)
    axes[1].set_xlabel('Test R^2 (higher is better)')
    axes[1].set_title('Model Comparison - Test R^2')
    for i, r2 in enumerate(results_sorted['Test R^2']):
        axes[1].text(r2 + 0.005, i, f'{r2:.4f}', va='center', fontsize=10)

    plt.tight_layout()
    return fig


# RMSE and R^2 side by side
def plot_ablation(ablation_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ablation_sorted = ablation_df.sort_values('Test RMSE', ascending=True)
    colors = ['green' if name == 'All features' else 'steelblue'
              for name in ablation_sorted['Feature Set']]

    axes[0].barh(ablation_sorted['Feature Set'], ablation_sorted['Test RMSE'], color=colors)
    axes[0].set_xlabel('Test RMSE (lower is better)')
    axes[0].set_title('Ablation Study - Test RMSE by Feature Set')
    for i, rmse in enumerate(ablation_sorted['Test RMSE']):
        axes[0].text(rmse + 0.001, i, f'{rmse:.4f}', va='center', fontsize=10)

    axes[1].barh(ablation_sorted['Feature Set'], ablation_sorted['Test R^2'], color=colors)
    axes[1].set_xlabel('Test R^2 (higher is better)')
    axes[1].set_title('Ablation Study - Test R^2 by Feature Set')
    for i, r2 in enumerate(ablation_sorted['Test R^2']):
        axes[1].text(r2 + 0.001, i, f'{r2:.4f}', va='center', fontsize=10)

    plt.tight_layout()
    return fig


# ============================================================
# SAVE MODELS
# ============================================================
# Save all trained models using joblib for reproducibility
def save_models(models_dict, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models_dict.items():
        path = os.path.join(output_dir, f'{name}.pkl')
        joblib.dump(model, path)
    print(f"\nAll models saved to {output_dir}/")
    for name in models_dict:
        print(f"  {name}.pkl")


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_training(data_dir='data/processed', output_dir='data/final', model_dir='models'):
    """Run full model training, evaluation, and ablation pipeline."""

    # Load processed data
    print("Loading processed data...")
    train_df = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
    val_df = pd.read_parquet(os.path.join(data_dir, 'val.parquet'))
    test_df = pd.read_parquet(os.path.join(data_dir, 'test.parquet'))
    with open(os.path.join(data_dir, 'feature_info.json'), 'r') as f:
        feature_info = json.load(f)

    # Extract feature groups for use in modeling and ablation
    all_features = feature_info['all_features']
    market_cols = feature_info['market_cols']
    lag_cols = feature_info['lag_cols']
    rolling_cols = feature_info['rolling_cols']

    # Prepare X(features), y(target), w(weights) for each split
    target_col = 'responder_6'
    X_train, y_train = train_df[all_features], train_df[target_col]
    X_val, y_val = val_df[all_features], val_df[target_col]
    X_test, y_test = test_df[all_features], test_df[target_col]
    w_train, w_val, w_test = train_df['weight'], val_df['weight'], test_df['weight']

    # ========== TRAIN ALL MODELS ==========
    print("\n" + "=" * 60 + "\nTRAINING ALL MODELS\n" + "=" * 60 + "\n")

    train_mean, triv_val_rmse, triv_val_r2 = train_trivial(y_train, y_val, w_val)
    triv_test_rmse, triv_test_r2 = test_trivial(train_mean, y_test, w_test)

    ridge_model, ridge_scaler, ridge_val_rmse, ridge_val_r2 = train_ridge(
        X_train, y_train, w_train, X_val, y_val, w_val)
    ridge_preds, ridge_test_rmse, ridge_test_r2 = test_ridge(
        ridge_model, ridge_scaler, X_test, y_test, w_test)

    rf_model, rf_val_rmse, rf_val_r2 = train_rf(
        X_train, y_train, w_train, X_val, y_val, w_val)
    rf_preds, rf_test_rmse, rf_test_r2 = test_rf(rf_model, X_test, y_test, w_test)

    xgb_model, xgb_val_rmse, xgb_val_r2 = train_xgboost_default(
        X_train, y_train, w_train, X_val, y_val, w_val)
    xgb_preds, xgb_test_rmse, xgb_test_r2 = test_xgboost_default(
        xgb_model, X_test, y_test, w_test)

    xgbt_model, xgbt_val_rmse, xgbt_val_r2 = train_xgboost_tuned(
        X_train, y_train, w_train, X_val, y_val, w_val)
    xgbt_preds, xgbt_test_rmse, xgbt_test_r2 = test_xgboost_tuned(
        xgbt_model, X_test, y_test, w_test)

    stack_model, stack_scaler, stack_val_rmse, stack_val_r2 = train_stacking(
        ridge_model, ridge_scaler, xgbt_model,
        X_train, y_train, w_train, X_val, y_val, w_val)
    stack_preds, stack_test_rmse, stack_test_r2 = test_stacking(
        stack_model, stack_scaler, ridge_model, ridge_scaler, xgbt_model,
        X_test, y_test, w_test)

    # ========== SAVE ALL MODELS ==========
    save_models({
        'trivial_mean': train_mean,
        'ridge_model': ridge_model,
        'ridge_scaler': ridge_scaler,
        'rf_model': rf_model,
        'xgb_model': xgb_model,
        'xgbt_model': xgbt_model,
        'stack_model': stack_model,
        'stack_scaler': stack_scaler,
    }, model_dir)

    # ========== MODEL COMPARISON ==========
    # Build results table
    results_df = pd.DataFrame([
        {'Model': 'Trivial (predict mean)', 'Val RMSE': triv_val_rmse, 'Test RMSE': triv_test_rmse, 'Test R^2': triv_test_r2},
        {'Model': 'Ridge Regression',       'Val RMSE': ridge_val_rmse, 'Test RMSE': ridge_test_rmse, 'Test R^2': ridge_test_r2},
        {'Model': 'Random Forest',          'Val RMSE': rf_val_rmse, 'Test RMSE': rf_test_rmse, 'Test R^2': rf_test_r2},
        {'Model': 'XGBoost (Default)',      'Val RMSE': xgb_val_rmse, 'Test RMSE': xgb_test_rmse, 'Test R^2': xgb_test_r2},
        {'Model': 'XGBoost (Tuned)',        'Val RMSE': xgbt_val_rmse, 'Test RMSE': xgbt_test_rmse, 'Test R^2': xgbt_test_r2},
        {'Model': 'Stacking Ensemble',      'Val RMSE': stack_val_rmse, 'Test RMSE': stack_test_rmse, 'Test R^2': stack_test_r2},
    ]).sort_values('Test RMSE', ascending=False).reset_index(drop=True)

    print("=" * 60 + "\nFINAL MODEL COMPARISON\n" + "=" * 60)
    print(results_df.to_string(index=False))

    # Identify best model and calculate improvement over trivial baseline
    best = results_df.loc[results_df['Test R^2'].idxmax()]
    improvement = (triv_test_rmse - float(best['Test RMSE'])) / triv_test_rmse * 100
    print(f"\nBest: {best['Model']} — RMSE {best['Test RMSE']:.4f}, R^2 {best['Test R^2']:.4f}")
    print(f"{improvement:.1f}% RMSE improvement over trivial baseline")

    # ========== ABLATION STUDY ==========
    print("\n" + "=" * 60 + "\nABLATION STUDY\n" + "=" * 60 + "\n")

    # Define feature subsets: start with lag only, progressively add more
    feature_groups = {
        'Lag only':      lag_cols,
        'Lag + Raw':     lag_cols + market_cols,
        'Lag + Rolling': lag_cols + rolling_cols,
        'All features':  all_features,
    }

    ablation_df = run_ablation(feature_groups, X_train, y_train, w_train,
                               X_val, y_val, w_val, X_test, y_test, w_test)
    print(ablation_df.to_string(index=False))

    # ========== SAVE RESULTS AND PLOTS ==========
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    ablation_df.to_csv(os.path.join(output_dir, 'ablation_results.csv'), index=False)

    fig1 = plot_model_comparison(results_df, best['Model'])
    fig1.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    fig2 = plot_ablation(ablation_df)
    fig2.savefig(os.path.join(output_dir, 'ablation_study.png'), dpi=150, bbox_inches='tight')

    print(f"\nResults saved to {output_dir}/")
    return results_df, ablation_df


if __name__ == '__main__':
    run_training()

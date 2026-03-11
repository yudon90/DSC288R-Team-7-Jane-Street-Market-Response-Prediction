# Jane Street Real-Time Market Prediction

**Group 7 — DSC 288R Winter 2026, UCSD**

Xiaolong Yu, Christopher Spears, Donald Yu, Mahir Oza

---

## Project Overview

Predicting short-horizon trading returns (`responder_6`) using the Jane Street Real-Time Market Forecasting dataset from Kaggle. We engineer temporal features (lag, rolling window, interaction) and compare six models of increasing complexity: Trivial Baseline, Ridge Regression, Random Forest, XGBoost (Default and Tuned), and a Stacking Ensemble.

**Best result:** Stacking Ensemble — R^2 = 0.8514, RMSE = 0.2964, 61.5% improvement over trivial baseline.

---

## Repository Structure

```
project/
├── config/                          Settings and hyperparameters
│   ├── main.yaml                    Top-level paths and dataset info
│   ├── model/                       One yaml per model
│   │   ├── ridge.yaml
│   │   ├── random_forest.yaml
│   │   ├── xgboost_default.yaml
│   │   ├── xgboost_tuned.yaml
│   │   └── stacking.yaml
│   └── process/
│       └── process1.yaml            Data processing parameters
│
├── data/                            All data (not committed to GitHub)
│   ├── raw/                         Original parquet files from Kaggle
│   ├── processed/                   Output of process.py (train/val/test)
│   ├── final/                       Output of train_model.py (results, plots)
│   └── raw.dvc                      DVC file for data version control
│
├── docs/                            Project documentation
│
├── models/                          Saved trained model files
│
├── notebooks/                       Jupyter notebooks
│   └── capstone_finalVersion.ipynb  Complete end-to-end notebook
│
├── src/                             Source code
│   ├── __init__.py
│   ├── process.py                   Data loading, cleaning, feature engineering
│   └── train_model.py               Model training, testing, comparison, ablation
│
├── tests/                           Unit tests
│   ├── __init__.py
│   ├── test_process.py              Tests for process.py
│   └── test_train_model.py          Tests for train_model.py
│
├── .gitignore
├── Makefile
├── README.md
└── README.txt                       Submission run instructions
```

---

## Dataset

- **Source:** [Jane Street Market Prediction on Kaggle](https://www.kaggle.com/competitions/jane-street-real-time-market-forecasting)
- **Size:** 47 million rows, 79 features, 9 responders
- **Sample used:** 4 million rows (296 dates, 20 symbols)
- **Target:** `responder_6` (continuous short-horizon return)

---

## Data Pipeline (`src/process.py`)

1. **Load** — 4M row sample from Parquet via kagglehub
2. **Clean** — drop 9 columns with >50% nulls, drop non-target responders, sort by symbol/date/time
3. **Feature Engineering:**
   - 2 lag features (`responder_6_lag_1`, `responder_7_lag_1`)
   - 6 rolling features (mean + std for `feature_46`, `feature_22`, `responder_6_lag_1`)
   - 1 interaction feature (`resp6_x_resp7`)
   - Final: **79 features** (70 market + 2 lag + 6 rolling + 1 interaction)
4. **Temporal Split** — 70% train / 15% val / 15% test by date (no random shuffle)
5. **Imputation** — training median applied to all splits (prevents leakage)

---

## Models (`src/train_model.py`)

| Model | Key Parameters | Description |
|-------|---------------|-------------|
| Trivial Baseline | — | Predict training mean for every trade |
| Ridge Regression | alpha=1.0 | Linear + L2 regularization, requires StandardScaler |
| Random Forest | 50 trees, depth 15, min leaf 50 | Bagged decision trees, captures non-linear patterns |
| XGBoost Default | 300 trees, depth 4, lr 0.1 | Gradient boosted trees with moderate settings |
| XGBoost Tuned | 2000 trees, depth 8, lr 0.03 | Early stopping at 100 rounds, stronger regularization |
| Stacking Ensemble | Ridge + XGBoost Tuned | Meta Ridge learns optimal blend of base model predictions |

---

## Results

| Model | Test RMSE | Test R^2 | Improvement |
|-------|-----------|----------|-------------|
| Trivial (mean) | 0.7689 | -0.000 | — |
| Ridge Regression | 0.3031 | 0.8446 | 60.6% |
| Random Forest | 0.3020 | 0.8457 | 60.7% |
| XGBoost (Default) | 0.2986 | 0.8491 | 61.2% |
| XGBoost (Tuned) | 0.2971 | 0.8507 | 61.4% |
| Stacking Ensemble | 0.2964 | 0.8514 | 61.5% |

### Ablation Study (XGBoost Tuned)

| Feature Set | # Features | Test RMSE | Test R^2 |
|-------------|-----------|-----------|----------|
| Lag only | 2 | 0.3603 | 0.7804 |
| Lag + Raw | 72 | 0.3198 | 0.8270 |
| Lag + Rolling | 8 | 0.3503 | 0.7925 |
| All features | 79 | 0.2971 | 0.8507 |

**Key finding:** Lag features alone (just 2 features) capture most of the predictive signal, confirming that temporal autocorrelation is the dominant signal in this dataset.

---

## How to Run

### Option 1: Jupyter Notebook (recommended)
```bash
# Open in Google Colab or Jupyter Notebook
notebooks/capstone_finalVersion.ipynb

# Run all cells top to bottom
# Full run on 4M rows: ~2 hours (Random Forest is slowest)
# Quick run: set sample_size = 500_000 (~15 min)
```

### Option 2: Python Scripts
```bash
# Step 1: Install dependencies
pip install numpy pandas pyarrow scikit-learn xgboost matplotlib seaborn statsmodels kagglehub pyyaml

# Step 2: Process data
python src/process.py

# Step 3: Train models and evaluate
python src/train_model.py
```

### Option 3: Using Makefile
```bash
make install    # Install dependencies
make process    # Run data pipeline
make train      # Train all models
make test       # Run unit tests
make all        # Run everything
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_process.py -v
pytest tests/test_train_model.py -v
```

Tests use small synthetic data (1000 rows) and run in seconds.

---

## Requirements

- Python 3.10+
- numpy
- pandas
- pyarrow
- scikit-learn
- xgboost
- matplotlib
- seaborn
- statsmodels
- kagglehub
- pyyaml
- pytest (for testing)
- joblib
---

## Key Findings

1. **Temporal features dominate:** `responder_6_lag_1` (correlation 0.89) provides far more signal than any raw market feature (max correlation 0.09)
2. **Diminishing returns from complexity:** Ridge to XGBoost is a clear gain, but Stacking adds minimal improvement over XGBoost Tuned alone
3. **Temporal split is critical:** Random splitting would leak future information due to strong autocorrelation (ACF significant at lags 1-5)

---

## References

1. Jane Street Market Prediction Competition, Kaggle
2. Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System
3. Breiman, L. (2001). Random Forests
4. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python

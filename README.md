# DSC288R Team 7: Jane Street Real-Time Market Data Forecasting

Team 7 - Mahir Oza, Christopher Spears, Donald Yu, Xiaolong Yu
 
Project Overview

Predicting short-term market responses in noisy financial environments is a challenging yet critical task for optimizing trading decisions. In this project, we formulate real-time market prediction as a continuous regression problem. The goal is to predict continuous returns (specifically responder_6), which is distinctly more valuable than binary classification as it informs quantitative models of both the direction and the magnitude of the market response. This repository contains the source code, data processing pipelines, and experimental models for the project.

--------
Data Sources

The dataset used in this project is the Jane Street Real-Time Market Data, available on Kaggle.

Source: Jane Street Real-Time Market Data Forecasting, https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/data

Description: The data represents real-world stock market trading data. It includes over 130 anonymized features representing different market signals over time.

Note on Size: Due to the dataset's size (12.3 GB), it is not stored in this repository. Users must download the data directly from Kaggle.

--------
Data Characteristics and Sources

The foundation of this research is the Jane Street Real-Time Market Data, a 12.3 GB dataset representing three years of historical trading activity. This data is characterized by a high volume of anonymized features, representing various market signals that are not explicitly defined, requiring deep exploratory data analysis to uncover underlying patterns. Because of the extreme file size, the data is not hosted directly on GitHub to ensure the repository remains easily navigable and maintainable. Instead, the original data can be retrieved from the Jane Street Kaggle Competition, where it should be downloaded and placed into the local data/raw/ directory for processing.

--------
Methodology and Implementation

Our approach focuses heavily on capturing temporal structures to supplement the existing high-dimensional, anonymized market data.

Feature Engineering: We constructed specific lag and rolling window features directly, rather than relying solely on raw anonymized data. This includes lagged responders (e.g., responder_6_lag_1) and rolling window statistics for the top correlated market features.

Data Splitting: The data is split temporally into training (70%), validation (15%), and test (15%) sets. We split by date rather than randomly because our exploratory data analysis showed strong autocorrelation between nearby trades; random splitting would leak future information into the training set.

Modeling Pipeline: We evaluated six models of increasing complexity: a Trivial Baseline, Ridge Regression, Random Forest, XGBoost (Default and Tuned), and a Stacking Ensemble.

--------
Results and Model Performance

Our evaluation on a dataset of four million rows demonstrates significant improvements over our baseline models. To evaluate success, we compared multiple algorithms using Root Mean Square Error (RMSE) and R-squared metrics.

Trivial (predict mean): Test RMSE: 0.7689 | Test R^2: -0.000 

Ridge Regression: Test RMSE: 0.3031 | Test R^2: 0.8446 

Random Forest: Test RMSE: 0.3020 | Test R^2: 0.8457 

XGBoost (Default): Test RMSE: 0.2987 | Test R^2: 0.8491 

XGBoost (Tuned): Test RMSE: 0.2971 | Test R^2: 0.8507 

Stacking Ensemble: Test RMSE: 0.2964 | Test R^2: 0.8514 

Every trained model beat the trivial baseline by over 60% in RMSE. The Stacking Ensemble achieved the best overall result with a 61.5% improvement over the baseline. Our Tuned XGBoost model, which utilized early stopping triggered at iteration 802 out of 2000, was the best single model. When compared to existing literature, where a published baseline using raw features yielded a public leaderboard R^2 of just 0.004 , our methodology demonstrates that explicit temporal feature engineering is required to extract predictive signals.

--------
Key Findings and Scaling Analysis

Our analysis uncovered several critical insights regarding high-frequency market data:

Lag features dominate the signal: The engineered feature responder_6_lag_1 has a 0.89 correlation with the target. Lag features alone (just 2 features) achieved an R^2 of 0.8465, confirming that the predictive signal relies far more on temporal autocorrelation than on individual raw features.

Nonlinear models outperform linear models: Tree models split data at decision points, finding patterns that linear models miss. Ridge Regression had the weakest performance among our trained models because it cannot model non-linear interactions between features.

Diminishing returns: The performance gap between models shrinks as complexity grows, suggesting we are near the signal ceiling for this specific feature set.

Live-Market Risks: There is a prevalent risk in live prediction settings that by the time the data and model are available to make predictions, the inherent latency means subsequent trades may have already occurred. Additionally, model drift can be exploited by market volatility across different trading phases.

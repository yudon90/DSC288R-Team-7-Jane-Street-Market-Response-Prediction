# DSC288R-Team-7-Jane-Street-Market-Response-Prediction
Jane Street Real-Time Market Data Forecasting

Project Overview

This project focuses on developing a predictive model to forecast market responses using real-time financial data. The goal is to identify trading opportunities by predicting the "action" (whether to trade or not) based on anonymized market features. This repository contains the source code, data processing pipelines, and experimental notebooks for the project.

--------
Data Sources
The dataset used in this project is the Jane Street Real-Time Market Data, available on Kaggle.
Source: Jane Street Real-Time Market Data Forecasting, https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/data
Description: The data represents real-world stock market trading data. It includes over 130 anonymized features representing different market signals over time.

Note on Size: Due to the dataset's size (12.3 GB), it is not stored in this repository. Users must download the data directly from Kaggle and place it in the data/raw/ directory.

--------
Data Characteristics and Sources

The foundation of this research is the Jane Street Real-Time Market Data, a 12.3 GB dataset representing three years of historical trading activity. This data is characterized by a high volume of anonymized features, representing various market signals that are not explicitly defined, requiring deep exploratory data analysis to uncover underlying patterns. Because of the extreme file size, the data is not hosted directly on GitHub to ensure the repository remains easily navigable and maintainable. Instead, the original data can be retrieved from the Jane Street Kaggle Competition, where it should be downloaded and placed into the local data/raw/ directory for processing.

--------
Methodology and Implementation

The project is implemented using a structured, module-based approach to ensure the code is both self-describing and understandable for future collaboration. The workflow begins in the notebooks/ directory with 1_data_exploration.ipynb, where we analyze feature correlations, handle missing values, and visualize the target "resp" values that drive the trading action. Core logic for data transformation and signal processing is moved into the src/ folder to maintain a clean environment for model training. The final models are stored in the models/ directory, allowing for versioned checkpoints and easy comparison of different architectures as we iterate toward a production-ready trading strategy.

--------
Results and Model Performance

Our evaluation on a dataset of four million rows demonstrates significant improvements over our baseline models. To evaluate success, we compared multiple algorithms using Root Mean Square Error (RMSE) and R-squared metrics.

Model,RMSE,R-squared
Trivial (predict mean),0.7689,0.0000
Ridge Regression,0.3031,0.8446
XGBoost (default),0.2987,0.8491
XGBoost (tuned),0.2971,0.8507

The tuned XGBoost model is our best performing algorithm, achieving a 61 percent RMSE reduction compared to the trivial baseline. An R-squared of 0.85 indicates that the model successfully explains 85 percent of the variance in market returns. Furthermore, the tuned XGBoost model improved upon the Ridge Regression baseline by 2 percent, which confirms that nonlinear patterns exist within the data. To prevent overfitting during training, early stopping was implemented and triggered at iteration 802 out of 2000.

When compared to existing literature, the results are highly favorable. Lin's raw feature linear model achieved an R-squared of 0.004. Our pipeline achieved an R-squared of 0.85, representing a 200x improvement. This massive gain is driven primarily by our lag feature engineering rather than model complexity alone.

--------
Key Findings and Scaling Analysis
Our analysis uncovered several critical insights regarding the Jane Street market data and how predictive models behave within it.

Lag features dominate the signal: The feature responder_6_lag_1 alone has a 0.89 correlation with the target variable. This far exceeds any raw feature, which maxed out at a 0.09 correlation. This proves that the predictive signal is highly temporal rather than cross sectional.

Performance varies with dataset size: Training on 2.5 million rows resulted in an RMSE of 0.3376. Training on 3 million rows improved the RMSE to 0.2797. Expanding to 4 million rows shifted the RMSE to 0.2971. This fluctuation indicates that different dataset sizes capture different market regimes during the test period, highlighting a well documented challenge in financial machine learning known as regime changes.

Nonlinear models outperform linear models: XGBoost beat Ridge Regression by approximately 2 percent in RMSE. This confirms the presence of nonlinear feature interactions. However, the vast majority of the predictive signal still originates from robust feature engineering rather than the choice of algorithm.

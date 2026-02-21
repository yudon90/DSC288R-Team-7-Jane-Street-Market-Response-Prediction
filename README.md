# DSC288R-Team-7-Jane-Street-Market-Response-Prediction
Jane Street Real-Time Market Data Forecasting

Project Overview
This project focuses on developing a predictive model to forecast market responses using real-time financial data. The goal is to identify trading opportunities by predicting the "action" (whether to trade or not) based on anonymized market features. This repository contains the source code, data processing pipelines, and experimental notebooks for the DSC 288 Graduate Capstone Project.

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

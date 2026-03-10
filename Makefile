.PHONY: install process train test all clean

install:
	pip install numpy pandas pyarrow scikit-learn xgboost matplotlib seaborn statsmodels kagglehub pyyaml pytest joblib

process:
	python src/process.py

train:
	python src/train_model.py

test:
	pytest tests/ -v

all: process train

clean:
	rm -rf data/processed/* data/final/* models/* __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache

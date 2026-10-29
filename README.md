# FM4017

This repository provides a framework for taking a dataset, preprocessing it, and comparing model performance against the energy footprint of each model.

## Overview

The project focuses on evaluating the trade-off between predictive quality and energy consumption. It includes data preprocessing utilities, example model training workflows, and scripts for benchmarking different models.

## Repository structure

- `Dataset/` – datasets used for experiments and evaluation.
- `preprocessing/` – scripts for transforming raw data into a usable formats for model training. With ElasticNet-based feature selection (`run_elasticnet_selection.py`).
- `gru/` - implementation for training and evaluating a GRU model.
- `lstm/` – implementation for training and evaluating an LSTM model.
- `presentation/` – Presentation material as plots in jupyter notebook.
- `randomforest/` - implementation for training and evaluating a Random Forest model
- `xgboost/` – implementation for training and evaluating a XGBoost model.

## Workflow

1. Acquire the required dataset.
2. Prepare and clean the data using the preprocessing pipeline.
3. Train models using the provided scripts.
4. Compare the predictive performance with the energy consumption of each model.

## Important note

The Nord Pool dataset is not publicly available and must be obtained separately before running the full workflow.

## Getting started

Clone the repository and use the scripts in the preprocessing and model folders to prepare data and train models. The project is intended to preprocess once, and use that as the foundation for all model implementations.

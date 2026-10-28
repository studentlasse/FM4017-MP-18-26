# FM4017

This repository provides a framework for taking a dataset, preprocessing it, and comparing model performance against the energy footprint of each model.

## Overview

The project focuses on evaluating the trade-off between predictive quality and energy consumption. It includes data preprocessing utilities, example model training workflows, and scripts for benchmarking different models.

## Repository structure

- `Dataset/` – datasets used for experiments and evaluation.
- `preprocessing/` – scripts for transforming raw data into a usable formats for model training. This folder also includes an example workflow for training a Random Forest Regressor (RFR) model, plus ElasticNet-based feature selection (`run_elasticnet_selection.py`).
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

## Feature selection with ElasticNet

`preprocessing/elasticnet_selection.py` provides a reusable ElasticNet feature selector, and `preprocessing/run_elasticnet_selection.py` runs it for every seasonal split on the training window of `ml_dataset.csv`.

The Elastic Net hyperparameters α and ρ are jointly optimized through expanding-window time-series cross-validation (`TimeSeriesSplit`). Candidate values of ρ span 0.05 to 1.0, while α is selected from a logarithmically spaced regularization path (100 values by default, controlled by `--n-alphas` and `--eps`).

Leakage controls:

- **Test period is never used.** Selection is masked to the training window of each split.
- **Preprocessing is fold-local.** Imputation and standardization are fitted inside each CV fold, on that fold's training rows only.
- **The validation slice is excluded.** Models hold out the last 15% of the training window, so selection defaults to `--train-ratio 0.85` and stops where model training ends.
- **Split-safe loading.** The selection file is keyed by split index, and model scripts request a split explicitly, so a selection can never be silently paired with the wrong test period.

```bash
# Every split, all three modes (default)
uv run python preprocessing/run_elasticnet_selection.py

# A single split with a curated candidate set
uv run python preprocessing/run_elasticnet_selection.py --feature-set full_set --split-index 1

# Selection on the whole training window instead of the first 85%
uv run python preprocessing/run_elasticnet_selection.py --train-ratio 1.0
```

Each split produces three feature sets from one CV run:

| Mode | `alpha_scale` | Behaviour |
|---|---:|---|
| `aggressive` | `1.0` | CV-selected α; strongest penalty, fewest features. |
| `moderate` | `0.2` | 20% of the CV-selected α; keeps more correlated features. |
| `none` | – | Every candidate feature; no selection. |

Override the scales with `--aggressive-alpha-scale` and `--moderate-alpha-scale`.

The script writes:

- `elasticnet_selected_features.json` – per-split `(α, ρ)`, CV RMSE (mean, standard deviation, and per fold), and the three feature lists.
- `elasticnet_selection_results.csv` – coefficients for every candidate feature, split, and mode.

Model scripts select a set with two constants and load it by split index:

```python
SPLIT_INDEX = 0
SELECTION_MODE = "moderate"   # "aggressive", "moderate", or "none"

from feature_sets import load_feature_set, load_feature_set_metadata

feature_columns = load_feature_set(SPLIT_INDEX, SELECTION_MODE)
metadata = load_feature_set_metadata(SPLIT_INDEX, SELECTION_MODE)
```

`load_feature_set_metadata` returns the α actually used for that mode, the CV-selected α, ρ, and CV RMSE, which the scripts print alongside their results.

## Important note

The Nord Pool dataset is not publicly available and must be obtained separately before running the full workflow.

## Getting started

Clone the repository and use the scripts in the preprocessing and model folders to prepare data and train models. The project is intended to preprocess once, and use that as the foundation for all model implementations.

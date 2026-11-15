# FM4017

This repository provides a framework for taking a dataset, preprocessing it, and comparing model performance against the energy footprint of each model.

## Overview

The project focuses on evaluating the trade-off between predictive quality and energy consumption. It includes data preprocessing utilities, example model training workflows, and scripts for benchmarking different models.

## Repository structure

- `Dataset/` – datasets used for experiments and evaluation.
- `preprocessing/` – scripts for transforming raw data into a usable formats for model training. With ElasticNet-based feature selection (`run_elasticnet_selection.py`).
- `gru/` - implementation for training and evaluating a GRU model.
- `lstm/` – implementation for training and evaluating an LSTM model.
- `randomforest/` - implementation for training and evaluating a Random Forest model
- `xgbr/` – implementation for training and evaluating an XGBoost model.
- `hyperparameter-pipeline-test/` – scripts that coordinate the full benchmark: `run_all_tuning.py` (hyperparameter search for every model/split/feature set) and `energy_aware_selection.py` (turns tuning results into performance-vs-energy comparisons).
- `logger/` – CSV logging helpers shared by all model scripts.
- `plotter/` – plotting helpers shared by all model scripts.
- `presentation/` – Presentation material as plots in jupyter notebook.
- `results/` – generated result CSVs (`git`-ignored).
- `plots/` – generated figures (`git`-ignored).

## Setup

The repository does not ship a `pyproject.toml`, so install the dependency list into a `uv`-managed virtual environment:

```bash
uv venv
uv pip install -r requirements.txt
```

All commands below assume they are run from the repository root.

## Pipeline overview

Each stage writes the files the next stage consumes. Detailed instructions for each step follow below.

| Step | Stage | Command | Produces |
| --- | --- | --- | --- |
| 1 | Provide the data | `preprocessing/entso-e_data_gathering.ipynb` (ENTSO-E), plus the Nord Pool database | `Dataset/Nordpool/thesis_database.db`, `Dataset/Entso-E/entsoe_data_hourly.csv` |
| 2 | Build the model dataset | `uv run python preprocessing/dataset_conversion.py` | `ml_dataset.csv` |
| 3 | Select features | `uv run python preprocessing/run_elasticnet_selection.py` | `elasticnet_selected_features.json`, `elasticnet_selection_results.csv` |
| 4 | Baseline model tests | `uv run python <model>/test_*.py` | `results/<model>_results.csv`, `plots/` |
| 5 | Hyperparameter tuning | `uv run python hyperparameter-pipeline-test/run_all_tuning.py` | `results/*_tuning_results.csv`, `results/*_search_results_split*.csv` |
| 6 | Energy-aware selection | `uv run python hyperparameter-pipeline-test/energy_aware_selection.py` | `results/energy_aware/*.csv`, `plots/energy_aware*.png/pdf` |

### Splits and feature sets

Every model is evaluated on three seasonal splits (the default in the scripts is split `1`):

| Index | Name | Train | Test |
| --- | --- | --- | --- |
| `0` | Entire timeline | Oct 2024 – May 2026 | Jun 2026 – Sep 2026 |
| `1` | Summer 2025 | Apr – Jun 2025 | Jul – Sep 2025 |
| `2` | Winter 2025 | Oct – Dec 2025 | Jan – Mar 2026 |

Each split has three feature sets produced by the ElasticNet stage:

- `aggressive` – CV-selected alpha, fewest features.
- `moderate` – 20 % of the CV-selected alpha, keeps more correlated features.
- `none` – all candidate features.

## Step 1 – Provide the data

Two inputs are needed. The Nord Pool database must be supplied locally; the ENTSO-E forecast file is tracked in the repository and only needs to be regenerated if the data changes.

1. **Nord Pool market data** – an SQLite database at `Dataset/Nordpool/thesis_database.db`. The database must contain the `Prices`, `Volumes`, `Flows`, `Capacities` and `BiddingZones` tables used by `preprocessing/dataset_conversion.py` (see `preprocessing/sql_query.sql` for the queries). The Nord Pool dataset is not publicly available and must be obtained separately.

2. **ENTSO-E forecasts** – generate `Dataset/Entso-E/entsoe_data_hourly.csv` with the gathering notebook:

   ```bash
   # Only needed to regenerate the forecast file
   uv pip install entsoe-py python-dotenv
   uv run --with jupyter jupyter notebook preprocessing/entso-e_data_gathering.ipynb
   ```

   Add `ENTSO_API_KEY=<your key>` to a `.env` file in the repository root, then run all cells. The notebook queries day-ahead prices and wind/solar, generation, load and reservoir forecasts from the ENTSO-E Transparency Platform, normalises everything to hourly rows, and writes `entsoe_data_hourly.csv` into its working directory — move the result to `Dataset/Entso-E/`.

   Set the notebook's `start` to `2023-10-01` so the file matches `ENTSOE_START` in `preprocessing/dataset_conversion.py`; the shipped CSV covers `2023-10-01` through `2026-10-01` (Europe/Brussels).

## Step 2 – Build the model dataset

```bash
uv run python preprocessing/dataset_conversion.py
```

This queries the SQLite database, joins flows, capacities, neighbor prices and ENTSO-E forecasts, engineers calendar and lag features, and writes `ml_dataset.csv` to the repository root. Rows before `2023-10-01` are dropped to stay within the forecast data coverage.

## Step 3 – Run ElasticNet feature selection

```bash
uv run python preprocessing/run_elasticnet_selection.py
```

Configuration is in the constants at the top of the script: `FEATURE_SET_NAME` (`"all"` uses every numeric column), `SPLIT_INDICES` (`None` runs all splits), `N_ALPHAS`, `CV_SPLITS`, and `TRAIN_RATIO`. Feature selection is fitted on the training window only (the first 85 % by default, matching the model validation split), so the test period never influences the result.

Outputs:

- `elasticnet_selected_features.json` – selected features per split and mode. **The model scripts refuse to run without it.**
- `elasticnet_selection_results.csv` – coefficients and diagnostics for each split/mode.

## Step 4 – Run baseline model tests

Each model script holds its configuration as constants near the bottom of the file: `SPLIT_INDEX`, `SELECTION_MODE` and a `parameters` dict. Edit those to test a different split, feature set or configuration, then run:

```bash
uv run python gru/test_gru_model.py
uv run python lstm/test_lstm_model.py
uv run python randomforest/test_randomforest_model.py
uv run python xgbr/test_xgboost_model.py
```

Outputs:

- one appended row per run in `results/<model>_results.csv`;
- test prediction/target CSVs in `results/`;
- prediction-vs-target figures in `plots/`;
- best checkpoints in the repository root: `best_gru_model.pt` for GRU and `best_lstm_model.pt` for LSTM.

Energy is captured with CodeCarbon's `OfflineEmissionsTracker` (`country_iso_code="NOR"`), so training, validation and test energy are reported in kWh next to RMSE/R²/MAE. Energy numbers are hardware-dependent; only compare runs made on the same machine.

## Step 5 – Hyperparameter tuning

To tune every model on every split and feature set, using both grid and random search:

```bash
uv run python hyperparameter-pipeline-test/run_all_tuning.py
```

This runs `4 models × 3 splits × 3 feature sets × 2 search methods = 72` experiments and can take many hours. To tune a single model instead, run its tuning script directly:

```bash
uv run python randomforest/test_randomforest_model_param_tuning.py
```

The single-model scripts loop over `SPLIT_INDICES` × `SELECTION_MODES` defined at the top of the file; set `search_method` (and `n_iter` for random search) at the bottom of the file. `run_all_tuning.py` sets `N_ITER_RANDOM = 100`, `RANDOM_STATE = 42`.

Outputs per run:

- `results/<model>_<method>_search_results_split<index>_<mode>.csv` – one row per configuration, with validation metrics and the measured energy of that configuration (`config_energy_kwh`);
- `results/<model>_tuning_results.csv` – one row per model/split/mode/method, containing the winning configuration retrained several times (`RETRAIN_REPEATS`), its test metrics and mean/total training and inference energy;
- `results/<model>_<method>_tuning_summary.csv` – the summary table written by a single tuning-script invocation;
- checkpoints for the recurrent models under `<model>_checkpoints/split_<index>_<mode>/`.

## Step 6 – Energy-aware model selection

```bash
uv run python hyperparameter-pipeline-test/energy_aware_selection.py
```

The script reads the tuning and per-configuration CSVs from `results/` and produces two comparisons:

1. **Measured selection** – compares the retrained best configurations across models/splits/feature sets using measured energy.
2. **Per-configuration selection** – compares every configuration inside a search, using measured `config_energy_kwh` when available and a documented complexity proxy otherwise.

Useful options:

```bash
# Custom tolerances (fractional, e.g. 1% / 2% / 5% / 10% quality loss allowed)
uv run python hyperparameter-pipeline-test/energy_aware_selection.py --tolerance 0.01 0.02 0.05

# Which energy definition drives the selection
uv run python hyperparameter-pipeline-test/energy_aware_selection.py --energy-basis deployment
#   experiment | deployment | search | training | inference

# Selection metric and IO locations
uv run python hyperparameter-pipeline-test/energy_aware_selection.py --performance test_rmse
uv run python hyperparameter-pipeline-test/energy_aware_selection.py --results-dir results --output-dir results/energy_aware --plot-dir plots

# Skip plotting (matplotlib is only imported when plots are generated)
uv run python hyperparameter-pipeline-test/energy_aware_selection.py --no-plot
```

Outputs:

- `results/energy_aware/measured_experiment_rows.csv` – one cleaned row per experiment;
- `results/energy_aware/measured_selection.csv` – cheapest model within each tolerance;
- `results/energy_aware/measured_pareto.csv` – measured Pareto front;
- `results/energy_aware/proxy_config_selection.csv` and `config_comparison.csv` – per-configuration and normal-best vs energy-optimized comparisons;
- `plots/energy_aware_pareto_by_split.{png,pdf}`, `plots/energy_aware_pareto_overall.{png,pdf}`, `plots/energy_aware_per_config_pareto*.png` – report figures. The measured Pareto plots are restricted to the summer split (index `1`).

## Important note

The Nord Pool dataset is not publicly available and must be obtained separately before running the full workflow. All generated artifacts (`ml_dataset.csv`, `elasticnet_selected_features.json`, `results/`, `plots/`, checkpoints) are ignored by `git`; archive them separately if the results need to be reproduced or shared.

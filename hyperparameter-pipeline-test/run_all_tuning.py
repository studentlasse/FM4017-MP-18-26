#!/usr/bin/env python
"""
Unified script to run hyperparameter tuning for all 4 models:
- GRU
- LSTM
- Random Forest
- XGBoost

Runs both grid search and random search for all seasonal splits and feature selection modes.
"""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import run_experiment from all 4 models
from gru.test_gru_model_param_tuning import run_experiment as run_gru_experiment
from lstm.test_lstm_model_param_tuning import run_experiment as run_lstm_experiment
from randomforest.test_randomforest_model_param_tuning import run_experiment as run_rf_experiment
from xgbr.test_xgboost_model_param_tuning import run_experiment as run_xgb_experiment

from feature_sets import SELECTION_MODES

# Configuration
SPLIT_INDICES = (0, 1, 2)
SEARCH_METHODS = ("grid", "random")
N_ITER_RANDOM = 100 # same as grid search 
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configurations: (name, run_function, needs_device)
MODELS = [
    ("GRU", run_gru_experiment, True),
    ("LSTM", run_lstm_experiment, True),
    ("RandomForest", run_rf_experiment, False),
    ("XGBoost", run_xgb_experiment, False),
]


def run_model_tuning(model_name, run_fn, needs_device, search_method, split_index, selection_mode):
    """Run tuning for a single model/split/mode combination."""
    print(
        f"\n{'='*60}"
        f"\n=== {model_name} | Split {split_index} | {selection_mode} | {search_method.upper()} ==="
        f"\n{'='*60}",
        flush=True,
    )
    
    try:
        if needs_device:
            run_fn(
                split_index=split_index,
                selection_mode=selection_mode,
                device=DEVICE,
                search_method=search_method,
                n_iter=N_ITER_RANDOM,
                random_state=RANDOM_STATE,
            )
        else:
            run_fn(
                split_index=split_index,
                selection_mode=selection_mode,
                search_method=search_method,
                n_iter=N_ITER_RANDOM,
                random_state=RANDOM_STATE,
            )
        print(f"✓ {model_name} completed successfully")
        return True
    except Exception as e:
        print(f"✗ {model_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print(f"Device: {DEVICE}")
    print(f"Splits: {SPLIT_INDICES}")
    print(f"Selection modes: {SELECTION_MODES}")
    print(f"Search methods: {SEARCH_METHODS}")
    print(f"Random search iterations: {N_ITER_RANDOM}")
    print(f"Random state: {RANDOM_STATE}")
    print(f"Models: {[m[0] for m in MODELS]}")
    
    total_runs = len(MODELS) * len(SPLIT_INDICES) * len(SELECTION_MODES) * len(SEARCH_METHODS)
    print(f"\nTotal experiment runs: {total_runs}")
    
    results = {}
    completed = 0
    failed = 0
    
    for search_method in SEARCH_METHODS:
        for model_name, run_fn, needs_device in MODELS:
            for split_index in SPLIT_INDICES:
                for selection_mode in SELECTION_MODES:
                    completed += 1
                    key = f"{model_name}_split{split_index}_{selection_mode}_{search_method}"
                    
                    success = run_model_tuning(
                        model_name, run_fn, needs_device,
                        search_method, split_index, selection_mode
                    )
                    
                    results[key] = "success" if success else "failed"
                    if success:
                        failed += 0
                    else:
                        failed += 1
                    
                    print(f"\nProgress: {completed}/{total_runs} ({failed} failed)")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for key, status in results.items():
        print(f"  {key}: {status}")
    
    success_count = sum(1 for v in results.values() if v == "success")
    print(f"\nTotal: {success_count}/{len(results)} succeeded")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
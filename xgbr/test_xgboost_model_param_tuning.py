import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from codecarbon import OfflineEmissionsTracker
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from logger.csv_logger import write_to_csv, save_plot_csv
from feature_sets import (
    SELECTION_MODES,
    load_feature_set,
    load_feature_set_metadata,
)
from plotter.result_plotter import plot_results


from itertools import product
import random


DATASET_PATH = PROJECT_ROOT / "ml_dataset.csv"
RESULTS_CSV_PATH = PROJECT_ROOT / "results" / "xgboost_tuning_results.csv"
RETRAIN_REPEATS = 5
RETRAIN_SEED_BASE = 42

# Seasonal splits and feature selection modes to evaluate. Every combination of
# SPLIT_INDICES x SELECTION_MODES is tuned in a separate search run. The feature
# set is read from elasticnet_selected_features.json for each split and mode.
SPLIT_INDICES = (0, 1, 2)
VALIDATION_FRACTION = 0.15
TRAIN_FRACTION = 1.0 - VALIDATION_FRACTION

SEARCH_SPACE = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.85],
    "colsample_bytree": [0.7, 0.85],
    "min_child_weight": [2, 3],
    "reg_lambda": [10.0],
    "early_stopping_rounds": [20],
}

def load_dataset(file_path=DATASET_PATH, split_index=0, selection_mode="moderate"):
    data = pd.read_csv(file_path)

    data["delivery_day"] = pd.to_datetime(data["delivery_day"])

    # Convert hourly observations to one row per day with the average price for that day
    data = (
        data.groupby("delivery_day", as_index=False)
        .mean(numeric_only=True)
        .sort_values("delivery_day")
        .reset_index(drop=True)
    )

    seasonal_splits = [
        (
            "Entire timeline (train: Oct 2024-May 2026, test: Jun 2026-Sep 2026)",
            data["delivery_day"].between("2026-06-01", "2026-09-30"),  # test
            data["delivery_day"].between("2024-10-01", "2026-05-31"),  # train
        ),
        
        (
            "Summer 2025 (train: Apr-Jun 2025, test: Jul-Sep 2025)",
            data["delivery_day"].between("2025-07-01", "2025-09-30"),  # test
            data["delivery_day"].between("2025-04-01", "2025-06-30"),  # train
        ),
        (
            "Winter 2025 (train: Oct-Dec 2025, test: Jan-Mar 2026)",
            data["delivery_day"].between("2026-01-01", "2026-03-31"),  # test
            data["delivery_day"].between("2025-10-01", "2025-12-31"),  # train
        ),
        ]
        
    
    data_filter, test_mask, train_mask = seasonal_splits[split_index]

    data = data.sort_values(["delivery_day", "hour"]).reset_index(drop=True)

    forecast_horizon = 1

    data["target_price_next_day"] = data["price_value"].shift(-forecast_horizon)
    # Predict the price change instead of the price level. The current price
    # (anchor) is added back during evaluation.
    data["target_price_delta"] = (
        data["target_price_next_day"] - data["price_value"]
    )

    data = data.dropna(subset=["target_price_next_day"])    

    delivery_days = data["delivery_day"].copy()

    drop_cols = [
        "delivery_day",
    ]

    data = data.drop(columns=drop_cols)

    # Feature set for this run's split and selection mode, built by
    # preprocessing/run_elasticnet_selection.py.
    feature_set_name = selection_mode
    feature_columns = load_feature_set(split_index, selection_mode)
    selection_metadata = load_feature_set_metadata(split_index, selection_mode)

    return (
        data,
        feature_columns,
        train_mask,
        test_mask,
        data_filter,
        forecast_horizon,
        feature_set_name,
        delivery_days,
        selection_metadata,
    )

def measure_energy(operation, *args, **kwargs):
    tracker = OfflineEmissionsTracker(
        country_iso_code="NOR",
        save_to_file=False,
        log_level="error",
    )
    tracker.start()
    try:
        result = operation(*args, **kwargs)
    finally:
        tracker.stop()

    energy_kwh = tracker.final_emissions_data.energy_consumed
    return result, energy_kwh

def train_model(model, X_train, y_train, X_validation, y_validation):
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )
    return model

def predict(model, data):
    return model.predict(data)

def evaluate_model(model, data, targets, anchors):

    predictions = model.predict(data)
    anchors = np.asarray(anchors)

    # Convert predicted and actual deltas back to prices using the anchor.
    predictions = predictions + anchors
    targets = targets + anchors

    rmse = root_mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)

    return rmse, r2, mae

def train_and_score_model(
    model_factory,
    config,
    X_train,
    y_train,
    X_validation,
    y_validation,
    validation_anchors,
):

    model = model_factory(config)
    model = train_model(
        model,
        X_train,
        y_train,
        X_validation,
        y_validation
    )

    validation_rmse, validation_r2, validation_mae = evaluate_model(
        model,
        X_validation,
        y_validation,
        validation_anchors,
    )

    return {
        **config,
        "validation_r2": validation_r2,
        "validation_rmse": validation_rmse,
        "validation_mae": validation_mae,
        "model": model,
    }


def train_and_evaluate_repeated(
    config,
    train_data,
    train_targets,
    validation_data,
    validation_targets,
    validation_anchors,
    test_data,
    test_targets,
    test_anchors,
    feature_scaler,
    feature_columns,
    data_filter,
    forecast_horizon,
    checkpoint_dir,
    repeats=RETRAIN_REPEATS,
    seed_base=RETRAIN_SEED_BASE,
):
    """Retrain the best configuration several times with different seeds.

    The repeated runs measure run-to-run variance from model initialization and
    batch shuffling. The model with the lowest validation RMSE is kept for the
    final test predictions.
    """
    validation_metrics = []
    test_metrics = []
    training_energies = []
    test_energies = []
    best_run = None

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for run_index in range(repeats):
        seed = seed_base + run_index
        np.random.seed(seed)

        model = make_xgboost(config)

        checkpoint_path = (
            checkpoint_dir / f"retrained_best_model_run{run_index}.pt"
        )

        _, run_training_energy = measure_energy(
            train_model,
            model,
            train_data,
            train_targets,
            validation_data,
            validation_targets,
        )

        validation_score = evaluate_model(
            model,
            validation_data,
            validation_targets,
            validation_anchors,
        )
        test_score, run_test_energy = measure_energy(
            evaluate_model,
            model,
            test_data,
            test_targets,
            test_anchors,
        )

        validation_metrics.append(validation_score)
        test_metrics.append(test_score)
        training_energies.append(run_training_energy)
        test_energies.append(run_test_energy)

        if best_run is None or validation_score[0] < best_run["validation_rmse"]:
            best_run = {
                "model": model,
                "validation_rmse": validation_score[0],
                "run_index": run_index,
                "seed": seed,
            }

    test_metrics = np.array(test_metrics)
    test_metrics_std = (
        test_metrics.std(axis=0, ddof=1)
        if repeats > 1
        else np.zeros(test_metrics.shape[1])
    )

    return {
        "validation_metrics": np.array(validation_metrics).mean(axis=0),
        "test_metrics": test_metrics.mean(axis=0),
        "test_metrics_std": test_metrics_std,
        "training_energy_kwh_per_run": float(np.mean(training_energies)),
        "training_energy_kwh_total": float(np.sum(training_energies)),
        "test_energy_kwh_per_run": float(np.mean(test_energies)),
        "test_energy_kwh_total": float(np.sum(test_energies)),
        "best_model": best_run["model"],
        "best_run_index": best_run["run_index"],
        "best_run_seed": best_run["seed"],
    }


def run_search(
    model_factory,
    search_space,
    X_train,
    y_train,
    X_validation,
    y_validation,
    validation_anchors,
    method="grid",
    n_iter=20,
    random_state=42,
):
    keys = list(search_space)
    all_configs = [
        dict(zip(keys, values))
        for values in product(*(search_space[key] for key in keys))
    ]

    if method == "grid":
        configs = all_configs
    elif method == "random":
        rng = random.Random(random_state)
        configs = rng.sample(all_configs, k=min(n_iter, len(all_configs)))
    else:
        raise ValueError(
            f"Unsupported search method: {method!r}. "
            "Use 'grid' or 'random'."
        )

    results = []
    fitted_models = []
    config_energies = []

    for index, config in enumerate(configs, start=1):

        result, config_energy_kwh = measure_energy(
            train_and_score_model,
            model_factory=model_factory,
            config=config,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            validation_anchors=validation_anchors,
        )
        result["config_energy_kwh"] = float(config_energy_kwh)
        config_energies.append(float(config_energy_kwh))

        fitted_models.append(result.pop("model"))
        results.append({**result, "search_method": method})


    if not results:
        raise ValueError("Search produced no configurations to evaluate.")

    results_df = (
        pd.DataFrame(results)
        .sort_values("validation_rmse")
        .reset_index(drop=True)
    )

    return results_df, fitted_models, float(np.sum(config_energies))


def make_xgboost(config):
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        min_child_weight=float(config["min_child_weight"]),
        reg_lambda=float(config["reg_lambda"]),
        random_state=42,
        early_stopping_rounds=int(config["early_stopping_rounds"]),
        n_jobs=-1,
    )


def run_experiment(split_index, selection_mode, search_method="grid", n_iter=100, random_state=42):
    (
        data,
        feature_columns,
        train_mask,
        test_mask,
        data_filter,
        forecast_horizon,
        feature_set_name,
        delivery_days,
        selection_metadata,
    ) = load_dataset(
        split_index=split_index,
        selection_mode=selection_mode,
    )


    feature_data = data[feature_columns]
    target_data = data['target_price_delta']
    anchor_data = data['price_value']


    train_data_all = feature_data.loc[train_mask]
    train_targets_all = target_data.loc[train_mask]
    train_anchors_all = anchor_data.loc[train_mask]

    test_data = feature_data.loc[test_mask]
    test_targets = target_data.loc[test_mask]
    test_anchors = anchor_data.loc[test_mask]
    test_delivery_days = delivery_days.iloc[test_mask]

    validation_end = int(len(train_data_all) * TRAIN_FRACTION)

    train_data = train_data_all.iloc[:validation_end]
    train_targets = train_targets_all.iloc[:validation_end]
    train_anchors = train_anchors_all.iloc[:validation_end]

    validation_data = train_data_all.iloc[validation_end:]
    validation_targets = train_targets_all.iloc[validation_end:]
    validation_anchors = train_anchors_all.iloc[validation_end:]


    search_space = SEARCH_SPACE

    if search_method not in {"grid", "random"}:
        raise ValueError(
            f"Unsupported search method: {search_method!r}. "
            "Use 'grid' or 'random'."
        )

    checkpoint_dir = (
        PROJECT_ROOT
        / "xgboost_checkpoints"
        / f"split_{split_index}_{selection_mode}"
    )

    results, _, search_energy_kwh = run_search(
        model_factory=make_xgboost,
        search_space=search_space,
        X_train=train_data,
        y_train=train_targets,
        X_validation=validation_data,
        y_validation=validation_targets,
        validation_anchors=validation_anchors,
        method=search_method,
        n_iter=n_iter,
        random_state=random_state,
    )

    results["search_method"] = search_method
    results["tuning_energy_kwh"] = search_energy_kwh

    results_path = (
        PROJECT_ROOT
        / "results"
        / f"xgboost_{search_method}_search_results_split{split_index}_"
        f"{selection_mode}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)

    best_result = results.iloc[0]
    best_config = best_result.to_dict()

    repeated_results = train_and_evaluate_repeated(
        config=best_config,
        train_data=train_data,
        train_targets=train_targets,
        validation_data=validation_data,
        validation_targets=validation_targets,
        validation_anchors=validation_anchors,
        test_data=test_data,
        test_targets=test_targets,
        test_anchors=test_anchors,
        feature_scaler=None,  # XGBoost doesn't use a feature scaler
        feature_columns=feature_columns,
        data_filter=data_filter,
        forecast_horizon=forecast_horizon,
        checkpoint_dir=checkpoint_dir,
        repeats=RETRAIN_REPEATS,
        seed_base=RETRAIN_SEED_BASE,
    )

    best_model = repeated_results["best_model"]
    validation_score = repeated_results["validation_metrics"]
    test_metrics = repeated_results["test_metrics"]
    test_metrics_std = repeated_results["test_metrics_std"]
    training_energy_kwh = repeated_results["training_energy_kwh_per_run"]
    training_energy_kwh_total = repeated_results["training_energy_kwh_total"]
    test_energy_kwh_total = repeated_results["test_energy_kwh_total"]
    test_energy_kwh_per_run = repeated_results["test_energy_kwh_per_run"]

    importance = pd.DataFrame(
        {
            "feature": feature_data.columns,
            "importance": best_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    results_df = pd.DataFrame({
        "split_index": split_index,
        "split_name": data_filter,
        "selection_mode": selection_mode,
        "data_filter": data_filter,
        "forecast_horizon": [forecast_horizon],
        "feature_count": [len(feature_columns)],
        "search_method": search_method,
        "n_configurations": len(results),
        "feature_columns": [feature_set_name],
        "validation_r2": [validation_score[1]],
        "validation_rmse": [validation_score[0]],
        "validation_mae": [validation_score[2]],
        "test_r2": [test_metrics[1]],
        "test_rmse": [test_metrics[0]],
        "test_mae": [test_metrics[2]],
        "test_r2_std": [test_metrics_std[0]],
        "test_rmse_std": [test_metrics_std[1]],
        "test_mae_std": [test_metrics_std[2]],
        "training_energy_kwh": [training_energy_kwh],
        "training_energy_kwh_total": [training_energy_kwh_total],
        "training_energy_kwh_per_run": [training_energy_kwh],
        "test_energy_kwh_total": [test_energy_kwh_total],
        "test_energy_kwh_per_run": [test_energy_kwh_per_run],
        "run_repeats": [RETRAIN_REPEATS],
        "best_run_index": [repeated_results["best_run_index"]],
        "best_run_seed": [repeated_results["best_run_seed"]],
        "search_energy_kwh": [search_energy_kwh],
        "best_config_energy_kwh": [best_config.get("config_energy_kwh")],
        "description": (
            "The best configuration is retrained run_repeats times with "
            "different seeds; test metrics are the mean and std across those "
            "runs. training_energy_kwh is the mean per training run and "
            "training_energy_kwh_total is the sum across runs. Test energy is "
            "measured once per trained model and summed across runs."
        ),
        "best_param_n_estimators": best_config["n_estimators"],
        "best_param_max_depth": best_config["max_depth"],
        "best_param_learning_rate": best_config["learning_rate"],
        "best_param_subsample": best_config["subsample"],
        "best_param_colsample_bytree": best_config["colsample_bytree"],
        "best_param_min_child_weight": best_config["min_child_weight"],
        "best_param_reg_lambda": best_config["reg_lambda"],
        "early_stopping_rounds": best_config["early_stopping_rounds"],
    })
    RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_to_csv(results_df, RESULTS_CSV_PATH)

    test_pred = best_model.predict(test_data) + test_anchors.to_numpy()
    test_targets_plot = test_targets.to_numpy() + test_anchors.to_numpy()

    plot_data = pd.DataFrame({
        "delivery_day": pd.to_datetime(test_delivery_days).to_numpy(),
        "test_prediction": test_pred,
        "test_target": test_targets_plot,
    })

    split_name = f"split_{split_index}"
    feature_info = f"{selection_mode}_{len(feature_columns)}features"

    plot_data_description = (f"_xgboost_tuning_{split_name}_{feature_info}")

    save_plot_csv(plot_data, plot_data_description)

    plot_results( 
            test_delivery_days, 
            test_pred, 
            test_targets_plot, 
            test_pred_label="Test predictions", 
            test_target_label="Test targets", 
            title=f"XGBoost predictions vs. targets, {data_filter}", 
            y_label="Target price",
            filename_suffix=(
                f"xgboost_test_pred_vs_targets_{split_name}_{selection_mode}"
            ),
    )

    return results_df


if __name__ == "__main__":
    search_method = "random"  # "grid" or "random"
    n_iter = 100  # for random search
    random_state = 42

    all_results = []

    for split_index in SPLIT_INDICES:
        for selection_mode in SELECTION_MODES:
            print(
                f"\n=== Tuning split {split_index} "
                f"({selection_mode} feature set) ===",
                flush=True,
            )
            all_results.append(
                run_experiment(
                    split_index=split_index,
                    selection_mode=selection_mode,
                    search_method=search_method,
                    n_iter=n_iter,
                    random_state=random_state,
                )
            )

    summary = pd.concat(all_results, ignore_index=True)
    summary_path = PROJECT_ROOT / "results" / f"xgboost_{search_method}_tuning_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)


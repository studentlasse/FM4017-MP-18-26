import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from codecarbon import OfflineEmissionsTracker
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from logger.csv_logger import write_to_csv
from feature_sets import FEATURE_SETS
from plotter.result_plotter import plot_results

from itertools import product
import random


DATASET_PATH = PROJECT_ROOT / "ml_dataset.csv"
RESULTS_CSV_PATH = PROJECT_ROOT / "rf_search_results.csv"

def load_dataset(file_path=DATASET_PATH):
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
        
    
    data_filter, test_mask, train_mask = seasonal_splits[0]

    data = data.sort_values(["delivery_day", "hour"]).reset_index(drop=True)

    forecast_horizon = 1

    data["target_price_next_day"] = data["price_value"].shift(-forecast_horizon)

    data = data.dropna(subset=["target_price_next_day"])    

    delivery_days = data["delivery_day"].copy()

    drop_cols = [
        "delivery_day",
    ]

    data = data.drop(columns=drop_cols)

    FEATURE_SET_NAME = "full_set"

    feature_columns = FEATURE_SETS[FEATURE_SET_NAME]

    return data, feature_columns, train_mask, test_mask, data_filter, forecast_horizon, FEATURE_SET_NAME, delivery_days

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
    )
    return model

def predict(model, data):
    return model.predict(data)

def evaluate_model(model, data, targets):

    predictions = model.predict(data)

    
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
    )

    return {
        **config,
        "validation_r2": validation_r2,
        "validation_rmse": validation_rmse,
        "validation_mae": validation_mae,
        "model": model,
    }

def run_search(
    model_factory,
    search_space,
    X_train,
    y_train,
    X_validation,
    y_validation,
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

    for index, config in enumerate(configs, start=1):
        print(
            f"{method.capitalize()} search "
            f"{index}/{len(configs)}: {config}"
        )

        result = train_and_score_model(
            model_factory=model_factory,
            config=config,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
        )

        fitted_models.append(result.pop("model"))
        results.append({**result, "search_method": method})

        print(f"Validation RMSE: {result['validation_rmse']:.4f}")

    if not results:
        raise ValueError("Search produced no configurations to evaluate.")

    results_df = (
        pd.DataFrame(results)
        .sort_values("validation_rmse")
        .reset_index(drop=True)
    )

    return results_df, fitted_models


def make_random_forest(config):
    return RandomForestRegressor(
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]),
        min_samples_split=int(config["min_samples_split"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_features=config["max_features"],
        random_state=42,
        n_jobs=-1,
    )


if __name__ == "__main__":
    data, feature_columns, train_mask, test_mask, data_filter, forecast_horizon, feature_set_name, delivery_days = load_dataset()

    feature_data = data[feature_columns]
    target_data = data['target_price_next_day']


    train_data_all = feature_data.loc[train_mask]
    train_targets_all = target_data.loc[train_mask]

    test_data = feature_data.loc[test_mask]
    test_targets = target_data.loc[test_mask]
    test_delivery_days = delivery_days.iloc[test_mask]

    validation_end = int(len(train_data_all) * 0.85)

    train_data = train_data_all.iloc[:validation_end]
    train_targets = train_targets_all.iloc[:validation_end]

    validation_data = train_data_all.iloc[validation_end:]
    validation_targets = train_targets_all.iloc[validation_end:]


    search_space = {
        "n_estimators": [100, 300, 500],
        "max_depth": [2, 5, 7],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [1.0, 0.7],
        }

    #search_method = "grid"
    search_method = "random"

    if search_method == "random":
        (results, rf_models), search_energy_kwh = measure_energy(
        run_search,
        model_factory=make_random_forest,
        search_space=search_space,
        X_train=train_data,
        y_train=train_targets,
        X_validation=validation_data,
        y_validation=validation_targets,
        method="random",
        n_iter=20,
        random_state=42,
        )

        print(f"Total grid-search energy: {search_energy_kwh:.9f} kWh")
        results["tuning_energy_kwh"] = search_energy_kwh
        results.to_csv("rf_random_search_results.csv", index=False)
    

    if search_method == "grid":
        (results, xgb_models), search_energy_kwh = measure_energy(
            run_search,
            model_factory=make_random_forest,
            search_space=search_space,
            X_train=train_data,
            y_train=train_targets,
            X_validation=validation_data,
            y_validation=validation_targets,
            method="grid",
        )

        print(f"Total grid-search energy: {search_energy_kwh:.9f} kWh")
        results["tuning_energy_kwh"] = search_energy_kwh
        results.to_csv("rf_grid_search_results.csv", index=False)


    best_config = results.iloc[0].to_dict()
    
    best_model = make_random_forest(best_config)
    best_model = train_model(
        best_model,
        train_data,
        train_targets,
        validation_data,
        validation_targets,
    )

    validation_score, validation_energy_kwh = measure_energy(
        evaluate_model, 
        best_model,
        validation_data,
        validation_targets,
    )
    
    test_metrics, test_energy_kwh = measure_energy(
        evaluate_model,
        best_model,
        test_data,
        test_targets,
    )

    importance = pd.DataFrame(
        {
            "feature": feature_data.columns,
            "importance": best_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    print(importance.head(20))


    print(f"Test R²: {test_metrics[1]:.4f}")
    print(f"Test RMSE: {test_metrics[0]:.4f}")
    print(f"Test MAE: {test_metrics[2]:.4f}")
    print(f"Test prediction energy consumption: {test_energy_kwh:.9f} kWh")

    results_df = pd.DataFrame({
        "data_filter": data_filter,
        "forecast_horizon": [forecast_horizon],
        "feature_count": [len(feature_columns)],
        "search_method": search_method,
        "n_configurations": len(results),
        "feature_columns": [feature_set_name],
        "validation_r2": [validation_score[0]],
        "validation_rmse": [validation_score[1]],
        "validation_mae": [validation_score[2]],
        "test_r2": [test_metrics[0]],
        "test_rmse": [test_metrics[1]],
        "test_mae": [test_metrics[2]],
        "search_energy_kwh": [search_energy_kwh],
        "validation_energy_kwh": [validation_energy_kwh],
        "test_energy_kwh": [test_energy_kwh],
        "description": f"Best model from {search_method} search",
    })
    write_to_csv(results_df, RESULTS_CSV_PATH)


    test_pred = best_model.predict(test_data)
   
    plot_results( 
        test_delivery_days, 
        test_pred, 
        test_targets, 
        test_pred_label="Test predictions", 
        test_target_label="Test targets", 
        title=f"Random Forest predictions vs. targets, {data_filter}", 
        y_label="Target price",
        filename_suffix = "rf_test_pred_vs_targets"
    )


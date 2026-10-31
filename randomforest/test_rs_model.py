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
from feature_sets import load_feature_set, load_feature_set_metadata
from plotter.result_plotter import plot_results

DATASET_PATH = PROJECT_ROOT / "ml_dataset.csv"
RESULTS_CSV_PATH = PROJECT_ROOT / "results/rf_results.csv"

# Seasonal split and feature selection mode to evaluate. The feature set is
# read from elasticnet_selected_features.json for this exact split and mode.
SPLIT_INDEX = 0
SELECTION_MODE = "moderate"  # "aggressive", "moderate", or "none"

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
        
    
    data_filter, test_mask, train_mask = seasonal_splits[SPLIT_INDEX]

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
    feature_set_name = SELECTION_MODE
    feature_columns = load_feature_set(SPLIT_INDEX, SELECTION_MODE)
    selection_metadata = load_feature_set_metadata(SPLIT_INDEX, SELECTION_MODE)

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

def repeated_test_evaluation(
    model,
    data,
    targets,
    anchors,
    repeats=10,
):
    metrics = []

    for _ in range(repeats):
        metrics.append(
            evaluate_model(
                model,
                data,
                targets,
                anchors,
            )
        )

    metrics = np.array(metrics)
    return metrics.mean(axis=0), metrics.std(axis=0)

def train_and_validate_model(
    model,
    train_data,
    train_targets,
    validation_data,
    validation_targets,
    validation_anchors,
):
    model = train_model(
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
    return model, validation_score

if __name__ == "__main__":
    (data, feature_columns, train_mask, test_mask, data_filter, forecast_horizon,
     feature_set_name, delivery_days, selection_metadata) = load_dataset()


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

    validation_end = int(len(train_data_all) * 0.85)

    train_data = train_data_all.iloc[:validation_end]
    train_targets = train_targets_all.iloc[:validation_end]
    train_anchors = train_anchors_all.iloc[:validation_end]

    validation_data = train_data_all.iloc[validation_end:]
    validation_targets = train_targets_all.iloc[validation_end:]
    validation_anchors = train_anchors_all.iloc[validation_end:]

    parameters = {
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1
    }

    model = RandomForestRegressor(
        n_estimators = parameters["n_estimators"],
        random_state = parameters["random_state"],
        n_jobs = parameters["n_jobs"],
    )

    (model, validation_score), training_energy_kwh = measure_energy(
        train_and_validate_model,
        model,
        train_data,
        train_targets,
        validation_data,
        validation_targets,
        validation_anchors,
    )

    (test_metrics, test_metrics_std), test_energy_kwh = measure_energy(
        repeated_test_evaluation,
        model,
        test_data,
        test_targets,
        test_anchors,
        repeats=10,
    )

    importance = pd.DataFrame(
        {
            "feature": feature_data.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)




    results_df = pd.DataFrame({
        "data_filter": data_filter,
        "forecast_horizon": [forecast_horizon],
        "feature_count": [len(feature_columns)],
        "feature_columns": [feature_set_name],
        "validation_r2": [validation_score[1]],
        "validation_rmse": [validation_score[0]],
        "validation_mae": [validation_score[2]],
        "test_r2": [test_metrics[1]],
        "test_rmse": [test_metrics[0]],
        "test_mae": [test_metrics[2]],
        "training_energy_kwh": [training_energy_kwh],
        "test_energy_kwh_10_repeats": [test_energy_kwh],
        "description": None,
    })

    parameters_df = pd.DataFrame([
    {f"param_{name}": value for name, value in parameters.items()}
    ])
    results_df = pd.concat([results_df, parameters_df], axis=1)

    write_to_csv(results_df, RESULTS_CSV_PATH)


    test_pred = model.predict(test_data) + test_anchors.to_numpy()
    test_targets_plot = test_targets.to_numpy() + test_anchors.to_numpy()

    plot_results( 
            test_delivery_days, 
            test_pred, 
            test_targets_plot, 
            test_pred_label="Test predictions", 
            test_target_label="Test targets", 
            title=f"Random Forest predictions vs. targets, {data_filter}", 
            y_label="Target price",
            filename_suffix = "rf_test_pred_vs_targets"
            )

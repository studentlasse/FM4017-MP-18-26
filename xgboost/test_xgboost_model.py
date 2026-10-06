import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from codecarbon import OfflineEmissionsTracker
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, root_mean_squared_error 
from logger.csv_logger import write_to_csv
from feature_sets import FEATURE_SETS

DATASET_PATH = PROJECT_ROOT / "ml_dataset.csv"
RESULTS_CSV_PATH = PROJECT_ROOT / "xgboost_results.csv"

def load_dataset(file_path=DATASET_PATH):
    data = pd.read_csv(file_path)

    data["delivery_day"] = pd.to_datetime(data["delivery_day"])

    # Define a date filter:
    #filter_end_date = pd.Timestamp("2020-01-01")
    # Use the entire timeline:
    filter_end_date = None

    if filter_end_date is not None:
        data = data[data["delivery_day"] < filter_end_date].copy()

        data_filter = f"delivery_day < {filter_end_date.date()}"

    else:
        data_filter = "entire timeline"

    data = data.sort_values(["delivery_day", "hour"]).reset_index(drop=True)

    forecast_horizon = 24

    data["target_price_next_day"] = data["price_value"].shift(-forecast_horizon)

    data = data.dropna(subset=["target_price_next_day"])       

    drop_cols = [
        "delivery_day",
    ]

    data = data.drop(columns=drop_cols)

    FEATURE_SET_NAME = "full_set"

    feature_columns = FEATURE_SETS[FEATURE_SET_NAME]

    return data, feature_columns, data_filter, forecast_horizon, FEATURE_SET_NAME

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
        verbose=True,
    )
    return model

def predict(model, data):
    return model.predict(data)

def evaluate_model(model, data, targets):

    predictions = model.predict(data)

    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)

    return mse, rmse, r2

if __name__ == "__main__":
    data, feature_columns, data_filter, forecast_horizon, feature_set_name = load_dataset()

    feature_data = data[feature_columns]
    target_data = data['target_price_next_day']
    train_end = int(len(feature_data) * 0.7)
    val_end = int(len(feature_data) * 0.85)

    train_data = feature_data.iloc[:train_end]
    train_targets = target_data.iloc[:train_end]

    validation_data = feature_data.iloc[train_end:val_end]
    validation_targets = target_data.iloc[train_end:val_end]

    test_data = feature_data.iloc[val_end:]
    test_targets = target_data.iloc[val_end:]


    model = XGBRegressor(
        n_estimators=300,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.6,
        random_state=42,
        n_jobs=-1,
    )

    _, training_energy_kwh = measure_energy(
        train_model,
        model,
        train_data,
        train_targets,
        validation_data,
        validation_targets,
    )

    validation_score, validation_energy_kwh = measure_energy(
        evaluate_model,
        model,
        validation_data,
        validation_targets,
    )

    test_metrics, test_energy_kwh = measure_energy(
        evaluate_model,
        model,
        test_data,
        test_targets
    )

    importance = pd.DataFrame(
        {
            "feature": feature_data.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    print(importance.head(20))

    print(f"Validation R²: {validation_score[2]:.4f}")
    print(f"Validation RMSE: {validation_score[1]:.4f}")
    print(f"Test R²: {test_metrics[2]:.4f}")
    print(f"Test RMSE: {test_metrics[1]:.4f}")
    print(f"Training energy consumption: {training_energy_kwh:.9f} kWh")
    print(f"Validation energy consumption: {validation_energy_kwh:.9f} kWh")
    print(f"Test prediction energy consumption: {test_energy_kwh:.9f} kWh")

    results_df = pd.DataFrame({
        "data_filter": data_filter,
        "forecast_horizon": [forecast_horizon],
        "feature_count": [len(feature_columns)],
        "feature_columns": [feature_set_name],
        "validation_r2": [validation_score[0]],
        "validation_rmse": [validation_score[1]],
        "test_r2": [test_metrics[0]],
        "test_rmse": [test_metrics[1]],
        "training_energy_kwh": [training_energy_kwh],
        "validation_energy_kwh": [validation_energy_kwh],
        "test_energy_kwh": [test_energy_kwh],
        "description": None,
    })

    write_to_csv(results_df, RESULTS_CSV_PATH)
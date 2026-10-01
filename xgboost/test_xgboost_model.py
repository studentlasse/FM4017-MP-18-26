
import pandas as pd
import numpy as np
from codecarbon import OfflineEmissionsTracker

from xgboost_model import XGBoostModel

DATASET_PATH = "ml_dataset.csv" # from preprocessing/dataset_conversion.py


def load_dataset(file_path=DATASET_PATH):
    data = pd.read_csv(file_path)


    data["delivery_day"] = pd.to_datetime(data["delivery_day"])
    data = data.sort_values(["delivery_day", "hour"]).reset_index(drop=True)

    data = data.dropna(subset=['price_value'])       

    drop_cols = [
        "delivery_day",
    ]

    data = data.drop(columns=drop_cols)

    feature_columns = [
        'hour', 
        'buy_volume_value', 
        'sell_volume_value',
        'net_volume', 
        'flow_DK1_export', 
        'flow_DK1_import', 
        'flow_GER_export',
        'flow_GER_import', 
        'flow_NL_export', 
        'flow_NL_import',
        'flow_NO1_export', 
        'flow_NO1_import', 
        'flow_NO5_export',
        'flow_NO5_import', 
        'capacity_GER', 
        'capacity_NL', 
        'capacity_NO1',
        'capacity_NO5', 
        'util_GER', 
        'util_NL', 
        'util_NO1', 
        'util_NO5',
        'capacity_missing_DK1',
        'capacity_missing_GER', 
        'capacity_missing_NL',
        'capacity_missing_NO1', 
        'capacity_missing_NO5', 
        'capacity_zero_DK1',
        'capacity_zero_GER', 
        'capacity_zero_NL', 
        'capacity_zero_NO1',
        'capacity_zero_NO5', 
        'price_DK1', 
        'price_NO1', 
        'price_NO5',
        'total_exports', 
        'total_imports', 
        'net_flow', 
        'day_of_week', 
        'hour_sin',
        'hour_cos', 
        'month', 
        'day_of_year', 
        'price_lag_24', 
        'price_lag_168',
        'price_value_lag24', 
        'price_DK1_lag24', 
        'price_NO1_lag24',
        'price_NO5_lag24', 
        'generation_forecast', 
        'load_forecast',
        'wind_forecast'
    ]

    return data, feature_columns

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

    return mse, rmse

if __name__ == "__main__":
    data, feature_columns = load_dataset()

    feature_data = data[feature_columns]
    target_data = data['price_value']

    train_end = int(len(feature_data) * 0.7)
    val_end = int(len(feature_data) * 0.85)

    train_data = feature_data.iloc[:train_end]
    train_targets = target_data.iloc[:train_end]

    validation_data = feature_data.iloc[train_end:val_end]
    validation_targets = target_data.iloc[train_end:val_end]

    test_data = feature_data.iloc[val_end:]
    test_targets = target_data.iloc[val_end:]


    model = XGBoostModel(
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

    validation_mse, validation_energy_kwh = measure_energy(
        evaluate_model,
        model,
        validation_data,
        validation_targets,
    )

    test_mse, test_energy_kwh = measure_energy(
        evaluate_model,
        model,
        test_data,
        test_targets
    )

    print(f"Validation MSE: {validation_mse[0]:.4f}")
    print(f"Validation RMSE: {validation_mse[1]:.4f}")
    print(f"Test MSE: {test_mse[0]:.4f}")
    print(f"Test RMSE: {test_mse[1]:.4f}")
    print(f"Training energy consumption: {training_energy_kwh:.9f} kWh")
    print(f"Validation energy consumption: {validation_energy_kwh:.9f} kWh")
    print(f"Test prediction energy consumption: {test_energy_kwh:.9f} kWh")
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import torch

from codecarbon import OfflineEmissionsTracker
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from lstm.lstm_model import LSTMModel
from logger.csv_logger import write_to_csv
from feature_sets import FEATURE_SETS
from plotter.result_plotter import plot_results

DATASET_PATH = PROJECT_ROOT / "ml_dataset.csv"
RESULTS_CSV_PATH = PROJECT_ROOT / "lstm_results.csv"

def create_sequences(data, seq_length):
    """Create sequences of data for LSTM input."""
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

def prepare_sequences(feature_data, target_data, seq_length, context_data=None):
    context_length = 0 if context_data is None else len(context_data)
    if context_data is not None:
        feature_data = np.concatenate((context_data, feature_data), axis=0)

    sequences = create_sequences(feature_data, seq_length)
    sequence_start = max(context_length - seq_length + 1, 0)
    sequences = sequences[sequence_start:]
    sequence_targets = target_data[:len(sequences)]

    return (
        torch.tensor(sequences, dtype=torch.float32),
        torch.tensor(sequence_targets, dtype=torch.float32).unsqueeze(1),
    )

def load_dataset(file_path=DATASET_PATH):
    data = pd.read_csv(file_path)
    data["delivery_day"] = pd.to_datetime(data["delivery_day"])
    data = (
        data.groupby("delivery_day", as_index=False)
        .mean(numeric_only=True)
        .sort_values("delivery_day")
        .reset_index(drop=True)
    )

    seasonal_splits = [
        (
            "Entire timeline (train: Oct 2024-May 2026, test: Jun 2026-Sep 2026)",
            data["delivery_day"].between("2026-06-01", "2026-09-30"),
            data["delivery_day"].between("2024-10-01", "2026-05-31"),
        ),
        (
            "Summer 2025 (train: Apr-Jun 2025, test: Jul-Sep 2025)",
            data["delivery_day"].between("2025-07-01", "2025-09-30"),
            data["delivery_day"].between("2025-04-01", "2025-06-30"),
        ),
        (
            "Winter 2025 (train: Oct-Dec 2025, test: Jan-Mar 2026)",
            data["delivery_day"].between("2026-01-01", "2026-03-31"),
            data["delivery_day"].between("2025-10-01", "2025-12-31"),
        ),
    ]

    data_filter, test_mask, train_mask = seasonal_splits[1]
    data = data.sort_values(["delivery_day", "hour"]).reset_index(drop=True)
    forecast_horizon = 1
    data["target_price_next_day"] = data["price_value"].shift(-forecast_horizon)
    data = data.dropna(subset=["target_price_next_day"])
    delivery_days = data["delivery_day"].copy()
    data = data.drop(columns=["delivery_day"])
    feature_set_name = "full_set"
    feature_columns = FEATURE_SETS[feature_set_name]

    return (
        data,
        feature_columns,
        train_mask,
        test_mask,
        data_filter,
        forecast_horizon,
        feature_set_name,
        delivery_days,
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

def train_model(
    model,
    data,
    targets,
    validation_data,
    validation_targets,
    target_scaler,
    checkpoint_metadata,
    num_epochs,
    patience,
    learning_rate,
    model_path="best_lstm_model.pt",
):
    criterion = torch.nn.MSELoss()  # Mean Squared Error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer
    best_validation_rmse = float("inf")
    epochs_without_improvement = 0

    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    for epoch in range(num_epochs):
        model.train()  
        for batch_data, batch_targets in dataloader:
            device = next(model.parameters()).device
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad() 
            predictions, _ = model(batch_data)  # Forward pass
            loss = criterion(predictions, batch_targets)
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        _, validation_rmse, _ = evaluate_model(
            model,
            validation_data,
            validation_targets,
            target_scaler,
        )

        if validation_rmse < best_validation_rmse - 0.001:
            best_validation_rmse = validation_rmse
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_epoch": epoch + 1,
                    "best_validation_rmse": best_validation_rmse,
                    **checkpoint_metadata,
                },
                model_path,
            )
            checkpoint_status = " - saved"
        else:
            checkpoint_status = ""
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation RMSE.")
                break

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Loss: {loss.item():.4f}, "
            f"Validation RMSE: {validation_rmse:.4f}"
            f"{checkpoint_status}"
        )

def predict(model, data):
    model.eval()  
    device = next(model.parameters()).device
    data = data.to(device)
    with torch.no_grad():
        predictions, _ = model(data)
    return predictions

def evaluate_model(model, data, targets, target_scaler):

    predictions = predict(model, data)

    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    # back to original price units
    predictions = target_scaler.inverse_transform(predictions)
    targets = target_scaler.inverse_transform(targets)

    r2 = r2_score(targets, predictions)
    rmse = root_mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)

    return r2, rmse, mae

def lstm_feature_importance(
    model,
    data,
    targets,
    target_scaler,
    feature_columns,
    repeats=5,
    ):
    model.eval()
    _, baseline_rmse, _ = evaluate_model(
        model,
        data,
        targets,
        target_scaler,
    )

    importance_rows = []

    for feature_index, feature_name in enumerate(feature_columns):
        rmse_scores = []

        for _ in range(repeats):
            permutation = torch.randperm(data.size(0))

            permuted_data = data.clone()

            permuted_data[:, :, feature_index] = (
                data[permutation, :, feature_index]
            )

            _, permuted_rmse, _ = evaluate_model(
                model,
                permuted_data,
                targets,
                target_scaler,
            )

            rmse_scores.append(permuted_rmse)

        feature_importance = np.mean(rmse_scores) - baseline_rmse
        
        importance_rows.append(
            {
                "feature": feature_name,
                "importance": feature_importance,
            }
        )
    
    return pd.DataFrame(importance_rows).sort_values(
        "importance",
        ascending=False,
    )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    (
        data,
        feature_columns,
        train_mask,
        test_mask,
        data_filter,
        forecast_horizon,
        feature_set_name,
        delivery_days,
    ) = load_dataset()

    feature_data = data[feature_columns]
    target_data = data["target_price_next_day"]
    train_data_all = feature_data.loc[train_mask]
    train_targets_all = target_data.loc[train_mask]
    test_data_all = feature_data.loc[test_mask]
    test_targets_all = target_data.loc[test_mask]
    validation_end = int(len(train_data_all) * 0.85)
    train_data = train_data_all.iloc[:validation_end]
    train_targets = train_targets_all.iloc[:validation_end]
    validation_data = train_data_all.iloc[validation_end:]
    validation_targets = train_targets_all.iloc[validation_end:]

    if train_data.empty or validation_data.empty or test_data_all.empty:
        raise ValueError(
            f"Seasonal split has insufficient rows: train={len(train_data)}, "
            f"validation={len(validation_data)}, test={len(test_data_all)}."
        )

    feature_fill_values = train_data.median().fillna(0.0)
    train_data = train_data.fillna(feature_fill_values)
    feature_scaler = StandardScaler().fit(train_data)
    target_scaler = StandardScaler().fit(
        train_targets.to_numpy().reshape(-1, 1)
    )

    def scale_split(features, targets):
        scaled_features = feature_scaler.transform(
            features.fillna(feature_fill_values)
        )
        scaled_targets = target_scaler.transform(
            targets.to_numpy().reshape(-1, 1)
        ).flatten()
        return scaled_features, scaled_targets

    seq_length = 24
    train_features, train_target_values = scale_split(train_data, train_targets)
    validation_features, validation_target_values = scale_split(
        validation_data, validation_targets
    )
    test_features, test_target_values = scale_split(
        test_data_all, test_targets_all
    )

    train_data, train_targets = prepare_sequences(
        train_features, train_target_values, seq_length
    )
    validation_data, validation_targets = prepare_sequences(
        validation_features,
        validation_target_values,
        seq_length,
        context_data=train_features[-(seq_length - 1):],
    )
    preceding_test_context = np.concatenate(
        (train_features, validation_features),
        axis=0,
    )[-(seq_length - 1):]
    test_data, test_targets = prepare_sequences(
        test_features,
        test_target_values,
        seq_length,
        context_data=preceding_test_context,
    )

    torch.manual_seed(42)
    np.random.seed(42)
    model = LSTMModel(
        input_size=train_data.shape[2],
        hidden_size=50,
        num_layers=2,
        output_size=1,
    ).to(device)
    model_path = "best_lstm_model.pt"
    checkpoint_metadata = {
        "feature_columns": feature_columns,
        "sequence_length": seq_length,
        "target_column": "target_price_next_day",
        "input_size": train_data.shape[2],
        "hidden_size": 50,
        "num_layers": 2,
        "output_size": 1,
        "feature_mean": feature_scaler.mean_.tolist(),
        "feature_scale": feature_scaler.scale_.tolist(),
        "target_mean": target_scaler.mean_.tolist(),
        "target_scale": target_scaler.scale_.tolist(),
        "test_period": data_filter,
        "forecast_horizon": forecast_horizon,
    }

    _, training_energy_kwh = measure_energy(
        train_model,
        model,
        train_data,
        train_targets,
        validation_data,
        validation_targets,
        target_scaler,
        checkpoint_metadata,
        num_epochs=30,
        patience=10,
        learning_rate=0.001,
        model_path=model_path,
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_score, validation_energy_kwh = measure_energy(
        evaluate_model, model, validation_data, validation_targets, target_scaler
    )
    test_metrics, test_energy_kwh = measure_energy(
        evaluate_model, model, test_data, test_targets, target_scaler
    )

    print(f"Validation R²: {validation_score[0]:.4f}")
    print(f"Validation RMSE: {validation_score[1]:.4f}")
    print(f"Validation MAE: {validation_score[2]:.4f}")
    print(f"Test R²: {test_metrics[0]:.4f}")
    print(f"Test RMSE: {test_metrics[1]:.4f}")
    print(f"Test MAE: {test_metrics[2]:.4f}")
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
        "validation_mae": [validation_score[2]],
        "test_r2": [test_metrics[0]],
        "test_rmse": [test_metrics[1]],
        "test_mae": [test_metrics[2]],
        "training_energy_kwh": [training_energy_kwh],
        "validation_energy_kwh": [validation_energy_kwh],
        "test_energy_kwh": [test_energy_kwh],
        "description": None,
    })
    write_to_csv(results_df, RESULTS_CSV_PATH)

    test_predictions = target_scaler.inverse_transform(
        predict(model, test_data).cpu().numpy()
    ).ravel()
    test_targets_plot = target_scaler.inverse_transform(
        test_targets.cpu().numpy()
    ).ravel()
    plot_results(
        delivery_days.loc[test_mask].to_numpy(),
        test_predictions,
        test_targets_plot,
        test_pred_label="Test predictions",
        test_target_label="Test targets",
        title=f"LSTM predictions vs. targets, {data_filter}",
        y_label="Target price",
        filename_suffix="lstm_test_pred_vs_targets",
    )
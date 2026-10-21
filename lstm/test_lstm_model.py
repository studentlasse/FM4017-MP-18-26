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
from sklearn.metrics import r2_score, root_mean_squared_error 

from lstm.lstm_model import LSTMModel
from logger.csv_logger import write_to_csv
from feature_sets import FEATURE_SETS

DATASET_PATH = PROJECT_ROOT / "ml_dataset.csv"
RESULTS_CSV_PATH = PROJECT_ROOT / "lstm_results.csv"

def create_sequences(data, seq_length):
    """Create sequences of data for LSTM input."""
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

def load_dataset(file_path=DATASET_PATH):
    data = pd.read_csv(
        file_path,
        parse_dates=["delivery_day"],
    )

    # Define a date filter:
    #filter_end_date = pd.Timestamp("2020-01-01")
    # Use the entire timeline:
    filter_end_date = None

    if filter_end_date is not None:
        data = data[data["delivery_day"] < filter_end_date].copy()

        data_filter = f"delivery_day < {filter_end_date.date()}"

    else:
        data_filter = "entire timeline"

    data = data.sort_values(
        ["delivery_day", "hour"]
    ).reset_index(drop=True)

    forecast_horizon = 24

    data["target_price_next_day"] = (
        data["price_value"].shift(-forecast_horizon)
    )

    feature_set_name = "lstm_set"
    feature_columns = FEATURE_SETS[feature_set_name].copy()

    required_columns = feature_columns + [
        "target_price_next_day"
    ]

    data = data.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    data = data.drop(columns=["delivery_day"])

    return (
        data,
        feature_columns,
        data_filter,
        forecast_horizon,
        feature_set_name,
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

        _, validation_rmse = evaluate_model(
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

    return r2, rmse

def lstm_feature_importance(
    model,
    data,
    targets,
    target_scaler,
    feature_columns,
    repeats=5,
    ):
    model.eval()
    _, baseline_rmse = evaluate_model(model, data, targets, target_scaler)

    importance_rows = []

    for feature_index, feature_name in enumerate(feature_columns):
        rmse_scores = []

        for _ in range(repeats):
            permutation = torch.randperm(data.size(0))

            permuted_data = data.clone()

            permuted_data[:, :, feature_index] = (
                data[permutation, :, feature_index]
            )

            _, permuted_rmse = evaluate_model(
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

    data, feature_columns, data_filter, forecast_horizon, feature_set_name = load_dataset()

    feature_data = data[feature_columns].to_numpy(dtype=np.float32)
    target_data = data["target_price_next_day"].to_numpy(dtype=np.float32)

    split_index = int(len(feature_data) * 0.7)
    validation_index = int(len(feature_data) * 0.85)

    # Scale features and target
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit only on training data
    feature_scaler.fit(feature_data[:split_index])

    target_scaler.fit(
        target_data[:split_index].reshape(-1, 1)
    )

    # Transform dataset
    feature_data = feature_scaler.transform(feature_data)

    target_data = target_scaler.transform(
        target_data.reshape(-1, 1)
    ).flatten()

    seq_length = 24

    data_sequences = create_sequences(
        feature_data,
        seq_length
    )

    target_start = seq_length - 1
    target_end = target_start + len(data_sequences)

    sequence_targets = target_data[target_start:target_end]

    sequence_targets = torch.tensor(
        sequence_targets,
        dtype=torch.float32
    ).unsqueeze(1)

    train_data = torch.tensor(
        data_sequences[:split_index - seq_length + 1],
        dtype=torch.float32
    )
    train_targets = sequence_targets[:split_index - seq_length + 1]

    validation_data = torch.tensor(
        data_sequences[split_index - seq_length + 1:validation_index - seq_length + 1],
        dtype=torch.float32
    )
    validation_targets = sequence_targets[split_index - seq_length + 1:validation_index - seq_length + 1]

    test_data = torch.tensor(
        data_sequences[validation_index - seq_length + 1:],
        dtype=torch.float32
    )

    model = LSTMModel(
        input_size=train_data.shape[2],
        hidden_size=50,  
        num_layers=2,  
        output_size=1  
    ).to(device)

    model_path = "best_lstm_model.pt"
    checkpoint_metadata = {
        "feature_columns": feature_columns,
        "sequence_length": seq_length,
        "target_column": "target_price_next_day",
        "input_size": train_data.shape[2],
        "hidden_size": 64,
        "num_layers": 2,
        "output_size": 1,
        "feature_mean": feature_scaler.mean_.tolist(),
        "feature_scale": feature_scaler.scale_.tolist(),
        "target_mean": target_scaler.mean_.tolist(),
        "target_scale": target_scaler.scale_.tolist(),
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
    print(
        f"Loaded best checkpoint from epoch {checkpoint['best_epoch']} "
        f"(validation RMSE: {checkpoint['best_validation_rmse']:.4f})"
    )

    validation_score, validation_energy_kwh = measure_energy(
        evaluate_model,
        model,
        validation_data,
        validation_targets,
        target_scaler,
    )

    test_metrics, test_energy_kwh = measure_energy(
        evaluate_model,
        model,
        test_data,
        sequence_targets[validation_index - seq_length + 1:],
        target_scaler,
    )

    print(f"Validation R²: {validation_score[0]:.4f}")
    print(f"Validation RMSE: {validation_score[1]:.4f}")
    print(f"Test R²: {test_metrics[0]:.4f}")
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
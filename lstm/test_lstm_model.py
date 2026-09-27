
import pandas as pd
import numpy as np
import torch
from codecarbon import OfflineEmissionsTracker

from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from lstm.lstm_model import LSTMModel
from data_preperation import create_sequences

DATASET_PATH = "ml_dataset.csv" # from preprocessing/dataset_conversion.py


def load_dataset(file_path=DATASET_PATH):
    data = pd.read_csv(file_path, parse_dates=["delivery_day", "datetime"])

    feature_columns = [
        "price_value",
        "buy_volume_value",
        "sell_volume_value",
        "net_volume",
        "incoming_flow",
        "outgoing_flow",
        "net_flow",
        "import_capacity",
        "export_capacity",
        "day_of_week",
        "month",
        "day_of_year",
        "price_lag_1",
        "price_lag_2",
        "price_lag_3",
        "price_lag_24",
        "price_lag_48",
        "price_lag_168",
        "price_ma_24",
        "price_ma_168",
    ]

    return data.sort_values("datetime").reset_index(drop=True), feature_columns

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
    learning_rate,
    model_path="best_lstm_model.pt",
):
    criterion = torch.nn.MSELoss()  # Mean Squared Error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer
    best_validation_rmse = float("inf")

    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(num_epochs):
        model.train()  
        for batch_data, batch_targets in dataloader:
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

        if validation_rmse < best_validation_rmse:
            best_validation_rmse = validation_rmse
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

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Loss: {loss.item():.4f}, "
            f"Validation RMSE: {validation_rmse:.4f}"
            f"{checkpoint_status}"
        )

def predict(model, data):
    model.eval()  
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

    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)

    return mse, rmse

if __name__ == "__main__":
    data, feature_columns = load_dataset()

    feature_data = data[feature_columns].to_numpy(dtype=np.float32)
    target_data = data["target_price_next_hour"].to_numpy(dtype=np.float32)

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

    prediction_data = torch.tensor(
        data_sequences[validation_index - seq_length + 1:],
        dtype=torch.float32
    )

    model = LSTMModel(
        input_size=train_data.shape[2],
        hidden_size=50,  
        num_layers=2,  
        output_size=1  
    )

    model_path = "best_lstm_model.pt"
    checkpoint_metadata = {
        "feature_columns": feature_columns,
        "sequence_length": seq_length,
        "target_column": "target_price_next_hour",
        "input_size": train_data.shape[2],
        "hidden_size": 50,
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
        learning_rate=0.001,
        model_path=model_path,
    )

    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Loaded best checkpoint from epoch {checkpoint['best_epoch']} "
        f"(validation RMSE: {checkpoint['best_validation_rmse']:.4f})"
    )

    validation_mse, validation_energy_kwh = measure_energy(
        evaluate_model,
        model,
        validation_data,
        validation_targets,
        target_scaler,
    )

    test_mse, test_energy_kwh = measure_energy(
        evaluate_model,
        model,
        prediction_data,
        sequence_targets[validation_index - seq_length + 1:],
        target_scaler,
    )

    print(f"Validation MSE: {validation_mse[0]:.4f}")
    print(f"Validation RMSE: {validation_mse[1]:.4f}")
    print(f"Test MSE: {test_mse[0]:.4f}")
    print(f"Test RMSE: {test_mse[1]:.4f}")
    print(f"Training energy consumption: {training_energy_kwh:.9f} kWh")
    print(f"Validation energy consumption: {validation_energy_kwh:.9f} kWh")
    print(f"Test prediction energy consumption: {test_energy_kwh:.9f} kWh")
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

from gru.gru_model import GRUModel
from logger.csv_logger import write_to_csv, save_plot_csv
from feature_sets import load_feature_set, load_feature_set_metadata
from plotter.result_plotter import plot_results

DATASET_PATH = PROJECT_ROOT / "ml_dataset.csv"
RESULTS_CSV_PATH = PROJECT_ROOT / "results/gru_results.csv"
TEST_REPEATS = 100

# Seasonal split and feature selection mode to evaluate. The feature set is
# read from elasticnet_selected_features.json for this exact split and mode.
SPLIT_INDEX = 1
SELECTION_MODE = "moderate"  # "aggressive", "moderate", or "none"

def create_sequences(data, seq_length):
    """Create sequences of data for GRU input."""
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

def prepare_sequences(
    feature_data,
    target_data,
    seq_length,
    context_data=None,
    anchor_data=None,
):
    context_length = 0 if context_data is None else len(context_data)
    if context_data is not None:
        feature_data = np.concatenate((context_data, feature_data), axis=0)

    sequences = create_sequences(feature_data, seq_length)
    sequence_start = max(context_length - seq_length + 1, 0)
    sequences = sequences[sequence_start:]

    target_start = sequence_start + seq_length - 1 - context_length
    sequence_targets = target_data[target_start:target_start + len(sequences)]

    anchors = None
    if anchor_data is not None:
        anchor_values = anchor_data[target_start:target_start + len(sequences)]
        anchors = torch.tensor(anchor_values, dtype=torch.float32).unsqueeze(1)

    return (
        torch.tensor(sequences, dtype=torch.float32),
        torch.tensor(sequence_targets, dtype=torch.float32).unsqueeze(1),
        anchors,
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

    data_filter, test_mask, train_mask = seasonal_splits[SPLIT_INDEX]
    data = data.sort_values(["delivery_day", "hour"]).reset_index(drop=True)
    forecast_horizon = 1
    data["target_price_next_day"] = data["price_value"].shift(-forecast_horizon)

    # Predict the change from the last observed price instead of the absolute
    # price level. This reduces sensitivity to price-level shifts between the
    # training and test periods; the anchor is added back during evaluation.
    
    data["target_price_delta"] = (
        data["target_price_next_day"] - data["price_value"]
    )
    data = data.dropna(subset=["target_price_next_day"])
    delivery_days = data["delivery_day"].copy()
    data = data.drop(columns=["delivery_day"])
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


def measure_repeated_test_energy(operation, *args, repeats, **kwargs):
    """Measure one tracker around a warmed-up repeated test workload."""
    operation(*args, repeats=1, **kwargs)
    return measure_energy(operation, *args, repeats=repeats, **kwargs)

def train_model(
    model,
    data,
    targets,
    validation_data,
    validation_targets,
    validation_anchors,
    target_scaler,
    checkpoint_metadata,
    num_epochs,
    patience,
    learning_rate,
    model_path="best_gru_model.pt",
):
    criterion = torch.nn.L1Loss() # Using L1 loss (MAE) for training 
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
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
            validation_anchors,
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
                break


def predict(model, data):
    model.eval()  
    device = next(model.parameters()).device
    data = data.to(device)
    with torch.no_grad():
        predictions, _ = model(data)
    return predictions

def evaluate_model(model, data, targets, target_scaler, anchors):

    predictions = predict(model, data)

    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    anchors = anchors.cpu().numpy()

    # back to original price units: predicted change + last observed price
    predictions = target_scaler.inverse_transform(predictions) + anchors
    targets = target_scaler.inverse_transform(targets) + anchors

    r2 = r2_score(targets, predictions)
    rmse = root_mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)

    return r2, rmse, mae

def repeated_test_evaluation(
        model,
        data, 
        targets,
        target_scaler,
        anchors,
        repeats=TEST_REPEATS,
    ):

    metrics = []

    for _ in range(repeats):
        metrics.append(
            evaluate_model(
                model,
                data,
                targets,
                target_scaler,
                anchors,
            )
        )

    metrics = np.array(metrics)
    return metrics.mean(axis=0), metrics.std(axis=0)

def gru_feature_importance(
    model,
    data,
    targets,
    target_scaler,
    anchors,
    feature_columns,
    repeats=5,
    ):
    model.eval()
    _, baseline_rmse, _ = evaluate_model(
        model,
        data,
        targets,
        target_scaler,
        anchors,
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
                anchors,
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
    ) = load_dataset()


    feature_data = data[feature_columns]
    target_data = data["target_price_delta"]
    anchor_data = data["price_value"]
    train_data_all = feature_data.loc[train_mask]
    train_targets_all = target_data.loc[train_mask]
    train_anchors_all = anchor_data.loc[train_mask]
    test_data_all = feature_data.loc[test_mask]
    test_targets_all = target_data.loc[test_mask]
    test_anchors_all = anchor_data.loc[test_mask]
    validation_end = int(len(train_data_all) * 0.85)
    train_data = train_data_all.iloc[:validation_end]
    train_targets = train_targets_all.iloc[:validation_end]
    train_anchors = train_anchors_all.iloc[:validation_end]
    validation_data = train_data_all.iloc[validation_end:]
    validation_targets = train_targets_all.iloc[validation_end:]
    validation_anchors = train_anchors_all.iloc[validation_end:]

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

    def scale_split(features, targets, anchors):
        scaled_features = feature_scaler.transform(
            features.fillna(feature_fill_values)
        )
        scaled_targets = target_scaler.transform(
            targets.to_numpy().reshape(-1, 1)
        ).flatten()
        return scaled_features, scaled_targets, anchors.to_numpy()

    seq_length = 24
    train_features, train_target_values, train_anchor_values = scale_split(
        train_data, train_targets, train_anchors
    )
    validation_features, validation_target_values, validation_anchor_values = scale_split(
        validation_data, validation_targets, validation_anchors
    )
    test_features, test_target_values, test_anchor_values = scale_split(
        test_data_all, test_targets_all, test_anchors_all
    )

    train_data, train_targets, train_anchors = prepare_sequences(
        train_features,
        train_target_values,
        seq_length,
        anchor_data=train_anchor_values,
    )
    validation_data, validation_targets, validation_anchors = prepare_sequences(
        validation_features,
        validation_target_values,
        seq_length,
        context_data=train_features[-(seq_length - 1):],
        anchor_data=validation_anchor_values,
    )
    preceding_test_context = np.concatenate(
        (train_features, validation_features),
        axis=0,
    )[-(seq_length - 1):]
    test_data, test_targets, test_anchors = prepare_sequences(
        test_features,
        test_target_values,
        seq_length,
        context_data=preceding_test_context,
        anchor_data=test_anchor_values,
    )

    parameters = {
        "hidden_size": 16,
        "num_layers": 2,
        "output_size": 1,
        "num_epochs" : 50,
        "patience" : 10,
        "learning_rate" : 0.03,
    }

    torch.manual_seed(42)
    np.random.seed(42)
    model = GRUModel(
        input_size=train_data.shape[2],
        hidden_size=parameters["hidden_size"],
        num_layers=parameters["num_layers"],
        output_size=parameters["output_size"],
    ).to(device)
    model_path = "best_gru_model.pt"
    checkpoint_metadata = {
        "feature_columns": feature_columns,
        "sequence_length": seq_length,
        "target_column": "target_price_delta",
        "target_mode": "delta",
        "input_size": train_data.shape[2],
        "hidden_size": parameters["hidden_size"],
        "num_layers": parameters["num_layers"],
        "output_size": parameters["output_size"],
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
        validation_anchors,
        target_scaler,
        checkpoint_metadata,
        num_epochs=parameters["num_epochs"],
        patience=parameters["patience"],
        learning_rate=parameters["learning_rate"],
        model_path=model_path,
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_score = evaluate_model(
        model,
        validation_data,
        validation_targets,
        target_scaler,
        validation_anchors,
    )

    (test_metrics, test_metrics_std), test_energy_kwh = measure_repeated_test_energy(
        repeated_test_evaluation,
        model,
        test_data,
        test_targets,
        target_scaler,
        test_anchors,
        repeats=TEST_REPEATS
    )



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
        "test_r2_std": [test_metrics_std[0]],
        "test_rmse_std": [test_metrics_std[1]],
        "test_mae_std": [test_metrics_std[2]],
        "training_energy_kwh": [training_energy_kwh],
        "test_energy_kwh_total": [test_energy_kwh],
        "test_energy_kwh_per_run": [test_energy_kwh / TEST_REPEATS],
        "test_repeats": [TEST_REPEATS],
        "description": (
            "Training energy is one complete training run including per-epoch "
            "validation; test energy is the total for repeated evaluations."
        ),
    })

    parameters_df = pd.DataFrame([
    {f"param_{name}": value for name, value in parameters.items()}
    ])
    results_df = pd.concat([results_df, parameters_df], axis=1)

    RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_to_csv(results_df, RESULTS_CSV_PATH)

    test_predictions = target_scaler.inverse_transform(
        predict(model, test_data).cpu().numpy()
    ).ravel() + test_anchors.numpy().ravel()
    test_targets_plot = target_scaler.inverse_transform(
        test_targets.cpu().numpy()
    ).ravel() + test_anchors.numpy().ravel()

    # persistence baseline: predict next day = last observed price (the anchor)
    persistence_pred = test_anchors.numpy().ravel()

    # metrics
    def rmse(a, b): return np.sqrt(np.mean((a - b) ** 2))
    def mae(a, b): return np.mean(np.abs(a - b))
    def r2(a, b): return 1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)

    plot_data = pd.DataFrame({
            "delivery_day": pd.to_datetime(delivery_days.loc[test_mask]).to_numpy(),
            "test_prediction": test_predictions,
            "test_target": test_targets_plot,
            })
        
    split_name = f"split_{SPLIT_INDEX}"
    feature_info = f"{SELECTION_MODE}_{len(feature_columns)}features"

    plot_data_description = (f"_gru_{split_name}_{feature_info}")

    save_plot_csv(plot_data, plot_data_description)

    plot_results(
        delivery_days.loc[test_mask].to_numpy(),
        test_predictions,
        test_targets_plot,
        test_pred_label="Test predictions",
        test_target_label="Test targets",
        title=f"GRU predictions vs. targets, {data_filter}",
        y_label="Target price",
        filename_suffix="gru_test_pred_vs_targets",
    )
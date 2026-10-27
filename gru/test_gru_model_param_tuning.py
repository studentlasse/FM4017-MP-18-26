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
from logger.csv_logger import write_to_csv
from feature_sets import FEATURE_SETS
from plotter.result_plotter import plot_results

from itertools import product
import random

DATASET_PATH = PROJECT_ROOT / "ml_dataset.csv"
RESULTS_CSV_PATH = PROJECT_ROOT / "gru_search_results.csv"

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
        anchor_data=None
        ):
    
    context_length = 0 if context_data is None else len(context_data)
    if context_data is not None:
        feature_data = np.concatenate((context_data, feature_data), axis=0)

    sequences = create_sequences(feature_data, seq_length)
    sequence_start = max(context_length - seq_length + 1, 0)
    sequences = sequences[sequence_start:]

    # Each window must be paired with the target of its LAST row, not the
    # first one. Context rows are only inputs, so they are subtracted here.
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

    data_filter, test_mask, train_mask = seasonal_splits[1]
    data = data.sort_values(["delivery_day", "hour"]).reset_index(drop=True)
    forecast_horizon = 1
    data["target_price_next_day"] = data["price_value"].shift(-forecast_horizon)
    # Predict the change from the last observed price instead of the price
    # level: the level is not identifiable out-of-sample (and drifts between
    # train and test periods), the change is.
    data["target_price_delta"] = (
        data["target_price_next_day"] - data["price_value"]
    )
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
    validation_anchors,
    target_scaler,
    checkpoint_metadata,
    num_epochs,
    patience,
    learning_rate,
    loss_name="mse",
    model_path="best_gru_model.pt",
):
    criterion = get_criterion(loss_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # Adam optimizer
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
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation RMSE.")
                break

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Loss: {loss.item():.4f}, "
            f"Validation RMSE: {validation_rmse:.4f}"
            f"{checkpoint_status}"
        )


def get_criterion(loss_name):
    if loss_name == "mse":
        return torch.nn.MSELoss()
    if loss_name == "mae":
        return torch.nn.L1Loss()
    if loss_name == "smooth_l1":
        return torch.nn.SmoothL1Loss()


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


def train_and_score(
        config, 
        train_data, 
        train_targets,
        validation_data, 
        validation_targets,
        validation_anchors,
        target_scaler, 
        checkpoint_dir, 
        device
        ):

    model = GRUModel(
        input_size=train_data.shape[2],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        output_size=1,
    ).to(device)

    checkpoint_path = checkpoint_dir / (
        f"hidden{config['hidden_size']}_"
        f"layers{config['num_layers']}_"
        f"lr{config['learning_rate']}_"
        f"loss{config['loss_name']}_"
        f"seq{config['sequence_length']}.pt"
    )

    checkpoint_metadata = {
        "model_type": "GRU",
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "learning_rate": config["learning_rate"],
        "loss_name": config["loss_name"],
        "sequence_length": config["sequence_length"],
        "target_column": "target_price_delta",
        "target_mode": "delta",
        "input_size": train_data.shape[2],
        "output_size": 1,
    }

    train_model(
        model,
        train_data,
        train_targets,
        validation_data,
        validation_targets,
        validation_anchors,
        target_scaler,
        checkpoint_metadata,
        num_epochs=config["num_epochs"],
        patience=config["patience"],
        learning_rate=config["learning_rate"],
        loss_name=config["loss_name"],
        model_path=str(checkpoint_path),
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    validation_metrics = evaluate_model(
        model,
        validation_data,
        validation_targets,
        target_scaler,
        validation_anchors,
    )

    return {
        **config,
        "validation_r2": validation_metrics[0],
        "validation_rmse": validation_metrics[1],
        "validation_mae": validation_metrics[2],
        "best_epoch": checkpoint["best_epoch"],
        "checkpoint_path": str(checkpoint_path),
    }


def run_search(
    search_space,
    train_data,
    train_targets,
    validation_data,
    validation_targets,
    validation_anchors,
    target_scaler,
    checkpoint_dir,
    device,
    method="grid",
    n_iter=10,
    random_state=42,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    keys = list(search_space)
    all_configs = [
        dict(zip(keys, values))
        for values in product(*(search_space[key] for key in keys))
    ]

    if method == "grid":
        configs = all_configs
    elif method == "random":
        rng = random.Random(random_state)
        n_samples = min(n_iter, len(all_configs))
        configs = rng.sample(all_configs, k=n_samples)
    else:
        raise ValueError(
            f"Unsupported search method: {method!r}. Use 'grid' or 'random'."
        )

    results = []

    for index, config in enumerate(configs, start=1):
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        config = {**config, "search_method": method}
        print(f"{method.capitalize()} search {index}/{len(configs)}: {config}")

        result = train_and_score(
            config=config,
            train_data=train_data,
            train_targets=train_targets,
            validation_data=validation_data,
            validation_targets=validation_targets,
            validation_anchors=validation_anchors,
            target_scaler=target_scaler,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )
        results.append(result)
        print(f"Validation RMSE: {result['validation_rmse']:.4f}")

    if not results:
        raise ValueError("Search produced no configurations to evaluate.")

    return (
        pd.DataFrame(results)
        .sort_values("validation_rmse")
        .reset_index(drop=True)
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
    (
        validation_features, 
        validation_target_values,
        validation_anchor_values
    ) = scale_split(validation_data, validation_targets, validation_anchors)
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


    search_space = {
    "hidden_size": [16, 32, 64, 128],
    "num_layers": [1, 2],
    "learning_rate": [0.001, 0.002, 0.003, 0.01],
    "loss_name": ["mse", "mae", "smooth_l1"],
    "sequence_length": [24],
    "num_epochs": [50],
    "patience": [10],
    }

    search_method = "grid"
    #search_method = "random"

    #results = pd.DataFrame()

    if search_method == "grid":
        ####  Grid search

        results, search_energy_kwh = measure_energy(
        run_search,
        search_space=search_space,
        train_data=train_data,
        train_targets=train_targets,
        validation_data=validation_data,
        validation_targets=validation_targets,
        validation_anchors=validation_anchors,
        target_scaler=target_scaler,
        checkpoint_dir=PROJECT_ROOT / "gru_checkpoints",
        device=device,
        method="grid",
        )

        print(f"Total grid-search energy: {search_energy_kwh:.9f} kWh")
        results["search_method"] = "grid"
        results["tuning_energy_kwh"] = search_energy_kwh

        results.to_csv(
        PROJECT_ROOT / "gru_grid_search_results.csv",
        index=False,
        )

        best_result = results.iloc[0]
        print("Best configuration:")
        print(best_result)



    if search_method == "random":
        ### Random Search

        results, search_energy_kwh = measure_energy(
        run_search,
        search_space=search_space,
        train_data=train_data,
        train_targets=train_targets,
        validation_data=validation_data,
        validation_targets=validation_targets,
        validation_anchors=validation_anchors,
        target_scaler=target_scaler,
        checkpoint_dir=PROJECT_ROOT / "gru_checkpoints",
        device=device,
        method="random",
        n_iter=10,
        random_state=42,
        )


        print(f"Total random-search energy: {search_energy_kwh:.9f} kWh")
        results["search_method"] = "random"
        results["tuning_energy_kwh"] = search_energy_kwh

        results.to_csv(
        PROJECT_ROOT / "gru_random_search_results.csv",
        index=False,
        )

        best_result = results.iloc[0]
        print("Best configuration:")
        print(best_result)


    best_config = best_result.to_dict()
    best_checkpoint_path = best_result["checkpoint_path"]

    best_model = GRUModel(
        input_size=train_data.shape[2],
        hidden_size=int(best_config["hidden_size"]),
        num_layers=int(best_config["num_layers"]),
        output_size=1,
    ).to(device)

    checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    best_model.load_state_dict(checkpoint["model_state_dict"])

    validation_score, validation_energy_kwh = measure_energy(
        evaluate_model, 
        best_model,
        validation_data,
        validation_targets,
        target_scaler,
        validation_anchors
    )

    test_metrics, test_energy_kwh = measure_energy(
        evaluate_model,
        best_model,
        test_data,
        test_targets,
        target_scaler,
        test_anchors,
    )

    print(f"Test R²: {test_metrics[0]:.4f}")
    print(f"Test RMSE: {test_metrics[1]:.4f}")
    print(f"Test MAE: {test_metrics[2]:.4f}")
    print(f"Test prediction energy: {test_energy_kwh:.9f} kWh")

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


    test_predictions = target_scaler.inverse_transform(
        predict(best_model, test_data).cpu().numpy()
    ).ravel() + test_anchors.numpy().ravel()
    test_targets_plot = target_scaler.inverse_transform(
        test_targets.cpu().numpy()
    ).ravel() + test_anchors.numpy().ravel()


    plot_results(
        delivery_days.loc[test_mask].to_numpy(),
        test_predictions,
        test_targets_plot,
        test_pred_label="Test predictions",
        test_target_label="Test targets",
        title=f"GRU predictions vs. targets, {data_filter}",
        y_label="Target price",
        filename_suffix="GRU_test_pred_vs_targets",
    )

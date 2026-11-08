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
RESULTS_CSV_PATH = PROJECT_ROOT / "results" / "lstm_tuning_results.csv"
RETRAIN_REPEATS = 5
RETRAIN_SEED_BASE = 42

# Seasonal splits and feature selection modes to evaluate. Every combination of
# SPLIT_INDICES x SELECTION_MODES is tuned in a separate search run. The feature
# set is read from elasticnet_selected_features.json for each split and mode.
SPLIT_INDICES = (0, 1, 2)
VALIDATION_FRACTION = 0.15
TRAIN_FRACTION = 1.0 - VALIDATION_FRACTION
BATCH_SIZE = 32

SEARCH_SPACE = {
    "hidden_size": [16, 32, 64, 128],
    "num_layers": [1, 2, 3],
    "learning_rate": [0.001, 0.02, 0.03, 0.01],
    "loss_name": ["mse"],
    "sequence_length": [24],
    "num_epochs": [50],
    "patience": [10],
}

def create_sequences(data, seq_length):
    """Create sequences of data for LSTM input."""
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

def load_dataset(file_path=DATASET_PATH, split_index=0, selection_mode="moderate"):
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

    data_filter, test_mask, train_mask = seasonal_splits[split_index]
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
    model_path="best_lstm_model.pt",
    batch_size=BATCH_SIZE,
):
    criterion = get_criterion(loss_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_validation_rmse = float("inf")
    epochs_without_improvement = 0

    device = next(model.parameters()).device
    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    validation_targets_original = (
        target_scaler.inverse_transform(validation_targets.cpu().numpy())
        + validation_anchors.cpu().numpy()
    )

    for epoch in range(num_epochs):
        model.train()
        for batch_data, batch_targets in dataloader:
            batch_data = batch_data.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            optimizer.zero_grad() 
            predictions, _ = model(batch_data)  # Forward pass
            loss = criterion(predictions, batch_targets)
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        validation_rmse = evaluate_rmse(
            model,
            validation_data,
            target_scaler,
            validation_anchors,
            validation_targets_original,
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
    with torch.inference_mode():
        predictions, _ = model(data)
    return predictions

def evaluate_rmse(
    model,
    data,
    target_scaler,
    anchors,
    targets_original,
):
    """Cheap validation metric for early stopping: RMSE in price units."""
    predictions = predict(model, data).cpu().numpy()
    predictions = target_scaler.inverse_transform(predictions) + anchors.cpu().numpy()
    return root_mean_squared_error(targets_original, predictions)

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
        repeats=100,
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


def lstm_feature_importance(
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

    model = LSTMModel(
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
        "model_type": "LSTM",
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
    target_scaler,
    feature_scaler,
    feature_columns,
    data_filter,
    forecast_horizon,
    checkpoint_dir,
    device,
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
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = LSTMModel(
            input_size=train_data.shape[2],
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_layers"]),
            output_size=1,
        ).to(device)

        checkpoint_path = (
            checkpoint_dir / f"retrained_best_model_run{run_index}.pt"
        )
        checkpoint_metadata = {
            "feature_columns": feature_columns,
            "target_column": "target_price_delta",
            "target_mode": "delta",
            "hidden_size": int(config["hidden_size"]),
            "num_layers": int(config["num_layers"]),
            "learning_rate": float(config["learning_rate"]),
            "loss_name": config["loss_name"],
            "sequence_length": int(config["sequence_length"]),
            "input_size": train_data.shape[2],
            "output_size": 1,
            "feature_mean": feature_scaler.mean_.tolist(),
            "feature_scale": feature_scaler.scale_.tolist(),
            "target_mean": target_scaler.mean_.tolist(),
            "target_scale": target_scaler.scale_.tolist(),
            "test_period": data_filter,
            "forecast_horizon": forecast_horizon,
            "run_index": run_index,
            "run_seed": seed,
        }

        _, run_training_energy = measure_energy(
            train_model,
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

        validation_score = evaluate_model(
            model,
            validation_data,
            validation_targets,
            target_scaler,
            validation_anchors,
        )
        test_score, run_test_energy = measure_energy(
            evaluate_model,
            model,
            test_data,
            test_targets,
            target_scaler,
            test_anchors,
        )

        validation_metrics.append(validation_score)
        test_metrics.append(test_score)
        training_energies.append(run_training_energy)
        test_energies.append(run_test_energy)

        if best_run is None or validation_score[1] < best_run["validation_rmse"]:
            best_run = {
                "model": model,
                "validation_rmse": validation_score[1],
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
    config_energies = []

    for index, config in enumerate(configs, start=1):
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        config = {**config, "search_method": method}

        result, config_energy_kwh = measure_energy(
            train_and_score,
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
        result["config_energy_kwh"] = float(config_energy_kwh)
        config_energies.append(float(config_energy_kwh))
        results.append(result)

    if not results:
        raise ValueError("Search produced no configurations to evaluate.")

    results_df = (
        pd.DataFrame(results)
        .sort_values("validation_rmse")
        .reset_index(drop=True)
    )
    return results_df, float(np.sum(config_energies))


def run_experiment(split_index, selection_mode, device, search_method="grid", n_iter=100, random_state=42):
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
    target_data = data["target_price_delta"]
    anchor_data = data["price_value"]
    train_data_all = feature_data.loc[train_mask]
    train_targets_all = target_data.loc[train_mask]
    train_anchors_all = anchor_data.loc[train_mask]
    test_data_all = feature_data.loc[test_mask]
    test_targets_all = target_data.loc[test_mask]
    test_anchors_all = anchor_data.loc[test_mask]
    validation_end = int(len(train_data_all) * TRAIN_FRACTION)
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

    search_space = SEARCH_SPACE

    if search_method not in {"grid", "random"}:
        raise ValueError(
            f"Unsupported search method: {search_method!r}. "
            "Use 'grid' or 'random'."
        )

    checkpoint_dir = (
        PROJECT_ROOT
        / "lstm_checkpoints"
        / f"split_{split_index}_{selection_mode}"
    )

    results, search_energy_kwh = run_search(
        search_space=search_space,
        train_data=train_data,
        train_targets=train_targets,
        validation_data=validation_data,
        validation_targets=validation_targets,
        validation_anchors=validation_anchors,
        target_scaler=target_scaler,
        checkpoint_dir=checkpoint_dir,
        device=device,
        method=search_method,
        n_iter=n_iter,
        random_state=random_state,
    )

    results["search_method"] = search_method
    results["tuning_energy_kwh"] = search_energy_kwh

    results_path = (
        PROJECT_ROOT
        / "results"
        / f"lstm_{search_method}_search_results_split{split_index}_"
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
        target_scaler=target_scaler,
        feature_scaler=feature_scaler,
        feature_columns=feature_columns,
        data_filter=data_filter,
        forecast_horizon=forecast_horizon,
        checkpoint_dir=checkpoint_dir,
        device=device,
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
        "best_param_hidden_size": best_config["hidden_size"],
        "best_param_num_layers": best_config["num_layers"],
        "best_param_learning_rate": best_config["learning_rate"],
        "best_param_loss_name": best_config["loss_name"],
        "best_param_sequence_length": best_config["sequence_length"],
        "best_param_num_epochs": best_config["num_epochs"],
        "best_param_patience": best_config["patience"],
       })
    RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_to_csv(results_df, RESULTS_CSV_PATH)

    test_predictions = target_scaler.inverse_transform(
        predict(best_model, test_data).cpu().numpy()
    ).ravel() + test_anchors.numpy().ravel()
    test_targets_plot = target_scaler.inverse_transform(
        test_targets.cpu().numpy()
    ).ravel() + test_anchors.numpy().ravel()

    plot_data = pd.DataFrame({
            "delivery_day": pd.to_datetime(delivery_days.loc[test_mask]).to_numpy(),
            "test_prediction": test_predictions,
            "test_target": test_targets_plot,
            })
        
    split_name = f"split_{split_index}"
    feature_info = f"{selection_mode}_{len(feature_columns)}features"

    plot_data_description = (f"_lstm_tuning_{split_name}_{feature_info}")

    save_plot_csv(plot_data, plot_data_description)

    plot_results(
        delivery_days.loc[test_mask].to_numpy(),
        test_predictions,
        test_targets_plot,
        test_pred_label="Test predictions",
        test_target_label="Test targets",
        title=f"LSTM predictions vs. targets, {data_filter}",
        y_label="Target price",
        filename_suffix=(
            f"LSTM_test_pred_vs_targets_{split_name}_{selection_mode}"
        ),
    )

    return results_df


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False # Disable cuDNN benchmark for reproducibility

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
                    device=device,
                    search_method=search_method,
                    n_iter=n_iter,
                    random_state=random_state,
                )
            )

    summary = pd.concat(all_results, ignore_index=True)
    summary_path = PROJECT_ROOT / "results" / f"lstm_{search_method}_tuning_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
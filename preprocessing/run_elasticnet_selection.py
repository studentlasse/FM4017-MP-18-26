"""Run ElasticNet feature selection for every seasonal split.

For each split, cross-validation is run once on the training window. The
selected ``(alpha, rho)`` pair is then reused to derive three feature sets:

``aggressive``
    The CV-selected alpha (the strongest penalty that still keeps features).
``moderate``
    A fraction of the CV-selected alpha, which keeps more correlated features.
``none``
    Every candidate feature, i.e. no feature selection.

The sets are written to a single JSON file keyed by split index and mode, so
the model scripts can pick the set that matches their own ``SPLIT_INDEX`` and
``SELECTION_MODE`` constants.

Selection is performed on the training split only, and by default on the
first 85% of that window so it matches the train/validation split used by the
model scripts. The test period never influences which features are kept.

Configuration
-------------
Run settings live in the constants at the top of this file: dataset path,
candidate feature set, which splits to run, the CV grid, and output paths.
Edit them and run:

    uv run python preprocessing/run_elasticnet_selection.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from feature_sets import FEATURE_SETS
from preprocessing.elasticnet_selection import (
    DEFAULT_L1_RATIOS,
    select_features_at_alpha,
    select_features_elasticnet,
)

DATASET_PATH = PROJECT_ROOT / "ml_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "elasticnet_selected_features.json"
COEFFICIENTS_PATH = PROJECT_ROOT / "elasticnet_selection_results.csv"

# ---- Run configuration ---------------------------------------------------
# Candidate feature set: a key from feature_sets.py, or "all" for every
# numeric column in the dataset.
FEATURE_SET_NAME = "all"

# Split indices to run. None runs every seasonal split. When a subset is
# given, the output file is replaced with only those splits.
SPLIT_INDICES = None

# Regularization path and cross-validation grid.
L1_RATIOS = DEFAULT_L1_RATIOS
N_ALPHAS = 100
EPS = 1e-3
CV_SPLITS = 5

# Fraction of the training window used for selection. 0.85 matches the model
# scripts' 15% validation hold-out.
TRAIN_RATIO = 0.85

# Limits applied when turning coefficients into the selected feature list.
MIN_ABS_COEF = 1e-8
MAX_FEATURES = None

# Number of top features printed per mode.
SHOW = 20

SEASONAL_SPLITS = [
    {
        "name": "Entire timeline (train: Oct 2024-May 2026, test: Jun 2026-Sep 2026)",
        "train": ("2024-10-01", "2026-05-31"),
        "test": ("2026-06-01", "2026-09-30"),
    },
    {
        "name": "Summer 2025 (train: Apr-Jun 2025, test: Jul-Sep 2025)",
        "train": ("2025-04-01", "2025-06-30"),
        "test": ("2025-07-01", "2025-09-30"),
    },
    {
        "name": "Winter 2025 (train: Oct-Dec 2025, test: Jan-Mar 2026)",
        "train": ("2025-10-01", "2025-12-31"),
        "test": ("2026-01-01", "2026-03-31"),
    },
]

# ``alpha_scale`` multiplies the alpha chosen by cross-validation. A scale of
# 1.0 is the CV-optimal (most aggressive) penalty; smaller values weaken the
# penalty and keep more features.
SELECTION_MODES = {
    "aggressive": {
        "alpha_scale": 1.0,
        "description": "CV-selected alpha; strongest penalty, fewest features.",
    },
    "moderate": {
        "alpha_scale": 0.2,
        "description": "20% of the CV-selected alpha; keeps more correlated features.",
    },
    "none": {
        "alpha_scale": None,
        "description": "All candidate features; no feature selection.",
    },
}


def load_training_data(file_path, feature_set_name, split_index, train_ratio=TRAIN_RATIO):
    """Load the daily dataset and return the training features and target.

    ``train_ratio`` keeps only the first fraction of the training window, so
    selection stops where the model's own training split ends. The model
    scripts use the remaining 15% for validation.
    """
    if not 0 < train_ratio <= 1:
        raise ValueError("train_ratio must be in (0, 1].")

    data = pd.read_csv(file_path)
    data["delivery_day"] = pd.to_datetime(data["delivery_day"])

    # One row per day, matching the model scripts.
    data = (
        data.groupby("delivery_day", as_index=False)
        .mean(numeric_only=True)
        .sort_values("delivery_day")
        .reset_index(drop=True)
    )

    split = SEASONAL_SPLITS[split_index]

    data = data.sort_values(["delivery_day", "hour"]).reset_index(drop=True)
    data["target_price_next_day"] = data["price_value"].shift(-1)
    data = data.dropna(subset=["target_price_next_day"]).reset_index(drop=True)

    train_mask = data["delivery_day"].between(*split["train"])

    if feature_set_name == "all":
        feature_columns = [
            column
            for column in data.columns
            if column not in {"delivery_day", "target_price_next_day"}
            and pd.api.types.is_numeric_dtype(data[column])
        ]
    else:
        feature_columns = FEATURE_SETS[feature_set_name]

    missing = [feature for feature in feature_columns if feature not in data.columns]
    if missing:
        raise ValueError(
            f"Feature set {feature_set_name!r} references columns that are not "
            f"in {file_path}: {missing}"
        )

    X_train = data.loc[train_mask, feature_columns]
    y_train = data.loc[train_mask, "target_price_next_day"]

    if train_ratio < 1.0:
        cutoff = int(len(X_train) * train_ratio)
        X_train = X_train.iloc[:cutoff]
        y_train = y_train.iloc[:cutoff]

    if X_train.empty:
        raise ValueError(
            f"The training period {split['train']} contains no rows in {file_path}."
        )

    return X_train, y_train, split


def select_for_split(split_index):
    """Run CV once and derive every selection mode for one split."""
    X_train, y_train, split = load_training_data(
        DATASET_PATH,
        FEATURE_SET_NAME,
        split_index,
        TRAIN_RATIO,
    )

    cv_selection = select_features_elasticnet(
        X_train,
        y_train,
        l1_ratios=L1_RATIOS,
        n_alphas=N_ALPHAS,
        eps=EPS,
        n_splits=CV_SPLITS,
        min_abs_coef=MIN_ABS_COEF,
        max_features=MAX_FEATURES,
        alpha_scale=1.0,
    )

    alpha_scales = {
        mode: settings["alpha_scale"]
        for mode, settings in SELECTION_MODES.items()
        if settings["alpha_scale"] is not None
    }

    features = {}
    coefficient_tables = []
    alphas = {}

    for mode, alpha_scale in alpha_scales.items():
        if alpha_scale == 1.0:
            selection = cv_selection
        else:
            selection = select_features_at_alpha(
                X_train,
                y_train,
                alpha=cv_selection.cv_alpha * alpha_scale,
                l1_ratio=cv_selection.l1_ratio,
                min_abs_coef=MIN_ABS_COEF,
                max_features=MAX_FEATURES,
            )

        features[mode] = selection.selected_features
        alphas[mode] = selection.alpha

        table = selection.coefficients.copy()
        table.insert(0, "selection_mode", mode)
        table.insert(0, "split_name", split["name"])
        table.insert(0, "split_index", split_index)
        table.insert(3, "alpha", selection.alpha)
        table.insert(4, "l1_ratio", selection.l1_ratio)
        coefficient_tables.append(table)

    features["none"] = list(X_train.columns)
    alphas["none"] = None

    print(f"Split {split_index}: {split['name']}")
    print(f"  Training rows: {len(X_train)} (train_ratio={TRAIN_RATIO:g})")
    print(
        f"  CV-selected alpha: {cv_selection.cv_alpha:.6g}, "
        f"rho: {cv_selection.l1_ratio:.3g}, "
        f"CV RMSE: {cv_selection.cv_rmse:.6g}"
    )
    for mode in SELECTION_MODES:
        print(f"  {mode:<10} {len(features[mode]):>3} features")
    print()

    split_payload = {
        "split_index": split_index,
        "split_name": split["name"],
        "train_period": list(split["train"]),
        "test_period": list(split["test"]),
        "train_ratio": TRAIN_RATIO,
        "feature_set": FEATURE_SET_NAME,
        "cv_folds": CV_SPLITS,
        "n_training_rows": len(X_train),
        "n_features_total": cv_selection.n_features,
        "cv_alpha": cv_selection.cv_alpha,
        "l1_ratio": cv_selection.l1_ratio,
        "rho": cv_selection.l1_ratio,
        "cv_rmse": cv_selection.cv_rmse,
        "cv_rmse_std": cv_selection.cv_rmse_std,
        "cv_fold_rmse": cv_selection.cv_fold_rmse,
        "alphas": alphas,
        "mode_alpha_scales": {
            mode: settings["alpha_scale"] for mode, settings in SELECTION_MODES.items()
        },
        "mode_feature_counts": {
            mode: len(mode_features) for mode, mode_features in features.items()
        },
        "features": features,
    }

    return split_payload, coefficient_tables


def main():
    if not 0 < TRAIN_RATIO <= 1:
        raise ValueError("TRAIN_RATIO must be in (0, 1].")

    for mode, settings in SELECTION_MODES.items():
        alpha_scale = settings["alpha_scale"]
        if alpha_scale is not None and alpha_scale <= 0:
            raise ValueError(f"Alpha scale for mode {mode!r} must be greater than 0.")

    if SPLIT_INDICES is None:
        split_indices = list(range(len(SEASONAL_SPLITS)))
    else:
        split_indices = sorted(set(SPLIT_INDICES))

    splits = {}
    coefficient_tables = []

    for split_index in split_indices:
        split_payload, tables = select_for_split(split_index)
        splits[str(split_index)] = split_payload
        coefficient_tables.extend(tables)

    modes = {
        mode: dict(settings)
        for mode, settings in SELECTION_MODES.items()
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(DATASET_PATH),
        "feature_set": FEATURE_SET_NAME,
        "train_ratio": TRAIN_RATIO,
        "cv_method": (
            f"expanding-window TimeSeriesSplit({CV_SPLITS}) "
            "with fold-wise preprocessing"
        ),
        "cv_folds": CV_SPLITS,
        "rho_candidates": list(L1_RATIOS),
        "n_alphas": N_ALPHAS,
        "eps": EPS,
        "candidate_features": next(iter(splits.values()))["features"]["none"],
        "modes": modes,
        "splits": splits,
    }

    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    if coefficient_tables:
        coefficients = pd.concat(coefficient_tables, ignore_index=True)
        coefficients = coefficients.sort_values(
            ["split_index", "selection_mode", "abs_coefficient"],
            ascending=[True, True, False],
            kind="mergesort",
        ).reset_index(drop=True)
        COEFFICIENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        coefficients.to_csv(COEFFICIENTS_PATH, index=False)

    for split_payload in splits.values():
        print(split_payload["split_name"])
        for mode in SELECTION_MODES:
            selected = split_payload["features"][mode]
            print(f"  {mode} ({len(selected)} features):")
            for feature in selected[:SHOW]:
                print(f"    {feature}")
        print()

    print(f"Selected features written to {OUTPUT_PATH}")
    if coefficient_tables:
        print(f"Coefficient table written to {COEFFICIENTS_PATH}")


if __name__ == "__main__":
    main()

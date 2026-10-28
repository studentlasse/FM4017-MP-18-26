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

Example
-------
    # All splits, all three modes
    uv run python preprocessing/run_elasticnet_selection.py
"""

import argparse
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


def load_training_data(file_path, feature_set_name, split_index, train_ratio=0.85):
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


def select_for_split(args, split_index):
    """Run CV once and derive every selection mode for one split."""
    X_train, y_train, split = load_training_data(
        args.dataset,
        args.feature_set,
        split_index,
        args.train_ratio,
    )

    cv_selection = select_features_elasticnet(
        X_train,
        y_train,
        l1_ratios=args.l1_ratios,
        n_alphas=args.n_alphas,
        eps=args.eps,
        n_splits=args.cv_splits,
        min_abs_coef=args.min_abs_coef,
        max_features=args.max_features,
        alpha_scale=1.0,
    )

    alpha_scales = {
        "aggressive": args.aggressive_alpha_scale,
        "moderate": args.moderate_alpha_scale,
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
                min_abs_coef=args.min_abs_coef,
                max_features=args.max_features,
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
    print(f"  Training rows: {len(X_train)} (train_ratio={args.train_ratio:g})")
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
        "train_ratio": args.train_ratio,
        "feature_set": args.feature_set,
        "cv_folds": args.cv_splits,
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
            "aggressive": args.aggressive_alpha_scale,
            "moderate": args.moderate_alpha_scale,
            "none": None,
        },
        "mode_feature_counts": {
            mode: len(mode_features) for mode, mode_features in features.items()
        },
        "features": features,
    }

    return split_payload, coefficient_tables


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select aggressive, moderate, and no-selection feature sets with "
            "ElasticNet for every seasonal split."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help=f"Path to the ML dataset (default: {DATASET_PATH.name}).",
    )
    parser.add_argument(
        "--feature-set",
        default="all",
        choices=sorted(FEATURE_SETS) + ["all"],
        help=(
            "Candidate feature set: a key from feature_sets.py, or 'all' "
            "for every numeric column in the dataset (default: all)."
        ),
    )
    parser.add_argument(
        "--split-index",
        type=int,
        nargs="+",
        default=None,
        choices=range(len(SEASONAL_SPLITS)),
        help=(
            "Split indices to run. Omit to run every seasonal split. When a "
            "subset is given, the output file is replaced with only those "
            "splits."
        ),
    )
    parser.add_argument(
        "--l1-ratios",
        default=",".join(f"{value:g}" for value in DEFAULT_L1_RATIOS),
        help=(
            "Comma-separated rho values tried by ElasticNetCV "
            "(default: 0.05 to 1.0 in 20 steps)."
        ),
    )
    parser.add_argument(
        "--n-alphas",
        type=int,
        default=100,
        help="Number of regularization strengths on the path (default: 100).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-3,
        help=(
            "Ratio between the smallest and largest alpha on the "
            "log-spaced path (default: 1e-3)."
        ),
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of TimeSeriesSplit folds (default: 5).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.85,
        help=(
            "Fraction of the training window used for selection. Use 0.85 "
            "to match the model scripts' 15%% validation hold-out "
            "(default: 0.85)."
        ),
    )
    parser.add_argument(
        "--min-abs-coef",
        type=float,
        default=1e-8,
        help="Minimum absolute coefficient for a feature to be kept.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Keep at most this many features, ranked by absolute coefficient.",
    )
    parser.add_argument(
        "--aggressive-alpha-scale",
        type=float,
        default=SELECTION_MODES["aggressive"]["alpha_scale"],
        help=(
            "Multiplier on the CV-selected alpha for the aggressive set (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--moderate-alpha-scale",
        type=float,
        default=SELECTION_MODES["moderate"]["alpha_scale"],
        help=(
            "Multiplier on the CV-selected alpha for the moderate set "
            "(default: 0.2; smaller keeps more features)."
        ),
    )
    parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="Number of top features to print per mode (default: 20).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Selected feature JSON (default: {OUTPUT_PATH.name}).",
    )
    parser.add_argument(
        "--coefficients-output",
        type=Path,
        default=COEFFICIENTS_PATH,
        help=f"Coefficient CSV (default: {COEFFICIENTS_PATH.name}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.l1_ratios = [float(value) for value in args.l1_ratios.split(",") if value]

    if args.aggressive_alpha_scale <= 0 or args.moderate_alpha_scale <= 0:
        raise ValueError("Alpha scales must be greater than 0.")

    if args.split_index is None:
        split_indices = list(range(len(SEASONAL_SPLITS)))
    else:
        split_indices = sorted(set(args.split_index))

    splits = {}
    coefficient_tables = []

    for split_index in split_indices:
        split_payload, tables = select_for_split(args, split_index)
        splits[str(split_index)] = split_payload
        coefficient_tables.extend(tables)

    modes = {
        "aggressive": {
            "alpha_scale": args.aggressive_alpha_scale,
            "description": SELECTION_MODES["aggressive"]["description"],
        },
        "moderate": {
            "alpha_scale": args.moderate_alpha_scale,
            "description": SELECTION_MODES["moderate"]["description"],
        },
        "none": {
            "alpha_scale": None,
            "description": SELECTION_MODES["none"]["description"],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "feature_set": args.feature_set,
        "train_ratio": args.train_ratio,
        "cv_method": (
            f"expanding-window TimeSeriesSplit({args.cv_splits}) "
            "with fold-wise preprocessing"
        ),
        "cv_folds": args.cv_splits,
        "rho_candidates": args.l1_ratios,
        "n_alphas": args.n_alphas,
        "eps": args.eps,
        "candidate_features": next(iter(splits.values()))["features"]["none"],
        "modes": modes,
        "splits": splits,
    }

    args.output.write_text(
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
        args.coefficients_output.parent.mkdir(parents=True, exist_ok=True)
        coefficients.to_csv(args.coefficients_output, index=False)

    for split_payload in splits.values():
        print(split_payload["split_name"])
        for mode in SELECTION_MODES:
            selected = split_payload["features"][mode]
            print(f"  {mode} ({len(selected)} features):")
            for feature in selected[: args.show]:
                print(f"    {feature}")
        print()

    print(f"Selected features written to {args.output}")
    if coefficient_tables:
        print(f"Coefficient table written to {args.coefficients_output}")


if __name__ == "__main__":
    main()

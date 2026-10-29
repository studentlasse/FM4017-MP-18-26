import json
from pathlib import Path

FEATURE_SETS = {
    "full_set": [
        "hour",
        "price_value",
        "buy_volume_value",
        "sell_volume_value",
        "net_volume",
        "flow_DK1_export",
        "flow_DK1_import",
        "flow_GER_export",
        "flow_GER_import",
        "flow_NL_export",
        "flow_NL_import",
        "flow_NO1_export",
        "flow_NO1_import",
        "flow_NO5_export",
        "flow_NO5_import",
        "capacity_NO5",
        "util_NO5",
        "price_DK1",
        "price_NO1",
        "price_NO5",
        "price_NL",
        "price_DE_LU",
        "price_BE",
        "price_FR",
        "total_exports",
        "total_imports",
        "net_flow",
        "day_of_week",
        "month",
        "day_of_year",
        "day_sin",
        "day_cos",
        "year_sin",
        "year_cos",
        "price_lag_24",
        "price_lag_168",
        "price_DK1_lag24",
        "price_NO1_lag24",
        "price_NO5_lag24",
        "price_DK1_lag168",
        "price_NO1_lag168",
        "price_NO5_lag168",
    ],
    "lstm_set": [
        "hour",
        "price_value",
        "buy_volume_value",
        "sell_volume_value",
        "net_volume",
        "flow_DK1_export",
        "flow_DK1_import",
        "flow_GER_export",
        "flow_GER_import",
        "flow_NL_export",
        "flow_NL_import",
        "flow_NO1_export",
        "flow_NO1_import",
        "flow_NO5_export",
        "flow_NO5_import",
        "price_DK1",
        "price_NO1",
        "price_NO5",
        "price_NL",
        "price_DE_LU",
        "price_BE",
        "price_FR",
        "total_exports",
        "total_imports",
        "net_flow",
        "day_of_week",
        "month",
        "day_of_year",
        "day_sin",
        "day_cos",
        "year_sin",
        "year_cos",
        "price_lag_24",
        "price_lag_168",
        "price_DK1_lag24",
        "price_NO1_lag24",
        "price_NO5_lag24",
        "price_DK1_lag168",
        "price_NO1_lag168",
        "price_NO5_lag168",
    ],
    "minimal_set": [
        "price_value",
        "price_lag_24",
        "price_lag_168",
        "day_sin",
        "day_cos",
    ],
    "minimal_no_price": [
        "price_lag_24",
        "price_lag_168",
        "day_sin",
        "day_cos",
    ],
    "market_prices": [
        "price_value",
        "price_lag_24",
        "price_lag_168",
        "price_DK1",
        "price_NO1",
        "price_NO5",
        "price_NL",
        "price_DE_LU",
        "price_BE",
        "price_FR",
        "day_sin",
        "day_cos",
    ],
    # "reservoir": [
    #     "price_value",
    #     "price_lag_24",
    #     "price_lag_168",
    #     "no2_reservoir_level_MWh",
    #     "no1_reservoir_level_MWh",
    #     "no5_reservoir_level_MWh",
    #     "day_sin",
    #     "day_cos",
    # ],
    "renewables": [
        "price_value",
        "price_lag_24",
        "price_lag_168",
        "Solar",
        "Wind Onshore",
        "Wind Offshore",
        "Forecasted Generation",
        "Forecasted Load",
        "day_sin",
        "day_cos",
    ],
}


SELECTION_MODES = ("aggressive", "moderate", "none")
SELECTION_PATH = Path(__file__).resolve().parent / "elasticnet_selected_features.json"


def _resolve_selection_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def load_selection(path=SELECTION_PATH):
    """Load the combined selection JSON written by the ElasticNet runner."""
    path = _resolve_selection_path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Feature selection file not found: {path}. Generate it with:\n"
            "  uv run python preprocessing/run_elasticnet_selection.py"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict) or "splits" not in payload:
        raise ValueError(
            f"{path} does not contain per-split selections. Regenerate it with:\n"
            "  uv run python preprocessing/run_elasticnet_selection.py"
        )

    return payload


def _get_split_selection(payload, split_index):
    key = str(split_index)
    splits = payload["splits"]

    if key not in splits:
        available = ", ".join(sorted(splits, key=int)) or "none"
        raise KeyError(
            f"Split {split_index} is not present in the selection file "
            f"(available: {available}). Regenerate it with:\n"
            "  uv run python preprocessing/run_elasticnet_selection.py"
        )

    return splits[key]


def load_feature_set(split_index, mode, path=SELECTION_PATH):
    """Return the selected features for one seasonal split and mode.

    ``mode`` is one of ``aggressive``, ``moderate``, or ``none``. The selection
    file is produced by ``preprocessing/run_elasticnet_selection.py``.
    """
    if mode not in SELECTION_MODES:
        raise ValueError(
            f"Unknown selection mode {mode!r}. Choose from {SELECTION_MODES}."
        )

    payload = load_selection(path)
    split = _get_split_selection(payload, split_index)

    try:
        features = split["features"][mode]
    except KeyError as error:
        available = ", ".join(split.get("features", {}))
        raise KeyError(
            f"Split {split_index} has no {mode!r} feature set "
            f"(available: {available}). Regenerate the selection file."
        ) from error

    if not features:
        raise ValueError(
            f"The {mode!r} set for split {split_index} is empty. "
            "Use a larger alpha scale or a richer candidate set."
        )

    return list(features)


def load_feature_set_metadata(split_index, mode, path=SELECTION_PATH):
    """Return metadata for one selected feature set."""
    payload = load_selection(path)
    split = _get_split_selection(payload, split_index)
    mode_info = payload.get("modes", {}).get(mode, {})
    features = split.get("features", {}).get(mode, [])

    return {
        "selection_path": str(_resolve_selection_path(path)),
        "generated_at": payload.get("generated_at"),
        "split_index": split.get("split_index", split_index),
        "split_name": split.get("split_name"),
        "train_period": split.get("train_period"),
        "test_period": split.get("test_period"),
        "train_ratio": payload.get("train_ratio"),
        "selection_mode": mode,
        "alpha_scale": mode_info.get("alpha_scale"),
        "algorithm_alpha": split.get("alphas", {}).get(mode),
        "cv_alpha": split.get("cv_alpha"),
        "l1_ratio": split.get("l1_ratio"),
        "cv_rmse": split.get("cv_rmse"),
        "n_features_total": split.get("n_features_total"),
        "n_features_selected": len(features),
    }

#!/usr/bin/env python
"""
Energy-aware model selection from existing tuning results.

Two levels of selection are produced:

1. MEASURED selection (cross-experiment)
   Each row in ``results/*_tuning_results.csv`` is the validation-best
   configuration of one model/split/selection-mode/search-method run, retrained
   and measured with CodeCarbon. We can therefore compare these rows on
   measured training, inference and search energy.

2. PER-CONFIGURATION selection (within a search)
   ``results/*_search_results_split*.csv`` contain one row per configuration.
   Newer files include ``config_energy_kwh``: the measured energy of training
   and validating that single configuration. When present it is used directly.
   Older files only contain ``tuning_energy_kwh``, the *total* search energy
   repeated on every row, so those configurations fall back to a documented
   complexity proxy. Each output row records which source was used in
   ``selection_energy_source``.

Usage:
    uv run python hyperparameter-pipeline-test/energy_aware_selection.py
    uv run python hyperparameter-pipeline-test/energy_aware_selection.py --tolerance 0.02
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

TUNING_FILES = {
    "gru": "gru_tuning_results.csv",
    "lstm": "lstm_tuning_results.csv",
    "rf": "rf_tuning_results.csv",
    "xgboost": "xgboost_tuning_results.csv",
}

SEARCH_PATTERN = re.compile(
    r"^(gru|lstm|rf|xgboost)_(grid|random)_search_results_split(\d)_(\w+)\.csv$"
)

# New per-config energy column, added after search_energy_kwh in the tuners.
NEW_TUNING_COLUMN = "best_config_energy_kwh"
NEW_TUNING_COLUMN_ANCHOR = "search_energy_kwh"

TEXT_COLUMNS = {
    "timestamp",
    "split_name",
    "selection_mode",
    "data_filter",
    "feature_columns",
    "search_method",
    "description",
    "model",
    "best_param_loss_name",
    "best_param_max_features",
}

SEARCH_COLUMNS = [
    "model",
    "search_method",
    "split_index",
    "selection_mode",
    "source_file",
    "validation_rmse",
    "config_energy_kwh",
    "cost_proxy",
    "energy_for_selection",
    "energy_source",
]

# Configurations within this fraction of the best score are treated as
# equivalent on prediction quality, and the cheapest one is selected.
DEFAULT_TOLERANCES = (0.01, 0.02, 0.05, 0.10)

# The measured Pareto plots are restricted to the summer split (split 1).
SUMMER_SPLIT_INDEX = 1
SUMMER_SPLIT_TITLE = "Summer 2025 (split 1)"


def _read_tuning_csv(path):
    """Read a tuning CSV even when rows contain the newer ``best_config_energy_kwh``.

    The tuners append rows to an existing CSV. When the new per-config energy
    column was introduced, older files kept their old header while new rows
    contained one extra field, which makes ``pd.read_csv`` fail. Rebuild the
    header here so both old and new rows can be read without editing the file.
    """
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame()

        # Very old LSTM files had no header and started with a stray "1".
        if not header or header[0].strip() != "timestamp":
            columns = pd.read_csv(
                RESULTS_DIR / TUNING_FILES["gru"], nrows=0
            ).columns.tolist()
            return pd.read_csv(path, skiprows=1, header=None, names=columns)

        rows = [row for row in reader if row]

    original_header = list(header)
    if (
        NEW_TUNING_COLUMN not in header
        and NEW_TUNING_COLUMN_ANCHOR in header
        and any(len(row) == len(header) + 1 for row in rows)
    ):
        insert_at = header.index(NEW_TUNING_COLUMN_ANCHOR) + 1
        header.insert(insert_at, NEW_TUNING_COLUMN)

    if header != original_header:
        # Old rows lack the new field; new rows already contain it.
        aligned_rows = []
        for row in rows:
            if len(row) == len(original_header):
                row.insert(insert_at, "")
            aligned_rows.append(row)
        rows = aligned_rows

    # Handle any remaining ragged rows by padding them to the header width.
    width = max([len(header), *(len(row) for row in rows)], default=len(header))
    if width > len(header):
        header.extend(f"extra_{index}" for index in range(len(header), width))
    rows = [row + [""] * (len(header) - len(row)) for row in rows]

    frame = pd.DataFrame(rows, columns=header)
    # Older LSTM files misspelled test_rmse_std.
    frame = frame.rename(columns={"test_rmsd_std": "test_rmse_std"})

    # csv.reader returns strings, so restore numeric dtypes for analysis.
    for column in frame.columns:
        if column not in TEXT_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def load_measured_results(results_dir):
    """Load one measured row per model/split/mode/search-method combination."""
    frames = []
    for model, filename in TUNING_FILES.items():
        path = results_dir / filename
        if not path.exists():
            print(f"[warn] missing {path}")
            continue

        frame = _read_tuning_csv(path)
        frame["model"] = model
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No tuning result files found in {results_dir}")

    results = pd.concat(frames, ignore_index=True)
    results["split_index"] = results["split_index"].astype(int)
    results["key"] = (
        results["model"]
        + "_split"
        + results["split_index"].astype(str)
        + "_"
        + results["selection_mode"]
        + "_"
        + results["search_method"]
    )

    # GRU was tuned twice; keep the latest run for each combination.
    results = (
        results.sort_values("timestamp")
        .drop_duplicates("key", keep="last")
        .reset_index(drop=True)
    )

    # Energy definitions:
    # - deployment: train once + run inference once
    # - experiment: search + all retraining runs + all inference runs
    results["deployment_energy_kwh"] = (
        results["training_energy_kwh_per_run"] + results["test_energy_kwh_per_run"]
    )
    results["experiment_energy_kwh"] = (
        results["search_energy_kwh"]
        + results["training_energy_kwh_total"]
        + results["test_energy_kwh_total"]
    )
    return results


def load_search_results(results_dir):
    """Load per-configuration validation results from the search CSVs."""
    frames = []
    for path in sorted(results_dir.glob("*_search_results_split*.csv")):
        match = SEARCH_PATTERN.match(path.name)
        if not match:
            print(f"[warn] skipping unrecognised file {path.name}")
            continue

        model, method, split_index, selection_mode = match.groups()
        frame = pd.read_csv(path)
        frame["model"] = model
        frame["search_method"] = method
        frame["split_index"] = int(split_index)
        frame["selection_mode"] = selection_mode
        frame["source_file"] = path.name
        frames.append(frame)

    if not frames:
        print(
            f"[warn] no per-configuration search files found in {results_dir}; "
            "skipping per-configuration selection"
        )
        return pd.DataFrame(columns=SEARCH_COLUMNS)

    return pd.concat(frames, ignore_index=True)


def add_cost_proxy(search):
    """Attach a per-config energy column for within-search selection.

    Older search CSVs only contain ``tuning_energy_kwh``, which is the *total*
    search energy repeated on every row. Newer CSVs (produced after per-config
    energy logging was added) also contain ``config_energy_kwh``: the measured
    energy of training and validating that single configuration.

    ``energy_for_selection`` prefers the measured value and falls back to a
    complexity proxy. The proxy is only valid for ranking configurations
    *within* one model, split, selection mode and search method; it is not
    calibrated to kWh.
    """
    search = search.copy()
    search["cost_proxy"] = np.nan

    for model, group in search.groupby("model"):
        if model in ("gru", "lstm"):
            # Recurrent work scales roughly with layers * hidden^2 * epochs.
            search.loc[group.index, "cost_proxy"] = (
                group["num_layers"] * group["hidden_size"] ** 2 * group["best_epoch"]
            )
        elif model == "rf":
            # Tree work scales with trees, depth and feature subsampling.
            search.loc[group.index, "cost_proxy"] = (
                group["n_estimators"]
                * (2.0 ** group["max_depth"])
                * group["max_features"]
            )
        elif model == "xgboost":
            search.loc[group.index, "cost_proxy"] = (
                group["n_estimators"]
                * group["max_depth"]
                * group["subsample"]
                * group["colsample_bytree"]
            )

    if "config_energy_kwh" in search.columns:
        measured = pd.to_numeric(search["config_energy_kwh"], errors="coerce")
    else:
        measured = pd.Series(np.nan, index=search.index, dtype=float)

    search["energy_for_selection"] = measured.where(measured.notna(), search["cost_proxy"])
    search["energy_source"] = np.where(measured.notna(), "measured", "proxy")

    return search


def pareto_front(frame, performance_column, energy_column):
    """Return rows not dominated on (lower performance, lower energy)."""
    values = frame[[performance_column, energy_column]].to_numpy(dtype=float)
    keep = np.ones(len(frame), dtype=bool)

    for i in range(len(frame)):
        if not keep[i]:
            continue
        for j in range(len(frame)):
            if i == j:
                continue
            dominates = np.all(values[j] <= values[i]) and np.any(
                values[j] < values[i]
            )
            if dominates:
                keep[i] = False
                break

    return frame.loc[keep].sort_values(performance_column)


def epsilon_constraint_selection(frame, group_columns, performance_column, energy_column, tolerances):
    """Pick the cheapest row whose score is within a tolerance of the best."""
    rows = []
    for keys, group in frame.groupby(group_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)

        best_score = group[performance_column].min()
        best_row = group.loc[group[performance_column].idxmin()]

        for tolerance in tolerances:
            eligible = group[group[performance_column] <= best_score * (1.0 + tolerance)]
            if eligible.empty:
                continue
            selected = eligible.loc[eligible[energy_column].idxmin()]

            row = dict(zip(group_columns, keys))
            row.update(
                {
                    "tolerance_pct": tolerance * 100,
                    "best_score": best_score,
                    "selected_score": selected[performance_column],
                    "score_increase_pct": (
                        selected[performance_column] / best_score - 1.0
                    )
                    * 100,
                    "best_model": best_row["model"],
                    "best_search_method": best_row["search_method"],
                    "selected_model": selected["model"],
                    "selected_search_method": selected["search_method"],
                    "best_energy_kwh": best_row[energy_column],
                    "selected_energy_kwh": selected[energy_column],
                    "energy_saving_factor": best_row[energy_column]
                    / selected[energy_column]
                    if selected[energy_column] > 0
                    else np.nan,
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def describe_config(row):
    model = row["model"]
    if model in ("gru", "lstm"):
        return (
            f"hidden={int(row['hidden_size'])}, layers={int(row['num_layers'])}, "
            f"lr={row['learning_rate']}, epoch={int(row['best_epoch'])}"
        )
    if model == "rf":
        return (
            f"trees={int(row['n_estimators'])}, depth={int(row['max_depth'])}, "
            f"split={int(row['min_samples_split'])}, leaf={int(row['min_samples_leaf'])}, "
            f"max_features={row['max_features']}"
        )
    return (
        f"trees={int(row['n_estimators'])}, depth={int(row['max_depth'])}, "
        f"lr={row['learning_rate']}, subsample={row['subsample']}, "
        f"colsample={row['colsample_bytree']}, min_child_weight={int(row['min_child_weight'])}"
    )


def proxy_config_selection(search, tolerances):
    """Pick the cheapest configuration within a validation-RMSE tolerance.

    Uses measured per-config energy (``config_energy_kwh``) when the search CSV
    provides it, otherwise a complexity proxy. ``selection_energy_source``
    records which one was used for each row.
    """
    rows = []
    groups = ["model", "split_index", "selection_mode", "search_method"]

    for keys, group in search.groupby(groups):
        best_score = group["validation_rmse"].min()
        best_row = group.loc[group["validation_rmse"].idxmin()]
        group_has_measured = bool((group["energy_source"] == "measured").any())

        for tolerance in tolerances:
            eligible = group[group["validation_rmse"] <= best_score * (1.0 + tolerance)]
            selected = eligible.loc[eligible["energy_for_selection"].idxmin()]

            row = dict(zip(groups, keys))
            row.update(
                {
                    "tolerance_pct": tolerance * 100,
                    "best_validation_rmse": best_score,
                    "selected_validation_rmse": selected["validation_rmse"],
                    "validation_increase_pct": (
                        selected["validation_rmse"] / best_score - 1.0
                    )
                    * 100,
                    "best_config": describe_config(best_row),
                    "selected_config": describe_config(selected),
                    "best_energy_for_selection": best_row["energy_for_selection"],
                    "selected_energy_for_selection": selected[
                        "energy_for_selection"
                    ],
                    "energy_reduction_factor": best_row["energy_for_selection"]
                    / selected["energy_for_selection"]
                    if selected["energy_for_selection"] > 0
                    else np.nan,
                    "best_cost_proxy": best_row["cost_proxy"],
                    "selected_cost_proxy": selected["cost_proxy"],
                    "cost_proxy_reduction_factor": best_row["cost_proxy"]
                    / selected["cost_proxy"],
                    "selection_energy_source": selected["energy_source"],
                    "group_has_measured_energy": group_has_measured,
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def build_config_comparison(search, tolerances):
    """Compare validation-best params with energy-optimized params.

    Both parameter sets come from the same search. The normal-best set is the
    row with the lowest validation RMSE. The energy-optimized set is the
    cheapest row within each relative validation-RMSE tolerance.
    """
    rows = []
    groups = ["model", "split_index", "selection_mode", "search_method"]

    for keys, group in search.groupby(groups):
        best_score = group["validation_rmse"].min()
        best_row = group.loc[group["validation_rmse"].idxmin()]
        group_has_measured = bool((group["energy_source"] == "measured").any())

        for tolerance in tolerances:
            eligible = group[group["validation_rmse"] <= best_score * (1.0 + tolerance)]
            selected = eligible.loc[eligible["energy_for_selection"].idxmin()]

            selected_energy = selected["energy_for_selection"]
            best_energy = best_row["energy_for_selection"]
            configs_differ = describe_config(selected) != describe_config(best_row)

            row = dict(zip(groups, keys))
            row.update(
                {
                    "tolerance_pct": tolerance * 100,
                    "normal_best_config": describe_config(best_row),
                    "normal_best_validation_rmse": best_score,
                    "normal_best_energy": best_energy,
                    "energy_optimized_config": describe_config(selected),
                    "energy_optimized_validation_rmse": selected["validation_rmse"],
                    "energy_optimized_energy": selected_energy,
                    "configs_differ": configs_differ,
                    "validation_rmse_increase_pct": (
                        selected["validation_rmse"] / best_score - 1.0
                    )
                    * 100,
                    "energy_reduction_factor": (
                        best_energy / selected_energy
                        if selected_energy > 0
                        else np.nan
                    ),
                    "selection_energy_source": selected["energy_source"],
                    "group_has_measured_energy": group_has_measured,
                    "comparison_scope": (
                        "measured per-config energy"
                        if selected["energy_source"] == "measured"
                        else "validation metrics only; energy is a proxy"
                    ),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


PERFORMANCE_DISPLAY = {
    "validation_rmse": "Validation RMSE",
    "test_rmse": "Test RMSE",
}

ENERGY_DISPLAY = {
    "experiment": "Lifecycle energy (search + retraining + inference)",
    "deployment": "Deployment energy (one retraining + inference)",
    "search": "Search energy",
    "training": "Training energy per run",
    "inference": "Inference energy per run",
}


def _metric_display_name(column):
    return PERFORMANCE_DISPLAY.get(
        column, column.replace("_", " ").title().replace("Rmse", "RMSE")
    )


def _energy_display_name(energy_basis):
    return ENERGY_DISPLAY.get(energy_basis, f"{energy_basis.capitalize()} energy")


def _annotate_front_points(
    axis, front, energy_column, performance_column, model_short
):
    """Label front points, moving each label until it does not overlap another."""
    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_bbox = axis.get_window_extent(renderer=renderer)

    offsets = [
        (11, 12),
        (11, -19),
        (-11, 12),
        (-11, -19),
        (11, 34),
        (11, -41),
        (-11, 34),
        (-11, -41),
        (38, 0),
        (-38, 0),
        (0, 24),
        (0, -30),
    ]
    label_style = dict(
        fontsize=11,
        zorder=6,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="none",
            alpha=0.82,
        ),
    )

    occupied = []
    for _, row in front.iterrows():
        label = (
            f"{model_short.get(row['model'], row['model'])} "
            f"{row['search_method']} ({row['selection_mode']})"
        )
        point = (row[energy_column], row[performance_column])

        placed = None
        for dx, dy in offsets:
            arrowprops = (
                dict(
                    arrowstyle="-",
                    color="0.45",
                    linewidth=0.8,
                    shrinkA=2,
                    shrinkB=4,
                )
                if abs(dx) >= 20 or abs(dy) >= 24
                else None
            )
            text = axis.annotate(
                label,
                point,
                textcoords="offset points",
                xytext=(dx, dy),
                ha="left" if dx >= 0 else "right",
                va="center" if dy == 0 else ("bottom" if dy > 0 else "top"),
                arrowprops=arrowprops,
                **label_style,
            )
            bbox = text.get_window_extent(renderer=renderer).expanded(1.06, 1.18)
            inside_axes = (
                axes_bbox.x0 <= bbox.x0
                and bbox.x1 <= axes_bbox.x1
                and axes_bbox.y0 <= bbox.y0
                and bbox.y1 <= axes_bbox.y1
            )
            if inside_axes and not any(bbox.overlaps(other) for other in occupied):
                placed = text
                occupied.append(bbox)
                break
            text.remove()

        if placed is None:
            # No free spot: keep the first candidate rather than dropping a label.
            dx, dy = offsets[0]
            placed = axis.annotate(
                label,
                point,
                textcoords="offset points",
                xytext=(dx, dy),
                ha="left",
                va="bottom",
                **label_style,
            )
            occupied.append(
                placed.get_window_extent(renderer=renderer).expanded(1.06, 1.18)
            )

    return occupied


def plot_measured_pareto(
    measured, output_dir, performance_column, energy_column, energy_label
):
    """Plot measured performance against the chosen energy basis for summer only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter

    output_dir.mkdir(parents=True, exist_ok=True)

    measured = measured[measured["split_index"] == SUMMER_SPLIT_INDEX]
    if measured.empty:
        raise ValueError(
            f"No measured rows for summer split {SUMMER_SPLIT_INDEX}"
        )

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 13,
            "figure.titlesize": 22,
        }
    )

    metric_label = _metric_display_name(performance_column)
    energy_display = _energy_display_name(energy_label)
    x_label = f"{energy_display} [kWh]"
    y_label = f"{metric_label} (lower is better)"

    model_colors = {
        "gru": "#7b5aa6",
        "lstm": "#2b6cb0",
        "rf": "#2f855a",
        "xgboost": "#c53030",
    }
    model_labels = {
        "gru": "GRU",
        "lstm": "LSTM",
        "rf": "Random Forest",
        "xgboost": "XGBoost",
    }
    method_markers = {"grid": "o", "random": "^"}

    fig, axis = plt.subplots(figsize=(18, 12))

    model_short = {
        "gru": "GRU",
        "lstm": "LSTM",
        "rf": "RF",
        "xgboost": "XGB",
    }

    for model, group in measured.groupby("model"):
        for method, method_group in group.groupby("search_method"):
            axis.scatter(
                method_group[energy_column],
                method_group[performance_column],
                s=180,
                color=model_colors.get(model, "grey"),
                marker=method_markers.get(method, "o"),
                edgecolor="black",
                linewidth=1.0,
                alpha=0.85,
                label=f"{model_labels.get(model, model)} ({method})",
                zorder=3,
            )

    front = pareto_front(measured, performance_column, energy_column)
    axis.plot(
        front[energy_column],
        front[performance_column],
        color="black",
        linestyle="--",
        linewidth=2.5,
        marker="o",
        markersize=8,
        zorder=2,
        label="Pareto front",
    )

    axis.set_xscale("log")
    axis.xaxis.set_major_locator(
        LogLocator(base=10, subs=(1.0, 5.0), numticks=100)
    )
    axis.xaxis.set_minor_locator(
        LogLocator(base=10, subs="auto", numticks=100)
    )
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_title(SUMMER_SPLIT_TITLE, pad=12)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, which="major", alpha=0.35)
    axis.grid(True, which="minor", alpha=0.15)
    axis.set_axisbelow(True)
    axis.tick_params(which="major", length=8, width=1.5)
    axis.tick_params(which="minor", length=4, width=1.0)

    # Draw once with all elements in place, then place front labels.
    fig.canvas.draw()
    _annotate_front_points(
        axis,
        front,
        energy_column,
        performance_column,
        model_short,
    )

    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=5,
        fontsize=15,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        markerscale=1.5,
    )
    fig.suptitle(
        f"Energy-aware selection: {metric_label} vs. {energy_label} energy "
        "(summer 2025)",
        fontsize=22,
        y=1.03,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    by_split_path = output_dir / "energy_aware_pareto_by_split.png"
    fig.savefig(by_split_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Overall view: all models together, still restricted to the summer split.
    fig, axis = plt.subplots(figsize=(18, 12))

    for model, group in measured.groupby("model"):
        color = model_colors.get(model, "grey")
        label_base = model_labels.get(model, model)
        axis.scatter(
            group[energy_column],
            group[performance_column],
            s=200,
            color=color,
            alpha=0.35,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        model_front = pareto_front(group, performance_column, energy_column)
        axis.plot(
            model_front[energy_column],
            model_front[performance_column],
            color=color,
            linestyle="--",
            linewidth=2.5,
            marker="o",
            markersize=8,
            zorder=2,
            label=f"{label_base} Pareto front",
        )

    axis.set_xscale("log")
    axis.xaxis.set_major_locator(
        LogLocator(base=10, subs=(1.0, 5.0), numticks=100)
    )
    axis.xaxis.set_minor_locator(LogLocator(base=10, subs="auto", numticks=100))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_title("Summer 2025 Pareto fronts by model", pad=16)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, which="major", alpha=0.35)
    axis.grid(True, which="minor", alpha=0.15)
    axis.set_axisbelow(True)
    axis.tick_params(which="major", length=8, width=1.5)
    axis.tick_params(which="minor", length=4, width=1.0)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=15,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        markerscale=1.5,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    overall_path = output_dir / "energy_aware_pareto_overall.png"
    fig.savefig(overall_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return by_split_path, overall_path


def plot_all_search_configs(
    search, output_dir, performance_column="validation_rmse", energy_column="energy_for_selection"
):
    """Plot all per-configuration search results with Pareto fronts per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 17,
        }
    )

    model_colors = {
        "gru": "#7b5aa6",
        "lstm": "#2b6cb0",
        "rf": "#2f855a",
        "xgboost": "#c53030",
    }
    model_labels = {
        "gru": "GRU",
        "lstm": "LSTM",
        "rf": "Random Forest",
        "xgboost": "XGBoost",
    }
    method_markers = {"grid": "o", "random": "^"}
    split_markers = {0: "s", 1: "o", 2: "^"}

    metric_label = _metric_display_name(performance_column)
    energy_display = _energy_display_name("per-config")
    x_label = f"Per-config {energy_display} [kWh]"
    y_label = f"{metric_label} (lower is better)"

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=False, sharey=False)
    axes_flat = axes.ravel()

    for ax, (model, subset) in zip(axes_flat, search.groupby("model")):
        color = model_colors.get(model, "grey")
        label_base = model_labels.get(model, model)

        # All configs as scatter, colored by split, shaped by method
        for split_idx, split_group in subset.groupby("split_index"):
            for method, method_group in split_group.groupby("search_method"):
                ax.scatter(
                    method_group[energy_column],
                    method_group[performance_column],
                    s=25,
                    color=color,
                    marker=method_markers.get(method, "o"),
                    edgecolor="white",
                    linewidth=0.3,
                    alpha=0.5,
                    label=f"{label_base} split{split_idx} ({method})",
                    zorder=2,
                )

        # Pareto front for this model
        front = pareto_front(subset, performance_column, energy_column)
        ax.plot(
            front[energy_column],
            front[performance_column],
            color="black",
            linestyle="--",
            linewidth=1.8,
            marker="o",
            markersize=6,
            zorder=3,
            label=f"{label_base} Pareto front",
        )

        # Annotate front points
        fig.canvas.draw()
        _annotate_front_points(ax, front, energy_column, performance_column, {m: m.upper() for m in subset.model.unique()})

        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 5.0), numticks=100))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs="auto", numticks=100))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{label_base} ({len(subset)} configs)", pad=12)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.15)
        ax.set_axisbelow(True)
        ax.tick_params(which="major", length=5, width=1.0)
        ax.tick_params(which="minor", length=2.5, width=0.7)

    # Single legend for all subplots
    handles, labels = axes_flat[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=6,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        markerscale=1.3,
    )
    fig.suptitle(
        f"Per-configuration energy-aware search: {metric_label} vs. energy",
        fontsize=17,
        y=1.01,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = output_dir / "energy_aware_per_config_pareto.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Also create an OVERALL combined plot (all models together)
    fig, ax = plt.subplots(figsize=(14, 9))
    for model, subset in search.groupby("model"):
        color = model_colors.get(model, "grey")
        label_base = model_labels.get(model, model)
        front = pareto_front(subset, performance_column, energy_column)
        ax.plot(
            front[energy_column],
            front[performance_column],
            color=color,
            linestyle="--",
            linewidth=2.0,
            marker="o",
            markersize=7,
            zorder=3,
            label=f"{label_base} Pareto front",
        )
        # Scatter all points very faint
        ax.scatter(
            subset[energy_column],
            subset[performance_column],
            s=12,
            color=color,
            alpha=0.15,
            edgecolor="none",
            zorder=1,
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 5.0), numticks=100))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs="auto", numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title("Combined Pareto fronts: all models, all splits, all configs", pad=14)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15)
    ax.set_axisbelow(True)
    ax.tick_params(which="major", length=6, width=1.1)
    ax.tick_params(which="minor", length=3, width=0.8)
    ax.legend(fontsize=11, ncol=2, markerscale=1.2)
    fig.tight_layout()
    overall_path = output_dir / "energy_aware_per_config_pareto_combined.png"
    fig.savefig(overall_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return path, overall_path


def print_coverage(measured, search):
    expected = {
        f"{model}_split{split}_{mode}_{method}"
        for model in TUNING_FILES
        for split in (0, 1, 2)
        for mode in ("aggressive", "moderate", "none")
        for method in ("grid", "random")
    }
    have = set(measured["key"].astype(str))
    missing = sorted(expected - have)
    print(f"\nMeasured experiment rows: {len(measured)} / {len(expected)}")
    if missing:
        print("Missing experiment rows:")
        for key in missing:
            print(f"  - {key}")

    if search.empty:
        print("\nNo search result files loaded; per-configuration analysis skipped.")
        return

    print("\nSearch files loaded per model/method:")
    summary = (
        search.groupby(["model", "search_method"])
        .agg(
            files=("source_file", "nunique"),
            configs=("source_file", "size"),
            measured_configs=("energy_source", lambda s: int((s == "measured").sum())),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))

    measured_share = (search["energy_source"] == "measured").mean()
    print(
        f"\nPer-config measured energy available for {measured_share:.1%} of "
        "loaded configurations."
    )
    if measured_share < 1.0:
        print(
            "Remaining configurations use the complexity proxy. Re-run the "
            "tuner scripts to populate config_energy_kwh in the search CSVs."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--tolerance",
        type=float,
        nargs="+",
        default=list(DEFAULT_TOLERANCES),
        help="Relative performance tolerances, e.g. 0.01 0.02 0.05",
    )
    parser.add_argument(
        "--energy-basis",
        choices=("experiment", "deployment", "search", "training", "inference"),
        default="experiment",
        help="Energy column used for the measured selection",
    )
    parser.add_argument(
        "--performance",
        choices=("test_rmse", "validation_rmse"),
        default="validation_rmse",
        help=(
            "Performance column for the measured selection. validation_rmse "
            "(default) is the honest choice rule when using the results to "
            "pick a model; test_rmse is a retrospective comparison."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "energy_aware",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=PROJECT_ROOT / "plots",
        help="Directory for the generated trade-off plots",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the Pareto plots",
    )
    args = parser.parse_args()

    energy_columns = {
        "experiment": "experiment_energy_kwh",
        "deployment": "deployment_energy_kwh",
        "search": "search_energy_kwh",
        "training": "training_energy_kwh_per_run",
        "inference": "test_energy_kwh_per_run",
    }
    energy_column = energy_columns[args.energy_basis]

    measured = load_measured_results(args.results_dir)
    search = add_cost_proxy(load_search_results(args.results_dir))
    print_coverage(measured, search)

    print(
        f"\n=== Measured selection across experiment runs "
        f"(performance: {_metric_display_name(args.performance)}, "
        f"energy basis: {args.energy_basis}) ==="
    )
    measured_selection = epsilon_constraint_selection(
        measured,
        group_columns=["split_index", "selection_mode"],
        performance_column=args.performance,
        energy_column=energy_column,
        tolerances=args.tolerance,
    )
    display_columns = [
        "split_index",
        "selection_mode",
        "tolerance_pct",
        "best_score",
        "selected_score",
        "score_increase_pct",
        "selected_model",
        "selected_search_method",
        "selected_energy_kwh",
        "energy_saving_factor",
    ]
    print(measured_selection[display_columns].round(6).to_string(index=False))

    print(
        f"\n=== Measured Pareto front: "
        f"{_metric_display_name(args.performance)} vs "
        f"{args.energy_basis} energy ==="
    )
    measured_pareto = pareto_front(measured, args.performance, energy_column)
    pareto_columns = [
        "model",
        "split_index",
        "selection_mode",
        "search_method",
        "validation_rmse",
        "test_r2",
        "test_rmse",
        "test_rmse_std",
        "training_energy_kwh_per_run",
        "test_energy_kwh_per_run",
        "search_energy_kwh",
        "experiment_energy_kwh",
        "feature_count",
    ]
    if "best_config_energy_kwh" in measured_pareto.columns:
        pareto_columns.insert(-1, "best_config_energy_kwh")
    print(measured_pareto[pareto_columns].round(8).to_string(index=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if search.empty:
        print(
            "\n=== Per-configuration selection skipped: no search result files ==="
        )
    else:
        print("\n=== Per-configuration selection (validation RMSE) ===")
        proxy_selection = proxy_config_selection(search, args.tolerance)
        proxy_display = [
            "model",
            "split_index",
            "selection_mode",
            "search_method",
            "tolerance_pct",
            "best_validation_rmse",
            "selected_validation_rmse",
            "validation_increase_pct",
            "selection_energy_source",
            "energy_reduction_factor",
            "selected_config",
        ]
        print(proxy_selection[proxy_display].round(4).to_string(index=False))

        config_comparison = build_config_comparison(search, args.tolerance)
        config_comparison.to_csv(
            args.output_dir / "config_comparison.csv", index=False
        )

        print("\n=== Normal-best vs energy-optimized configurations ===")
        comparison_summary = (
            config_comparison.groupby(["model", "selection_energy_source"])
            .agg(
                comparisons=("configs_differ", "size"),
                configurations_differ=("configs_differ", "sum"),
                mean_validation_rmse_increase_pct=(
                    "validation_rmse_increase_pct",
                    "mean",
                ),
                mean_energy_reduction_factor=("energy_reduction_factor", "mean"),
            )
            .reset_index()
        )
        print(comparison_summary.round(3).to_string(index=False))
        print(
            "\nFull side-by-side parameter sets saved to "
            f"{args.output_dir / 'config_comparison.csv'}"
        )

    measured.to_csv(args.output_dir / "measured_experiment_rows.csv", index=False)
    measured_selection.to_csv(args.output_dir / "measured_selection.csv", index=False)
    measured_pareto.to_csv(args.output_dir / "measured_pareto.csv", index=False)
    if not search.empty:
        proxy_selection.to_csv(
            args.output_dir / "proxy_config_selection.csv", index=False
        )

    print(f"\nSaved outputs to {args.output_dir}")

    if not args.no_plot:
        by_split_path, overall_path = plot_measured_pareto(
            measured,
            args.plot_dir,
            performance_column=args.performance,
            energy_column=energy_column,
            energy_label=args.energy_basis,
        )
        print(f"Saved plot: {by_split_path}")
        print(f"Saved plot: {overall_path}")

        if not search.empty:
            print("\nGenerating per-configuration Pareto plots...")
            per_config_path, per_config_combined = plot_all_search_configs(
                search,
                args.plot_dir,
                performance_column=args.performance,
                energy_column="energy_for_selection",
            )
            print(f"Saved per-config plot (by model): {per_config_path}")
            print(f"Saved per-config plot (combined): {per_config_combined}")


if __name__ == "__main__":
    main()

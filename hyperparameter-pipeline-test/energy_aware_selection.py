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

# Configurations within this fraction of the best score are treated as
# equivalent on prediction quality, and the cheapest one is selected.
DEFAULT_TOLERANCES = (0.01, 0.02, 0.05, 0.10)


def _repair_lstm_csv(path):
    """The LSTM summary file is missing its header and starts with a stray '1'."""
    with open(path, "r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()

    if first_line.startswith("timestamp,"):
        return pd.read_csv(path)

    # Recover the header from the GRU file (same schema) and skip the bad line.
    columns = pd.read_csv(RESULTS_DIR / TUNING_FILES["gru"], nrows=0).columns
    return pd.read_csv(path, skiprows=1, header=None, names=columns)


def load_measured_results(results_dir):
    """Load one measured row per model/split/mode/search-method combination."""
    frames = []
    for model, filename in TUNING_FILES.items():
        path = results_dir / filename
        if not path.exists():
            print(f"[warn] missing {path}")
            continue

        if model == "lstm":
            frame = _repair_lstm_csv(path)
        else:
            frame = pd.read_csv(path)

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
        raise FileNotFoundError(f"No search result files found in {results_dir}")

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


def plot_measured_pareto(measured, output_dir, performance_column):
    """Plot measured performance against experiment/deployment energy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=False)
    split_titles = {
        0: "Split 0: entire timeline",
        1: "Split 1: summer 2025",
        2: "Split 2: winter 2025",
    }

    model_short = {
        "gru": "GRU",
        "lstm": "LSTM",
        "rf": "RF",
        "xgboost": "XGB",
    }
    annotation_offsets = [(6, 8), (6, -16), (-46, 6), (8, 18), (-52, -16)]

    for axis, split_index in zip(axes, (0, 1, 2)):
        subset = measured[measured["split_index"] == split_index]

        for model, group in subset.groupby("model"):
            for method, method_group in group.groupby("search_method"):
                axis.scatter(
                    method_group["experiment_energy_kwh"],
                    method_group[performance_column],
                    s=70,
                    color=model_colors.get(model, "grey"),
                    marker=method_markers.get(method, "o"),
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.85,
                    label=f"{model_labels.get(model, model)} ({method})",
                    zorder=3,
                )

        front = pareto_front(subset, performance_column, "experiment_energy_kwh")
        axis.plot(
            front["experiment_energy_kwh"],
            front[performance_column],
            color="black",
            linestyle="--",
            linewidth=1.3,
            marker="o",
            markersize=4,
            zorder=2,
            label="Pareto front",
        )

        for index, (_, row) in enumerate(front.iterrows()):
            axis.annotate(
                f"{model_short.get(row['model'], row['model'])} "
                f"{row['search_method']}",
                (row["experiment_energy_kwh"], row[performance_column]),
                textcoords="offset points",
                xytext=annotation_offsets[index % len(annotation_offsets)],
                fontsize=7,
            )

        axis.set_xscale("log")
        axis.set_title(split_titles[split_index])
        axis.set_xlabel("Lifecycle energy [kWh] (search + training + inference)")
        axis.set_ylabel("Test RMSE (lower is better)")
        axis.grid(True, which="both", alpha=0.25)
        axis.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Energy-aware selection: measured test RMSE vs lifecycle energy",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    by_split_path = output_dir / "energy_aware_pareto_by_split.png"
    fig.savefig(by_split_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Overall view: all splits together, to show the global front.
    fig, axis = plt.subplots(figsize=(11, 7))
    for model, group in measured.groupby("model"):
        for method, method_group in group.groupby("search_method"):
            axis.scatter(
                method_group["experiment_energy_kwh"],
                method_group[performance_column],
                s=75,
                color=model_colors.get(model, "grey"),
                marker=method_markers.get(method, "o"),
                edgecolor="black",
                linewidth=0.4,
                alpha=0.85,
                label=f"{model_labels.get(model, model)} ({method})",
                zorder=3,
            )

    overall_front = pareto_front(measured, performance_column, "experiment_energy_kwh")
    axis.plot(
        overall_front["experiment_energy_kwh"],
        overall_front[performance_column],
        color="black",
        linestyle="--",
        linewidth=1.3,
        marker="o",
        markersize=4,
        zorder=2,
        label="Pareto front",
    )

    axis.set_xscale("log")
    axis.set_title("Overall measured Pareto front (all splits)")
    axis.set_xlabel("Lifecycle energy [kWh] (search + training + inference)")
    axis.set_ylabel("Test RMSE (lower is better)")
    axis.grid(True, which="both", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    overall_path = output_dir / "energy_aware_pareto_overall.png"
    fig.savefig(overall_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return by_split_path, overall_path


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
        default="test_rmse",
        help=(
            "Performance column for the measured selection. test_rmse is a "
            "retrospective comparison; validation_rmse is the honest choice "
            "rule when using the results to pick a model."
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
        f"(performance: {args.performance}, energy basis: {args.energy_basis}) ==="
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

    print(f"\n=== Measured Pareto front: {args.performance} vs experiment energy ===")
    measured_pareto = pareto_front(measured, args.performance, "experiment_energy_kwh")
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    measured.to_csv(args.output_dir / "measured_experiment_rows.csv", index=False)
    measured_selection.to_csv(args.output_dir / "measured_selection.csv", index=False)
    measured_pareto.to_csv(args.output_dir / "measured_pareto.csv", index=False)
    proxy_selection.to_csv(args.output_dir / "proxy_config_selection.csv", index=False)

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

    print(f"\nSaved outputs to {args.output_dir}")

    if not args.no_plot:
        by_split_path, overall_path = plot_measured_pareto(
            measured,
            args.plot_dir,
            performance_column=args.performance,
        )
        print(f"Saved plot: {by_split_path}")
        print(f"Saved plot: {overall_path}")


if __name__ == "__main__":
    main()

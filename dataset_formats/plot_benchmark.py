"""Plot the dataset file-format save/load benchmark for the report.

Reads ``dataset_save_results.csv`` and ``dataset_load_results.csv`` (written by
``save_ml_dataset.py`` and ``load_ml_dataset.py``), keeps the most recent run of
each benchmark, and writes a two-by-two figure with the time and energy cost of
saving and loading every format. The output goes to ``plots/``.

Run:

    uv run python dataset_formats/plot_benchmark.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAVE_RESULTS_PATH = PROJECT_ROOT / "dataset_save_results.csv"
LOAD_RESULTS_PATH = PROJECT_ROOT / "dataset_load_results.csv"
OUTPUT_DIR = PROJECT_ROOT / "plots"

# 1 kWh = 3.6e6 J.
JOULES_PER_KWH = 3.6e6

FORMAT_LABELS = {
    "csv": "CSV",
    "csv_gzip": "CSV gzip",
    "parquet": "Parquet",
    "feather": "Feather",
    "pickle": "Pickle",
    "hdf5": "HDF5",
    "json": "JSON",
}

FORMAT_COLORS = {
    "csv": "tab:blue",
    "csv_gzip": "tab:cyan",
    "parquet": "tab:green",
    "feather": "tab:olive",
    "pickle": "tab:orange",
    "hdf5": "tab:purple",
    "json": "tab:red",
}


def latest_run(results_path):
    """Return only the rows from the most recent benchmark run in ``results_path``."""
    results = pd.read_csv(results_path)
    latest = results["timestamp"].max()
    return results[results["timestamp"] == latest].copy()


def summarise(results):
    """Aggregate repeats into mean and standard deviation per format."""
    summary = (
        results.groupby("format", sort=False)
        .agg(
            file_size_mb=("file_size_mb", "first"),
            duration_s=("duration_s", "mean"),
            duration_std_s=("duration_s", "std"),
            energy_kwh=("energy_kwh", "mean"),
            energy_std_kwh=("energy_kwh", "std"),
        )
        .reset_index()
    )
    summary["duration_ms"] = summary["duration_s"] * 1e3
    summary["duration_std_ms"] = summary["duration_std_s"].fillna(0.0) * 1e3
    summary["energy_j"] = summary["energy_kwh"] * JOULES_PER_KWH
    summary["energy_std_j"] = summary["energy_std_kwh"].fillna(0.0) * JOULES_PER_KWH
    summary["label"] = summary["format"].map(FORMAT_LABELS)
    return summary


def draw_panel(ax, summary, metric, std_metric, title):
    """Draw one horizontal bar panel with a logarithmic x-axis."""
    positions = np.arange(len(summary))
    values = summary[metric].to_numpy()
    errors = summary[std_metric].to_numpy()
    colors = [FORMAT_COLORS[name] for name in summary["format"]]

    # Symmetric error bars would cross zero on a log axis for the fastest
    # formats, so clip the lower half to keep every whisker positive.
    lower_errors = np.minimum(errors, values * 0.8)
    asymmetric_errors = np.vstack([lower_errors, errors])

    ax.barh(
        positions,
        values,
        xerr=asymmetric_errors,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        error_kw={"ecolor": "black", "elinewidth": 0.8, "capsize": 3},
        alpha=0.85,
    )

    ax.set_xscale("log")
    ax.set_yticks(positions)
    ax.set_yticklabels(summary["label"])
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", which="both", alpha=0.25)
    ax.set_axisbelow(True)

    # Headroom for the value labels after the upper error whisker.
    label_positions = values + errors
    ax.set_xlim(right=label_positions.max() * 6)

    for position, value, label_position in zip(positions, values, label_positions):
        ax.text(
            label_position * 1.08,
            position,
            f"{value:,.1f}",
            va="center",
            ha="left",
            fontsize=8,
        )


def make_plot(save_summary, load_summary):
    """Build the two-by-two report figure."""
    # Keep one order across all panels, sorted by save duration (fastest first).
    order = save_summary.sort_values("duration_ms")["format"].tolist()
    save_summary = save_summary.set_index("format").loc[order].reset_index()
    load_summary = load_summary.set_index("format").loc[order].reset_index()

    # Include the file size in the shared y-tick labels.
    size_labels = [
        f"{FORMAT_LABELS[name]} ({size:,.1f} MB)"
        for name, size in zip(save_summary["format"], save_summary["file_size_mb"])
    ]
    for summary in (save_summary, load_summary):
        summary["label"] = size_labels

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharey=True,
        gridspec_kw={"hspace": 0.35, "wspace": 0.08},
    )

    draw_panel(axes[0, 0], save_summary, "duration_ms", "duration_std_ms", "Save: time")
    draw_panel(axes[0, 1], load_summary, "duration_ms", "duration_std_ms", "Load: time")
    draw_panel(axes[1, 0], save_summary, "energy_j", "energy_std_j", "Save: energy")
    draw_panel(axes[1, 1], load_summary, "energy_j", "energy_std_j", "Load: energy")

    # The panels share their y-axis, so invert it once to put the fastest
    # format at the top of every panel.
    axes[0, 0].invert_yaxis()

    axes[0, 0].set_ylabel("Time per operation (ms)")
    axes[1, 0].set_ylabel("Energy per operation (J)")

    fig.suptitle(
        "Dataset file-format I/O benchmark",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.935,
        "25,797 rows x 169 columns | mean of 5 runs ± std | "
        "CodeCarbon (NOR) | lower is better",
        ha="center",
        fontsize=9.5,
    )
    fig.text(
        0.5,
        0.015,
        "Logarithmic axes. Loads are measured from the OS page cache after a "
        "warm-up, so they reflect repeated loading in one session.",
        ha="center",
        fontsize=8,
        color="dimgray",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{timestamp}_dataset_format_benchmark.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    save_summary = summarise(latest_run(SAVE_RESULTS_PATH))
    load_summary = summarise(latest_run(LOAD_RESULTS_PATH))
    output_path = make_plot(save_summary, load_summary)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()

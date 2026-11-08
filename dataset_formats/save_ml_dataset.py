"""Save ``ml_dataset`` in several file formats and measure the energy use.

Each format is first written once without measurement: this warms up its
optional library, creates the file, and calibrates how many writes fit into one
measurement window. The format is then measured ``REPEATS`` times with
CodeCarbon's ``OfflineEmissionsTracker``; one row per repetition is appended to
``dataset_save_results.csv``.

Configuration lives in the constants at the top of this file. ``FORMAT_NAMES``
can be set to a subset of the names in ``dataset_formats/formats.py``.

Run:

    uv run python dataset_formats/save_ml_dataset.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from dataset_formats.benchmark import (
    calibrate_batch_size,
    file_size_bytes,
    measure_operation,
)
from dataset_formats.formats import dataset_path, select_formats
from logger.csv_logger import write_to_csv

DATASET_PATH = PROJECT_ROOT / "ml_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "dataset_formats" / "benchmark_files"
RESULTS_CSV_PATH = PROJECT_ROOT / "dataset_save_results.csv"

# None measures every format in the registry. Use e.g. ["csv", "parquet"] to
# restrict the run.
FORMAT_NAMES = None

# Number of measured saves per format. Each measurement runs a batch of saves
# chosen by a calibration call (see TARGET_MEASUREMENT_SECONDS).
REPEATS = 5

# Aim for measurements of at least this length and cap how many saves go into
# one measurement.
TARGET_MEASUREMENT_SECONDS = 1.0
MAX_BATCH_SIZE = 20


def load_dataset(file_path=DATASET_PATH):
    """Load the source dataset and parse the date column.

    Parsing happens before the benchmark so every format stores the same
    typed frame instead of a CSV-specific string date.
    """
    data = pd.read_csv(file_path)
    data["delivery_day"] = pd.to_datetime(data["delivery_day"])
    return data


def save_dataset(data, dataset_format, output_dir):
    """Write ``data`` with one format and return the output path."""
    path = dataset_path(dataset_format, output_dir)
    dataset_format.write(data, path)
    return path


def benchmark_format(data, dataset_format, output_dir, repeats):
    """Measure repeated saves for one format and return the result rows."""
    # Warm-up and calibration: import the format's library, create the file,
    # and choose how many saves make one measurement.
    _, calibration, batch_size = calibrate_batch_size(
        save_dataset,
        data,
        dataset_format,
        output_dir,
        target_seconds=TARGET_MEASUREMENT_SECONDS,
        max_batch_size=MAX_BATCH_SIZE,
    )
    print(
        f"  calibration: {calibration.duration_s:.3f} s/save, "
        f"{batch_size} save(s) per measurement"
    )

    rows = []
    for repeat in range(1, repeats + 1):
        path, measurement = measure_operation(
            save_dataset,
            data,
            dataset_format,
            output_dir,
            batch_size=batch_size,
        )
        size = file_size_bytes(path)
        rows.append(
            {
                "format": dataset_format.name,
                "description": dataset_format.description,
                "repeat": repeat,
                "batch_size": batch_size,
                "rows": len(data),
                "columns": len(data.columns),
                "file_size_bytes": size,
                "file_size_mb": size / (1024**2),
                "energy_kwh": measurement.energy_kwh,
                "emissions_kg": measurement.emissions_kg,
                "duration_s": measurement.duration_s,
                "output_path": str(path.relative_to(PROJECT_ROOT)),
            }
        )
        print(
            f"  {dataset_format.name} {repeat}/{repeats}: "
            f"{measurement.duration_s:.3f} s/save, "
            f"{measurement.energy_kwh:.9f} kWh/save, "
            f"{size / (1024**2):.2f} MB"
        )

    return rows


def print_summary(results):
    """Print mean energy, duration, and size per format."""
    summary = (
        results.groupby("format", sort=False)
        .agg(
            runs=("repeat", "count"),
            file_size_mb=("file_size_mb", "first"),
            mean_duration_s=("duration_s", "mean"),
            mean_energy_kwh=("energy_kwh", "mean"),
            std_energy_kwh=("energy_kwh", "std"),
            mean_emissions_kg=("emissions_kg", "mean"),
        )
        .reset_index()
        .sort_values("mean_energy_kwh")
    )
    print("\nSave benchmark summary:")
    print(summary.to_string(index=False))


def main():
    if REPEATS < 1:
        raise ValueError("REPEATS must be at least 1.")

    formats = select_formats(FORMAT_NAMES)
    if not formats:
        raise RuntimeError("No formats to benchmark.")

    data = load_dataset()
    print(
        f"Loaded {DATASET_PATH.name}: {len(data):,} rows x {len(data.columns)} columns"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset_format in formats:
        print(f"Saving as {dataset_format.name} ({dataset_format.description})")
        rows.extend(benchmark_format(data, dataset_format, OUTPUT_DIR, REPEATS))

    results = pd.DataFrame(rows)
    write_to_csv(results, RESULTS_CSV_PATH)
    print_summary(results)
    print(f"\nResults appended to {RESULTS_CSV_PATH}")
    print(f"Files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

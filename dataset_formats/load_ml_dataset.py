"""Load ``ml_dataset`` from several file formats and measure the energy use.

The files are produced by ``dataset_formats/save_ml_dataset.py``. Each format
is read once without measurement: this warms up its optional library, fills the
page cache, and calibrates how many reads fit into one measurement window. The
format is then measured ``REPEATS`` times with CodeCarbon's
``OfflineEmissionsTracker``. One row per repetition is appended to
``dataset_load_results.csv``, including whether the loaded frame matches the
source dataset.

The warm-up leaves the file in the operating system's page cache, so the
measured runs reflect repeated loading in one experiment session rather than a
cold disk read.

Run:

    uv run python dataset_formats/save_ml_dataset.py   # create the files first
    uv run python dataset_formats/load_ml_dataset.py
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
    verify_frame,
)
from dataset_formats.formats import dataset_path, select_formats
from logger.csv_logger import write_to_csv

DATASET_PATH = PROJECT_ROOT / "ml_dataset.csv"
INPUT_DIR = PROJECT_ROOT / "dataset_formats" / "benchmark_files"
RESULTS_CSV_PATH = PROJECT_ROOT / "dataset_load_results.csv"

# None measures every format in the registry. Use e.g. ["csv", "parquet"] to
# restrict the run.
FORMAT_NAMES = None

# Number of measured loads per format. Each measurement runs a batch of loads
# chosen by a calibration read.
REPEATS = 5

# Aim for measurements of at least this length and cap how many loads go into
# one measurement.
TARGET_MEASUREMENT_SECONDS = 1.0
MAX_BATCH_SIZE = 100


def load_dataset(file_path=DATASET_PATH):
    """Load the source dataset used to verify the benchmark files."""
    data = pd.read_csv(file_path)
    data["delivery_day"] = pd.to_datetime(data["delivery_day"])
    return data


def load_with_format(dataset_format, input_dir):
    """Read one dataset format and return the DataFrame."""
    return dataset_format.read(dataset_path(dataset_format, input_dir))


def ensure_file(dataset_format, reference, input_dir):
    """Create the format's file if it does not exist yet, without measuring."""
    path = dataset_path(dataset_format, input_dir)
    if not path.exists():
        print(
            f"  {path.name} is missing; writing it once so the load can be "
            "measured (use save_ml_dataset.py for the save benchmark)."
        )
        input_dir.mkdir(parents=True, exist_ok=True)
        dataset_format.write(reference, path)
    return path


def benchmark_format(dataset_format, reference, input_dir, repeats):
    """Measure repeated loads for one format and return the result rows."""
    path = ensure_file(dataset_format, reference, input_dir)
    size = file_size_bytes(path)

    # Warm-up and calibration: import the format's library, populate the page
    # cache, and choose how many loads make one measurement.
    last_loaded, calibration, batch_size = calibrate_batch_size(
        load_with_format,
        dataset_format,
        input_dir,
        target_seconds=TARGET_MEASUREMENT_SECONDS,
        max_batch_size=MAX_BATCH_SIZE,
    )
    print(
        f"  calibration: {calibration.duration_s:.3f} s/load, "
        f"{batch_size} load(s) per measurement"
    )

    rows = []
    for repeat in range(1, repeats + 1):
        last_loaded, measurement = measure_operation(
            load_with_format,
            dataset_format,
            input_dir,
            batch_size=batch_size,
        )
        rows.append(
            {
                "format": dataset_format.name,
                "description": dataset_format.description,
                "repeat": repeat,
                "batch_size": batch_size,
                "rows": len(last_loaded),
                "columns": len(last_loaded.columns),
                "file_size_bytes": size,
                "file_size_mb": size / (1024**2),
                "energy_kwh": measurement.energy_kwh,
                "emissions_kg": measurement.emissions_kg,
                "duration_s": measurement.duration_s,
                "values_match": None,
                "input_path": str(path.relative_to(PROJECT_ROOT)),
            }
        )
        print(
            f"  {dataset_format.name} {repeat}/{repeats}: "
            f"{measurement.duration_s:.3f} s/load, "
            f"{measurement.energy_kwh:.9f} kWh/load"
        )

    # Verify once per format: the loaded frame must have the same columns,
    # row count, and numeric values as the source dataset.
    matches = verify_frame(last_loaded, reference)
    for row in rows:
        row["values_match"] = matches
    if not matches:
        print(f"  WARNING: {dataset_format.name} did not round-trip correctly.")

    return rows


def print_summary(results):
    """Print mean load energy and duration per format."""
    summary = (
        results.groupby("format", sort=False)
        .agg(
            runs=("repeat", "count"),
            file_size_mb=("file_size_mb", "first"),
            mean_duration_s=("duration_s", "mean"),
            mean_energy_kwh=("energy_kwh", "mean"),
            std_energy_kwh=("energy_kwh", "std"),
            mean_emissions_kg=("emissions_kg", "mean"),
            values_match=("values_match", "last"),
        )
        .reset_index()
        .sort_values("mean_energy_kwh")
    )
    print("\nLoad benchmark summary:")
    print(summary.to_string(index=False))


def main():
    if REPEATS < 1:
        raise ValueError("REPEATS must be at least 1.")

    formats = select_formats(FORMAT_NAMES)
    if not formats:
        raise RuntimeError("No formats to benchmark.")

    reference = load_dataset()
    print(
        f"Loaded {DATASET_PATH.name}: "
        f"{len(reference):,} rows x {len(reference.columns)} columns"
    )

    rows = []
    for dataset_format in formats:
        print(f"Loading {dataset_format.name} ({dataset_format.description})")
        rows.extend(benchmark_format(dataset_format, reference, INPUT_DIR, REPEATS))

    results = pd.DataFrame(rows)
    write_to_csv(results, RESULTS_CSV_PATH)
    print_summary(results)
    print(f"\nResults appended to {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()

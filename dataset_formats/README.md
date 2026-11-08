# Dataset format energy benchmark

Measures how much time and energy it takes to save and load `ml_dataset` in
different file formats. Both scripts use CodeCarbon's
`OfflineEmissionsTracker` with `country_iso_code="NOR"`, matching the model
training scripts.

## Files

| File | Purpose |
| --- | --- |
| `formats.py` | Registry of supported formats with their writer/reader. |
| `benchmark.py` | CodeCarbon measurement, batch calibration, and round-trip checks. |
| `save_ml_dataset.py` | Writes `ml_dataset` in every format, measures saving. |
| `load_ml_dataset.py` | Reads every format, measures loading, verifies the data. |
| `plot_benchmark.py` | Plots the latest save/load results (time and energy per format). |

Generated files go to `dataset_formats/benchmark_files/` (ignored by git).
Results are appended to CSV files in the project root:

- `dataset_save_results.csv` – one row per save measurement.
- `dataset_load_results.csv` – one row per load measurement.

## Usage

```powershell
uv run python dataset_formats/save_ml_dataset.py
uv run python dataset_formats/load_ml_dataset.py
uv run python dataset_formats/plot_benchmark.py
```

Run the save script first so the load script has files to read. If a file is
missing, the load script writes it once without measuring and prints a note.
The plot script uses the most recent run in each result file and writes a
timestamped PNG to `plots/`.

Settings are constants at the top of each script:

- `FORMAT_NAMES` – `None` for every format, or a list such as `["csv", "parquet"]`.
- `REPEATS` – number of measured runs per format (default 5).
- `TARGET_MEASUREMENT_SECONDS` – minimum length of one CodeCarbon measurement.
- `MAX_BATCH_SIZE` – cap on how many save/load calls go into one measurement.

## Formats

| Name | File | Notes |
| --- | --- | --- |
| `csv` | `ml_dataset.csv` | Text baseline. Dates come back as strings. |
| `csv_gzip` | `ml_dataset.csv.gz` | CSV with gzip compression. |
| `parquet` | `ml_dataset.parquet` | Columnar binary, Snappy compression. |
| `feather` | `ml_dataset.feather` | Columnar binary optimised for fast I/O. |
| `pickle` | `ml_dataset.pkl` | Fastest Python-native round trip, but unsafe for untrusted files. |
| `hdf5` | `ml_dataset.h5` | HDF5 fixed format via PyTables. |
| `json` | `ml_dataset.json` | pandas table orientation; at most 15 significant digits. |

`parquet` and `feather` need `pyarrow`; `hdf5` needs `tables`. Formats whose
dependency is missing are skipped with a message.

## Measurement method

1. The source CSV is parsed once so all formats write the same typed DataFrame
   (including a datetime `delivery_day`).
2. Each format runs one calibration call without measurement. This imports the
   format's library, warms the page cache, and measures how long one operation
   takes.
3. The batch size is chosen so one measured run lasts roughly
   `TARGET_MEASUREMENT_SECONDS` (capped by `MAX_BATCH_SIZE`). Batching matters
   because CodeCarbon cannot resolve the energy of a 10–20 ms feather write on
   its own.
4. `REPEATS` measured runs are performed. Energy, emissions, and duration are
   divided by the batch size, so result rows are **per save/load operation**.
5. The load script verifies the last loaded frame against the source (columns,
   row count, numeric values) and records `values_match`.

Because of the calibration step, the measured loads come from the operating
system's page cache. Results therefore describe repeated loading during an
experiment, not a cold disk read. Run the save script again before a serious
load benchmark if the files have been evicted from the cache.

## Result columns

Both result files share:

- `format`, `description`, `repeat`, `batch_size`
- `rows`, `columns`, `file_size_bytes`, `file_size_mb`
- `energy_kwh`, `emissions_kg` – per operation
- `duration_s` – per operation
- `output_path` / `input_path`

`dataset_load_results.csv` additionally has `values_match`.

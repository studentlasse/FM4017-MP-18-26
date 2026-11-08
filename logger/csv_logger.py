import csv
from datetime import datetime
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _read_csv_with_evolved_header(path, incoming_columns):
    """Read an appended CSV whose rows may contain columns missing from its header.

    ``write_to_csv`` appends without rewriting the header, so adding a new
    column to the DataFrame leaves older files with a shorter header. New
    columns are inserted after their preceding column in ``incoming_columns``,
    which keeps existing rows aligned.
    """
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            original_header = next(reader)
        except StopIteration:
            return pd.DataFrame(columns=incoming_columns)

        rows = [row for row in reader if row]

    header = list(original_header)
    for column in incoming_columns:
        if column in header:
            continue
        position = incoming_columns.index(column)
        anchor = next(
            (
                previous
                for previous in reversed(incoming_columns[:position])
                if previous in header
            ),
            None,
        )
        if anchor is None:
            header.append(column)
        else:
            header.insert(header.index(anchor) + 1, column)

    records = []
    for row in rows:
        if len(row) == len(original_header):
            record = dict(zip(original_header, row))
        else:
            record = dict(zip(header, row))
        records.append(record)

    return pd.DataFrame(records, columns=header)


def write_to_csv(df, filename):
    """Append a row while keeping the CSV header consistent with its columns.

    If the DataFrame has new columns that the existing file does not, the file
    is rewritten with the merged schema instead of appending a row with extra
    fields. This keeps the CSV readable by pandas and downstream scripts.
    """
    df = df.copy()
    df.insert(
        0,
        "timestamp",
        datetime.now().isoformat(timespec="seconds"),
    )

    path = Path(filename)
    file_exists = path.exists() and path.stat().st_size > 0

    if not file_exists:
        df.to_csv(path, index=False)
        return f"DataFrame added to {path} successfully."

    existing = _read_csv_with_evolved_header(path, df.columns.tolist())
    columns = existing.columns.tolist() + [
        column for column in df.columns if column not in existing.columns
    ]
    combined = pd.concat(
        [existing, df.reindex(columns=columns)],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(path, index=False)

    return f"DataFrame added to {path} successfully."


def save_plot_csv(data, description):
    df = data.copy()

    plot_data_dir = PROJECT_ROOT / "results"
    plot_data_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{timestamp}_{description}.csv"
    csv_path = plot_data_dir / filename

    df.to_csv(csv_path, index=False)

    return f"Plotdata saved to {filename} successfully."

"""File-format registry shared by the dataset save/load energy benchmarks.

Each format is described by a :class:`DatasetFormat` holding a writer and a
reader with the same signature as ``pandas.DataFrame.to_*`` / ``pandas.read_*``.
The save and load scripts iterate over this registry so that both measure the
exact same set of formats.

Formats that need an optional package (``pyarrow`` for parquet/feather,
``tables`` for HDF5) report themselves as unavailable when it is not installed.
"""

from collections.abc import Callable
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import pandas as pd

HDF5_KEY = "dataset"


@dataclass(frozen=True)
class DatasetFormat:
    """One on-disk representation of the dataset.

    Attributes:
        name: Short identifier used in result tables.
        suffix: File suffix, including the leading dot.
        write: Callable that writes a DataFrame to a path.
        read: Callable that reads a DataFrame from a path.
        description: Human-readable description of the encoding.
        requires: Optional package required to read/write this format.
    """

    name: str
    suffix: str
    write: Callable[[pd.DataFrame, Path], None]
    read: Callable[[Path], pd.DataFrame]
    description: str
    requires: str | None = None

    def is_available(self) -> bool:
        """Return True when the optional dependency for this format is installed."""
        return self.requires is None or find_spec(self.requires) is not None


def _write_csv(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(path, index=False)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _write_csv_gzip(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(path, index=False, compression="gzip")


def _write_parquet(data: pd.DataFrame, path: Path) -> None:
    data.to_parquet(path, index=False)


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _write_feather(data: pd.DataFrame, path: Path) -> None:
    data.to_feather(path)


def _read_feather(path: Path) -> pd.DataFrame:
    return pd.read_feather(path)


def _write_pickle(data: pd.DataFrame, path: Path) -> None:
    data.to_pickle(path)


def _read_pickle(path: Path) -> pd.DataFrame:
    return pd.read_pickle(path)


def _write_hdf5(data: pd.DataFrame, path: Path) -> None:
    data.to_hdf(path, key=HDF5_KEY, mode="w", format="fixed")


def _read_hdf5(path: Path) -> pd.DataFrame:
    return pd.read_hdf(path, key=HDF5_KEY)


def _write_json(data: pd.DataFrame, path: Path) -> None:
    # pandas caps JSON float precision at 15 significant digits, so JSON can
    # differ from the other formats by a few ULPs.
    data.to_json(path, orient="table", double_precision=15)


def _read_json(path: Path) -> pd.DataFrame:
    return pd.read_json(path, orient="table")


FORMATS = (
    DatasetFormat(
        name="csv",
        suffix=".csv",
        write=_write_csv,
        read=_read_csv,
        description="Uncompressed, human-readable text",
    ),
    DatasetFormat(
        name="csv_gzip",
        suffix=".csv.gz",
        write=_write_csv_gzip,
        read=_read_csv,
        description="CSV compressed with gzip",
    ),
    DatasetFormat(
        name="parquet",
        suffix=".parquet",
        write=_write_parquet,
        read=_read_parquet,
        description="Columnar binary with Snappy compression (pyarrow)",
        requires="pyarrow",
    ),
    DatasetFormat(
        name="feather",
        suffix=".feather",
        write=_write_feather,
        read=_read_feather,
        description="Columnar, zero-copy-oriented binary (pyarrow)",
        requires="pyarrow",
    ),
    DatasetFormat(
        name="pickle",
        suffix=".pkl",
        write=_write_pickle,
        read=_read_pickle,
        description="Python pickle of the in-memory DataFrame (unsafe for untrusted files)",
    ),
    DatasetFormat(
        name="hdf5",
        suffix=".h5",
        write=_write_hdf5,
        read=_read_hdf5,
        description="HDF5 container using the fixed format (PyTables)",
        requires="tables",
    ),
    DatasetFormat(
        name="json",
        suffix=".json",
        write=_write_json,
        read=_read_json,
        description="JSON in pandas' table orientation (15-digit floats)",
    ),
)


def select_formats(names=None) -> list[DatasetFormat]:
    """Return the requested formats, skipping ones with missing dependencies.

    ``names`` is an iterable of format names, or None for every format.
    """
    if names is None:
        selected = list(FORMATS)
    else:
        by_name = {dataset_format.name: dataset_format for dataset_format in FORMATS}
        unknown = [name for name in names if name not in by_name]
        if unknown:
            available = ", ".join(by_name)
            raise ValueError(
                f"Unknown format(s) {unknown}. Available formats: {available}."
            )
        selected = [by_name[name] for name in names]

    available = []
    for dataset_format in selected:
        if dataset_format.is_available():
            available.append(dataset_format)
        else:
            print(
                f"Skipping {dataset_format.name!r}: requires the "
                f"{dataset_format.requires!r} package."
            )

    return available


def dataset_path(dataset_format: DatasetFormat, directory: Path) -> Path:
    """Return the benchmark file path for one format inside ``directory``."""
    return Path(directory) / f"ml_dataset{dataset_format.suffix}"

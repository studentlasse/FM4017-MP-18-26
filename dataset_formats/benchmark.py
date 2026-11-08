"""Shared helpers for the dataset format energy benchmarks.

Energy is measured with CodeCarbon's ``OfflineEmissionsTracker``, matching the
model scripts in this repository (``country_iso_code="NOR"``). Wall-clock
duration is measured with ``time.perf_counter`` so every result row can relate
energy use to elapsed time.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from codecarbon import OfflineEmissionsTracker

COUNTRY_ISO_CODE = "NOR"

# Operations this quick are below CodeCarbon's reliable sampling window, so
# they are repeated in batches until one measurement covers about a second.
DEFAULT_TARGET_MEASUREMENT_SECONDS = 1.0
DEFAULT_MAX_BATCH_SIZE = 100


@dataclass(frozen=True)
class Measurement:
    """Energy, emissions, and duration of one operation."""

    energy_kwh: float
    emissions_kg: float
    duration_s: float


def measure_operation(
    operation: Callable,
    *args,
    batch_size: int = 1,
    **kwargs,
) -> tuple[Any, Measurement]:
    """Run ``operation`` ``batch_size`` times once and return a per-run measurement.

    The tracker is started immediately before the first call and stopped after
    the last one, so only the batch itself is measured. Energy, emissions, and
    duration are divided by ``batch_size`` to report the cost of one operation.

    Batching keeps very fast formats (e.g. feather) inside CodeCarbon's
    reliable measurement window instead of reporting near-zero energy.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    tracker = OfflineEmissionsTracker(
        country_iso_code=COUNTRY_ISO_CODE,
        save_to_file=False,
        log_level="error",
    )
    tracker.start()

    start = time.perf_counter()
    try:
        result = None
        for _ in range(batch_size):
            result = operation(*args, **kwargs)
    finally:
        duration_s = time.perf_counter() - start
        tracker.stop()

    emissions_data = tracker.final_emissions_data
    measurement = Measurement(
        energy_kwh=emissions_data.energy_consumed / batch_size,
        emissions_kg=emissions_data.emissions / batch_size,
        duration_s=duration_s / batch_size,
    )
    return result, measurement


def calibrate_batch_size(
    operation: Callable,
    *args,
    target_seconds: float = DEFAULT_TARGET_MEASUREMENT_SECONDS,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    **kwargs,
) -> tuple[Any, Measurement, int]:
    """Time one call and choose a batch size for the measured runs.

    Returns the call's result, its measurement (effectively the warm-up), and
    the number of calls that should be batched into one measured run.
    """
    result, measurement = measure_operation(operation, *args, **kwargs)

    if measurement.duration_s <= 0:
        batch_size = max_batch_size
    else:
        batch_size = math.ceil(target_seconds / measurement.duration_s)
    batch_size = min(max(batch_size, 1), max_batch_size)

    return result, measurement, batch_size


def file_size_bytes(path) -> int:
    """Return the size of ``path`` in bytes."""
    return path.stat().st_size


def verify_frame(loaded: pd.DataFrame, reference: pd.DataFrame) -> bool:
    """Check that a loaded frame kept the reference shape, columns, and values.

    Numeric columns are compared with a small tolerance because JSON stores at
    most 15 significant digits and can therefore differ by a few ULPs.
    ``delivery_day`` is excluded from the value comparison because a CSV round
    trip turns it back into a string, while binary formats preserve the
    datetime dtype.
    """
    if list(loaded.columns) != list(reference.columns):
        return False
    if len(loaded) != len(reference):
        return False

    numeric_columns = [
        column
        for column in reference.columns
        if pd.api.types.is_numeric_dtype(reference[column])
    ]
    expected = reference[numeric_columns].to_numpy(dtype="float64")
    actual = loaded[numeric_columns].to_numpy(dtype="float64")

    return bool(np.allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True))

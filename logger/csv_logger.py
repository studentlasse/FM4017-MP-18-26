from datetime import datetime
from pathlib import Path

import pandas as pd


def write_to_csv(df, filename):
    df = df.copy()
    df.insert(
        0,
        "timestamp",
        datetime.now().isoformat(timespec="seconds"),
    )

    file_exists = Path(filename).exists()

    df.to_csv(
        filename,
        mode="a",
        header=not file_exists,
        index=False,
    )

    return f"DataFrame added to {filename} successfully."
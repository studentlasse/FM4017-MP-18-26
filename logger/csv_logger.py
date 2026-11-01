from datetime import datetime
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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

def save_plot_csv(data, description):
    df = data.copy()

    plot_data_dir = PROJECT_ROOT / "results" 
    plot_data_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{timestamp}_{description}.csv"
    csv_path = plot_data_dir / filename

    df.to_csv(csv_path, index=False)

    return f"Plotdata saved to {filename} successfully."
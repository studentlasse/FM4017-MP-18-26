"""Simple plot of dataset format benchmark results."""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

SAVE_CSV = Path(__file__).parents[1] / "dataset_save_results.csv"
LOAD_CSV = Path(__file__).parents[1] / "dataset_load_results.csv"

save = pd.read_csv(SAVE_CSV)
load = pd.read_csv(LOAD_CSV)

# Latest run only
save = save[save.timestamp == save.timestamp.max()]
load = load[load.timestamp == load.timestamp.max()]

# Mean per format
s = save.groupby("format").agg(
    size=("file_size_mb", "first"),
    save_energy=("energy_kwh", "mean"),
)
l = load.groupby("format").agg(
    load_energy=("energy_kwh", "mean"),
)
df = s.join(l).sort_values("save_energy")

labels = {"csv": "CSV", "csv_gzip": "CSV gzip", "parquet": "Parquet",
          "feather": "Feather", "pickle": "Pickle", "hdf5": "HDF5", "json": "JSON"}
df["label"] = df.index.map(labels)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, col, title in [(ax1, "save_energy", "Save energy"), (ax2, "load_energy", "Load energy")]:
    ax.barh(df["label"], df[col], color="steelblue", edgecolor="k", height=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Energy (kWh, log)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    for i, v in enumerate(df[col]):
        ax.text(v * 1.1, i, f"{v:.2e}", va="center", fontsize=8)

fig.suptitle("Dataset I/O energy: 25,797 rows × 169 cols (mean of 5 runs, CodeCarbon NOR)")
fig.tight_layout()
plt.savefig("plots/simple_benchmark.png", dpi=200, bbox_inches="tight")
print("Saved to plots/simple_benchmark.png")
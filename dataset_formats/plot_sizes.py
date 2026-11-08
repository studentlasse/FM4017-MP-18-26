"""Simple file size plot from dataset format benchmark."""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

SAVE_CSV = Path(__file__).parents[1] / "dataset_save_results.csv"

save = pd.read_csv(SAVE_CSV)
save = save[save.timestamp == save.timestamp.max()]

df = save.groupby("format").agg(size=("file_size_mb", "first")).sort_values("size")

labels = {"csv": "CSV", "csv_gzip": "CSV gzip", "parquet": "Parquet",
          "feather": "Feather", "pickle": "Pickle", "hdf5": "HDF5", "json": "JSON"}
df["label"] = df.index.map(labels)

fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(df["label"], df["size"], color="steelblue", edgecolor="k", height=0.6)
ax.set_xlabel("File size (MB)")
ax.set_title("Dataset file size by format")
ax.grid(True, axis="x", alpha=0.3)
for i, v in enumerate(df["size"]):
    ax.text(v * 1.02, i, f"{v:,.1f} MB", va="center", fontsize=9)

fig.tight_layout()
plt.savefig("plots/file_sizes.png", dpi=200, bbox_inches="tight")
print("Saved to plots/file_sizes.png")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

plots_dir = Path("plots")
plots_dir.mkdir(parents=True, exist_ok=True)


xgb_baseline = pd.read_csv("xgboost_results.csv")
gru_baseline = pd.read_csv("gru_results.csv")
lstm_baseline = pd.read_csv("lstm_results.csv")
rf_baseline =pd.read_csv("rf_results.csv")

xgb_tuned = pd.read_csv("xgboost_search_results.csv")
gru_tuned = pd.read_csv("gru_search_results.csv")
lstm_tuned = pd.read_csv("lstm_search_results.csv")
rf_tuned =pd.read_csv("rf_search_results.csv")


tuned_frames = {
    "XGBoost": xgb_tuned,
    "GRU": gru_tuned,
    "LSTM": lstm_tuned,
    "Random Forest": rf_tuned,
}


baseline_frames = {
    "XGBoost": xgb_baseline,
    "GRU": gru_baseline,
    "LSTM": lstm_baseline,
    "Random Forest": rf_baseline,
}

baseline = pd.DataFrame({
    model: df.iloc[0]
    for model, df in baseline_frames.items()
}).T

tuned = pd.DataFrame({
    model: df.iloc[0]
    for model, df in tuned_frames.items()
}).T

for model in baseline.index:
    method = str(tuned.loc[model, "search_method"])
    method_label = method.replace("_", " ").title()

metrics = {
    "test_rmse": "Test RMSE",
    "validation_rmse": "Validation RMSE",
    "test_mae": "Test MAE",
    "validation_mae": "Validation MAE",
   "training_energy_kwh": "Training energy (kWh)",
    "test_energy_kwh": "Test energy (kWh)",
    "validation_energy_kwh": "Validation energy (kWh)",
}

lower_is_better = {
    "test_rmse", 
    "validation_rmse", 
    "test_mae", 
    "validation_mae",
    "training_energy_kwh", 
    "test_energy_kwh", 
    "validation_energy_kwh",
}


metric_keys = list(metrics.keys())
axis_labels = [metrics[key] for key in metric_keys]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for model in baseline.index:
    baseline_values = baseline.loc[model, metric_keys].astype(float)
    tuned_values = tuned.loc[model, metric_keys].astype(float)

    baseline_plot = [100.0] * len(metric_keys)
    tuned_plot = []

    for metric in metric_keys:
        base_value = baseline_values[metric]
        tuned_value = tuned_values[metric]

        if base_value == 0 or tuned_value == 0:
            tuned_plot.append(np.nan)
        elif metric in lower_is_better:
            tuned_plot.append(100 * base_value / tuned_value)
        else:
            tuned_plot.append(100 * tuned_value / base_value)

    baseline_plot += baseline_plot[:1]
    tuned_plot += tuned_plot[:1]

    angles = np.linspace(
        0, 2 * np.pi, len(metric_keys), endpoint=False
    ).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(7, 7),
        subplot_kw={"polar": True},
    )

    ax.plot(
        angles, baseline_plot,
        color="tab:blue", marker="o", label="Baseline"
    )
    ax.fill(angles, baseline_plot, color="tab:blue", alpha=0.10)

    ax.plot(
        angles, tuned_plot,
        color="tab:orange", marker="o", label="Tuned"
    )
    ax.fill(angles, tuned_plot, color="tab:orange", alpha=0.15)

    ax.axhline(100, color="gray", linewidth=1, linestyle="--")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels, fontsize=9)
    ax.set_title(
        f"{model}: baseline vs. tuned ({method_label})",
        pad=20
    )
    ax.set_ylabel("Relativ ytelse (baseline = 100 %)")
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    model_name = str(model).strip("_")
    model_method = method.strip("_")
    output_path = plots_dir / f"{timestamp}_{model_name}_{model_method}_radarplot.png"

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")
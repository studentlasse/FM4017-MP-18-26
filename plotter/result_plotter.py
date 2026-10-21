import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


def plot_results(
        x_data, 
        test_pred, 
        test_targets, 
        test_pred_label="Prediction", 
        test_target_label="Target", 
        title="Predictions vs. targets", 
        y_label="Value",
        filename_suffix = "pred_vs_targets"
        ):  

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(x_data, test_pred, label=test_pred_label, linewidth=1)
    ax.plot(x_data, test_targets, label=test_target_label, linewidth=1)

    ax.set(
        xlabel="Time",
        ylabel=y_label,
        title=title,
    )

    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"{timestamp}_{filename_suffix}.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")


def plot_barchart(results, metrics):
    
    models = list(results.keys())

    color_dict = {
        "xgb": "tab:red",
        "rf": "tab:green",
        "dummy": "tab:orange",
        "lstm": "tab:blue",
    }

    for metric in metrics:
        scores = [results[model][metric] for model in models]
        colors = [color_dict[model] for model in models]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(models, scores, color=colors, edgecolor="grey")
        ax.set_title(metric + " for models")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=25)
        #ax.bar_label(bars, fmt="%.3f", padding=3)
        fig.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{timestamp}_{metric}_results.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


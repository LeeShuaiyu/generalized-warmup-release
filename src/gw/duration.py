"""Tables and plots for the warm-up duration experiment."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from .presentation import load_predictions


def summarize(root, config, directory=None):
    runs = load_predictions(root, config, directory)
    epoch_map = {item["label"]: item["warmup_epochs"] for item in config["conditions"]}
    runs["Epochs"] = runs["Method"].map(epoch_map)
    summary = runs.groupby("Epochs")["ECE (%)"].agg(["mean", "std", "count"])
    if len(runs) != 33 or not (summary["count"] == 3).all():
        raise ValueError("Expected all eleven durations and three paired seeds")
    return runs, summary


def show_table(summary):
    table = pd.DataFrame({"ECE (%) ↓": [f"{row['mean']:.2f} ± {row['std']:.2f}" for _, row in summary.iterrows()]},
                         index=summary.index.rename("Warm-up epochs"))
    display(table.style.set_properties(**{"padding": "8px 16px", "font-size": "14px"}))


def comparison(root, summary):
    paper = pd.read_csv(Path(root) / "results/s7_length/paper.csv").set_index("Epochs")
    table = pd.DataFrame(index=summary.index.rename("Warm-up epochs"))
    table["Paper ECE (%)"] = [f"{row['Mean']:.2f} ± {row['SD']:.2f}" for _, row in paper.iterrows()]
    table["Recomputed ECE (%)"] = [f"{row['mean']:.2f} ± {row['std']:.2f}" for _, row in summary.iterrows()]
    table["Δ mean (pp)"] = summary["mean"] - paper["Mean"]
    table["Match"] = ((summary["mean"] - paper["Mean"]).abs() <= .005 + 1e-12) & ((summary["std"] - paper["SD"]).abs() <= .005 + 1e-12)
    return table


def plot(runs, summary, target):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.titlesize": 15, "axes.labelsize": 12, "figure.dpi": 130})
    figure, axis = plt.subplots(figsize=(8.4, 3.8), layout="constrained")
    axis.errorbar(summary.index, summary["mean"], yerr=summary["std"], color="#226b91",
                  marker="D", markersize=6, capsize=4, linewidth=1.5)
    for epoch in summary.index:
        values = runs.loc[runs["Epochs"] == epoch, "ECE (%)"].to_numpy()
        axis.scatter(epoch + np.linspace(-.11, .11, len(values)), values,
                     s=24, color="#667681", alpha=.5, zorder=3)
    axis.set_xticks(summary.index)
    axis.set_xlim(-.4, 10.4)
    axis.set_ylim(0, max(float((summary["mean"] + summary["std"]).max()), float(runs["ECE (%)"].max())) * 1.12)
    axis.set_xlabel("Warm-up epochs")
    axis.set_ylabel("ECE (%) ↓")
    axis.set_title("Calibration by warm-up duration", loc="left", pad=12)
    axis.grid(axis="y", color="#e4e8ec", linewidth=.7)
    axis.spines[["top", "right"]].set_visible(False)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    return figure

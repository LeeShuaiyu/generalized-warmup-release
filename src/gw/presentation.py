"""Concise table and figure presentation for the paper notebooks."""
from pathlib import Path
import hashlib
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display

from .metrics import classification_metrics

COLUMNS = ["ECE (%)", "Accuracy (%)", "Macro-F1 (%)", "NLL", "Brier"]
METRIC_KEYS = ["ece_15", "accuracy", "macro_f1", "nll", "brier"]
PRECISION = {"ECE (%)": 2, "Accuracy (%)": 1, "Macro-F1 (%)": 1, "NLL": 5, "Brier": 5}


def read_experiment(root, name):
    root = Path(root)
    return json.loads((root / "configs" / f"{name}.json").read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predictions(root, config, prediction_directory=None):
    root = Path(root)
    folder = Path(prediction_directory) if prediction_directory else root / "results" / config["experiment_id"] / "predictions"
    rows = []
    asset_manifest = json.loads((root / "artifacts" / "manifest.json").read_text())
    for condition in config["conditions"]:
        for seed in config["seeds"]:
            filename = f"{condition['id']}_{seed}.npz"
            path = folder / filename
            try:
                relative = path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                relative = None
            expected = asset_manifest["source_files"].get(relative)
            if expected and sha256(path) != expected["sha256"]:
                raise ValueError(f"Prediction checksum mismatch: {relative}")
            if prediction_directory is None and expected is None:
                raise ValueError(f"Unregistered prediction file: {filename}")
            with np.load(path, allow_pickle=False) as content:
                logits = content["logits"]
                labels = content["labels"]
            if logits.shape != (config["evaluation_samples"], config["num_classes"]):
                raise ValueError(f"Unexpected prediction shape: {filename}")
            if not np.isfinite(logits).all() or labels.shape != (logits.shape[0],):
                raise ValueError(f"Invalid predictions: {filename}")
            if not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels >= config["num_classes"])):
                raise ValueError(f"Invalid class labels: {filename}")
            metrics = classification_metrics(logits, labels, n_bins=config["ece_bins"])
            rows.append({"Method": condition["label"], "Seed": seed,
                         **{column: metrics[key] * (100 if i < 3 else 1)
                            for i, (key, column) in enumerate(zip(METRIC_KEYS, COLUMNS))}})
    frame = pd.DataFrame(rows)
    if frame.duplicated(["Method", "Seed"]).any():
        raise ValueError("Duplicated experiment runs")
    return frame


def summarize(frame, config):
    order = [condition["label"] for condition in config["conditions"]]
    grouped = frame.groupby("Method", sort=False)[COLUMNS]
    return grouped.mean().reindex(order), grouped.std(ddof=1).reindex(order)


def show_table(frame, highlight=True):
    styling = frame.style.format({column: f"{{:.{PRECISION[column]}f}}" for column in COLUMNS if column in frame})
    if highlight:
        for column in COLUMNS:
            if column in frame:
                styling = styling.highlight_min(subset=[column], props="font-weight:700;color:#17547a;background-color:#edf5fa;") if column in {"ECE (%)", "NLL", "Brier"} else styling.highlight_max(subset=[column], props="font-weight:700;color:#17547a;background-color:#edf5fa;")
    styling = styling.set_table_styles([
        {"selector": "table", "props": [("border-collapse", "collapse"), ("font-family", "Arial, sans-serif"), ("font-size", "14px")]},
        {"selector": "th", "props": [("background", "#f4f6f8"), ("padding", "10px 14px"), ("border-bottom", "2px solid #d5dce2"), ("text-align", "right")]},
        {"selector": "td", "props": [("padding", "10px 14px"), ("border-bottom", "1px solid #e4e8ec"), ("text-align", "right")]},
        {"selector": ".row_heading", "props": [("text-align", "left"), ("font-weight", "500")]}
    ])
    display(styling)


def calibration_plot(frame, mean, std, output_path=None):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.titlesize": 15, "axes.labelsize": 12,
                         "xtick.labelsize": 11, "ytick.labelsize": 12,
                         "svg.fonttype": "none", "figure.dpi": 130})
    figure, axis = plt.subplots(figsize=(8.4, 3.4), layout="constrained")
    positions = np.arange(len(mean))
    for position, method in enumerate(mean.index):
        values = frame.loc[frame["Method"] == method, "ECE (%)"].to_numpy()
        color = "#226b91" if method == "GW" else "#667681"
        axis.scatter(values, position + np.linspace(-.12, .12, len(values)), color=color,
                     alpha=.45, s=24, marker="o", linewidths=0, zorder=2)
        axis.errorbar(mean.loc[method, "ECE (%)"], position, xerr=std.loc[method, "ECE (%)"],
                      fmt="D", color=color, markersize=7, capsize=4, elinewidth=1.5, zorder=3)
        axis.text(1.015, position, f"{mean.loc[method, 'ECE (%)']:.2f}",
                  transform=axis.get_yaxis_transform(), va="center", color="#273540", fontsize=12)
    axis.set_yticks(positions, mean.index)
    axis.invert_yaxis()
    axis.set_xlim(0, math.ceil(float((mean["ECE (%)"] + std["ECE (%)"]).max())) * 1.08)
    axis.set_xlabel("ECE (%) ↓")
    axis.set_title("Calibration by warm-up input", loc="left", pad=14)
    axis.grid(axis="x", color="#e4e8ec", linewidth=.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, pad=10)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight", dpi=180)
    return figure


def paper_comparison(root, config, mean):
    paper = pd.read_csv(Path(root) / "results" / config["experiment_id"] / "paper.csv").set_index("Method")
    rows = []
    for method in mean.index:
        for metric in COLUMNS:
            expected = float(paper.loc[method, metric])
            actual = float(mean.loc[method, metric])
            precision = PRECISION[metric]
            rows.append({"Method": method, "Metric": metric, "Paper": expected,
                         "Recomputed": actual, "Absolute difference": abs(actual - expected),
                         "Matches displayed precision": abs(actual - expected) <= .5 * 10 ** (-precision) + 1e-12})
    return pd.DataFrame(rows)


def settings(config):
    return pd.DataFrame({"Setting": ["Dataset", "Model", "Evaluation", "Runs per condition", "Input", "Warm-up", "Supervised training", "Gate probability", "Strength"],
                         "Value": [config["dataset"], "DenseNet-121", f"Validation · {config['evaluation_samples']} images", str(len(config["seeds"])),
                                   "128 × 128 · grayscale", "5 epochs", "50 epochs", "0.5", "0.5"]}).set_index("Setting")

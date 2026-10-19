"""Recompute paired directional responses and their bootstrap intervals."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import direction_experiment as experiment
from .presentation import sha256


def recompute(root):
    root = Path(root)
    path = root / "results/s2_direction/paired_effects.csv"
    manifest = json.loads((root / "artifacts/manifest.json").read_text())
    if sha256(path) != manifest["source_files"][path.relative_to(root).as_posix()]["sha256"]:
        raise ValueError("Paired response data checksum mismatch")
    config = experiment.ExperimentConfig()
    raw = pd.read_csv(path)
    raw = raw[(raw.architecture == "linear") & (raw.coverage_v == config.primary_coverage_v)
              & (raw.baseline == "wrong_w") & (raw.supervised_horizon == 1000)].sort_values("seed")
    if len(raw) != 100 or raw.seed.nunique() != 100:
        raise ValueError("Expected one complete set of 100 paired seeds")
    summaries = []
    for time, key, count, role in [
        ("After warm-up", "targeted_minus_baseline_log_R_init", "targeted_lower_at_init", experiment.RNGRole.BOOTSTRAP_G1),
        ("Supervised step 1,000", "targeted_minus_baseline_log_R_t", "targeted_lower_at_t", experiment.RNGRole.BOOTSTRAP_G2),
    ]:
        values = raw[key].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite paired response")
        interval = experiment.paired_bootstrap_mean_ci(values, config.bootstrap_repetitions,
            experiment.hierarchical_rng(config, experiment.RNG_GLOBAL_SEED_INDEX, role, "linear"))
        summaries.append({"Time": time, "Mean": values.mean(), "Low": interval[0], "High": interval[1],
                          "Lower response": int(raw[count].sum()), "Seeds": len(raw)})
    return raw, pd.DataFrame(summaries).set_index("Time")


def table(summary):
    return pd.DataFrame({"Mean log difference Δ": summary["Mean"],
        "95% CI": [f"[{row['Low']:.4f}, {row['High']:.4f}]" for _, row in summary.iterrows()],
        "Lower response": [f"{int(row['Lower response'])}/{int(row['Seeds'])}" for _, row in summary.iterrows()]})


def comparison(root, summary):
    paper = pd.read_csv(Path(root) / "results/s2_direction/paper.csv").set_index("Time")
    matches = np.all(np.abs(summary[["Mean", "Low", "High"]].to_numpy() - paper[["Mean", "Low", "High"]].to_numpy()) <= .00005 + 1e-12, axis=1)
    matches &= summary["Lower response"].to_numpy() == paper["Lower response"].to_numpy()
    return pd.DataFrame({"Paper Δ": paper["Mean"], "Recomputed Δ": summary["Mean"], "Match": matches}, index=summary.index)


def plot(summary, target):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.titlesize": 15, "figure.dpi": 130})
    figure, axis = plt.subplots(figsize=(8.4, 2.8), layout="constrained")
    y = np.arange(len(summary))
    errors = np.stack([summary["Mean"] - summary["Low"], summary["High"] - summary["Mean"]])
    axis.errorbar(summary["Mean"], y, xerr=errors, fmt="D", color="#226b91", capsize=5, markersize=7)
    axis.axvline(0, color="#667681", linewidth=1, linestyle="--")
    axis.set_yticks(y, summary.index)
    axis.invert_yaxis()
    axis.set_ylim(1.6, -.6)
    axis.set_xlim(float(summary["Low"].min()) - 1, 1)
    axis.set_xlabel("Paired mean log difference Δ")
    axis.set_title("Directional response", loc="left", pad=12)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", color="#e4e8ec", linewidth=.7)
    axis.tick_params(axis="y", length=0)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    return figure

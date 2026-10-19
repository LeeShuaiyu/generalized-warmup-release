"""Combine training-time records with checkpoint-derived post-hoc evaluations."""
from pathlib import Path

import pandas as pd

from . import classification_report as report, posthoc, training_compat_report
from .presentation import COLUMNS


def calculate(root, posthoc_directory=None):
    root = Path(root)
    training, _, _ = training_compat_report.calculate(root)
    directory = (root/'results/posthoc_compat/model_refit_fp32'
                 if posthoc_directory is None else Path(posthoc_directory))
    calibrated, _, _ = posthoc.summarize(root, directory)
    runs = pd.concat([training, calibrated], ignore_index=True)
    config = report.read(root, 'compatibility')
    order = pd.MultiIndex.from_tuples([tuple(r) for r in config['row_order']], names=report.INDEX)
    if runs.duplicated(report.INDEX+['Seed']).any():
        raise ValueError('Duplicate compatibility evaluation')
    groups = runs.groupby(report.INDEX, sort=False)[COLUMNS]
    if not (groups.size() == config['seeds_per_condition']).all():
        raise ValueError('Incomplete compatibility seed coverage')
    if set(groups.size().index) != set(order):
        raise ValueError('Compatibility conditions do not match the configuration')
    return runs, groups.mean().reindex(order), groups.std(ddof=1).reindex(order)

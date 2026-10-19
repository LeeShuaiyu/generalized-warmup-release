"""Prediction-backed multi-dataset tables and paired calibration plots."""
from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from .metrics import classification_metrics
from .presentation import sha256, COLUMNS, METRIC_KEYS

INDEX = ['Dataset', 'Model', 'Method']
DISPLAY = ['ECE (%) ↓', 'Accuracy (%) ↑', 'Macro-F1 (%) ↑', 'NLL ↓', 'Brier ↓']


def read(root, experiment):
    return json.loads((Path(root) / 'configs' / f'{experiment}.json').read_text())


def select_dataset(config, dataset=None):
    if dataset is None:
        return config
    runs = [r for r in config['runs'] if r['dataset'] == dataset]
    if not runs:
        raise ValueError(f'Unknown dataset: {dataset}')
    return dict(config, runs=runs, row_order=[r for r in config['row_order'] if r[0] == dataset])


def select_reference(reference, config):
    selected = {tuple(r) for r in config['row_order']}
    rows = [r for r in reference['rows'] if tuple(r['key']) in selected]
    if {tuple(r['key']) for r in rows} != selected:
        raise ValueError('Reference does not cover the selected conditions')
    return dict(reference, rows=rows)


def calculate(root, config, directory=None):
    root = Path(root)
    entries = json.loads((root / 'artifacts/manifest.json').read_text())['source_files']
    evaluated = None
    if directory is not None:
        execution = Path(directory) / 'run.json'
        if not execution.is_file():
            raise ValueError('Model evaluation identity record is required')
        if execution.resolve().is_relative_to(root.resolve()):
            relative = execution.resolve().relative_to(root.resolve()).as_posix()
            if relative in entries and sha256(execution) != entries[relative]['sha256']:
                raise ValueError('Saved GPU execution identity changed')
        records = json.loads(execution.read_text())['records']
        evaluated = {r['id']: r for r in records}
        if len(evaluated) != len(records):
            raise ValueError('Duplicate evaluated run identity')
    rows = []
    for record in config['runs']:
        if record.get('metrics_file') and directory is None:
            relative = record['metrics_file']
            path = root/relative
            if sha256(path) != entries[relative]['sha256']:
                raise ValueError('Evaluation record checksum mismatch')
            stored = json.loads(path.read_text())
            if stored['seed'] != record['seed'] or stored['checkpoint_sha256'] != record['checkpoint_sha256']:
                raise ValueError('Evaluation identity mismatch')
            rows.append(dict(zip(INDEX,[record['dataset'],record['model'],record['method']]),
                             Seed=record['seed'],Samples=record['samples'],
                             **{col:stored['metrics'][key]*(100 if i<3 else 1)
                                for i,(col,key) in enumerate(zip(COLUMNS,METRIC_KEYS))}))
            continue
        path = Path(directory) / f"{record['id']}.npz" if directory else root / record['predictions']
        if evaluated is not None:
            identity = evaluated.get(record['id'])
            if identity is None or identity['seed'] != record['seed'] or identity['samples'] != record['samples']:
                raise ValueError('Incomplete or inconsistent model evaluation')
            if identity['checkpoint_sha256'] != record['checkpoint_sha256'] or sha256(path) != identity['predictions_sha256']:
                raise ValueError('Model evaluation file identity changed')
        if directory is None:
            if sha256(path) != entries[record['predictions']]['sha256']:
                raise ValueError(f"Prediction checksum mismatch: {record['id']}")
        with np.load(path, allow_pickle=False) as saved:
            logits, labels = saved[record['logits_key']], saved[record['labels_key']]
        if logits.shape != (record['samples'], record['classes']):
            raise ValueError(f"Unexpected prediction dimensions: {record['id']}")
        if labels.shape != (len(logits),) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"Invalid labels: {record['id']}")
        if not np.isfinite(logits).all() or np.any((labels < 0) | (labels >= record['classes'])):
            raise ValueError(f"Invalid predictions: {record['id']}")
        metrics = classification_metrics(logits, labels, n_bins=config['ece_bins'])
        rows.append(dict(zip(INDEX, [record['dataset'], record['model'], record['method']]),
                         Seed=record['seed'], Samples=len(labels),
                         **{col: metrics[key] * (100 if i < 3 else 1)
                            for i, (col, key) in enumerate(zip(COLUMNS, METRIC_KEYS))}))
    runs = pd.DataFrame(rows)
    if runs.duplicated(INDEX + ['Seed']).any():
        raise ValueError('Duplicate run identity')
    groups = runs.groupby(INDEX, sort=False)[COLUMNS]
    if not (groups.size() == config['seeds_per_condition']).all():
        raise ValueError('Incomplete seed coverage')
    order = pd.MultiIndex.from_tuples([tuple(row) for row in config['row_order']], names=INDEX)
    return runs, groups.mean().reindex(order), groups.std(ddof=1).reindex(order)


def paper(root, notebook):
    return json.loads((Path(root) / 'results/paper' / f'{notebook}.json').read_text())


def saved_evaluation(root, config):
    """Prefer complete saved model evaluations, without hiding partial coverage."""
    root = Path(root)
    directory = root / config.get('saved_evaluation_directory',
        f"results/{config['experiment_id']}/gpu_predictions")
    execution = directory / 'run.json'
    evaluated = []
    if execution.exists():
        evaluated = json.loads(execution.read_text())['records']
        ids = [row['id'] for row in evaluated]
        known = {row['id'] for row in config['runs']}
        if len(ids) != len(set(ids)) or not set(ids) <= known:
            raise ValueError('Duplicate or unknown saved model evaluation')
    available = {row['id'] for row in evaluated}
    frames, sources = [], []
    for dataset in dict.fromkeys(row['dataset'] for row in config['runs']):
        subset = select_dataset(config, dataset)
        expected = {row['id'] for row in subset['runs']}
        present = expected & available
        if present and present != expected:
            raise ValueError(f'Incomplete saved model evaluations for {dataset}')
        use_models = bool(present)
        runs, _, _ = calculate(root, subset, directory if use_models else None)
        frames.append(runs)
        for model in dict.fromkeys(row['model'] for row in subset['runs']):
            count = sum(row['model'] == model for row in subset['runs'])
            sources.append({'Dataset': dataset, 'Model': model, 'Runs': count,
                            'Model evaluations': count if use_models else 0,
                            'Source': ('Replayed checkpoint evaluation' if use_models and
                                any(r.get('checkpoint_kind')=='retrained_checkpoint' for r in subset['runs'])
                                else 'Saved model evaluation' if use_models else 'Historical predictions')})
    runs = pd.concat(frames, ignore_index=True)
    grouped = runs.groupby(INDEX, sort=False)[COLUMNS]
    order = pd.MultiIndex.from_tuples([tuple(row) for row in config['row_order']], names=INDEX)
    return runs, grouped.mean().reindex(order), grouped.std(ddof=1).reindex(order), pd.DataFrame(sources).set_index(['Dataset', 'Model'])


def table_frame(mean, sd, reference):
    columns = reference['columns']
    result = pd.DataFrame(index=mean.index)
    for column in columns:
        label = DISPLAY[COLUMNS.index(column)]
        values = []
        for key,a,b in zip(mean.index,mean[column],sd[column]):
            entry = next(x for x in reference['rows'] if tuple(x['key'])==key)
            precision = entry.get('precision',reference['precision'])[column]
            values.append(f'{a:.{precision}f} ± {b:.{precision}f}')
        result[label] = values
    return result


def show(frame, formatters=None):
    display(frame.style.format(formatters).set_table_styles([
        {'selector': 'th', 'props': [('padding', '9px 12px'), ('background', '#f3f6f8'),
                                    ('text-align', 'left'), ('border-bottom', '1px solid #d6dfe5')]},
        {'selector': 'td', 'props': [('padding', '9px 12px'), ('text-align', 'right'),
                                    ('white-space', 'nowrap'),
                                    ('border-bottom', '1px solid #e4e8ec')]},
        {'selector': 'table', 'props': [('font-family', 'Arial, sans-serif'), ('font-size', '14px')]},
    ]))


def comparison(mean, sd, reference):
    rows = []
    for entry in reference['rows']:
        key = tuple(entry['key'])
        for column in reference['columns']:
            expected = entry[column]
            actual = [float(mean.loc[key, column]), float(sd.loc[key, column])]
            digits = entry.get('precision',reference['precision'])[column]
            rows.append(dict(zip(INDEX, key), Metric=DISPLAY[COLUMNS.index(column)],
                             Paper=f'{expected[0]:.{digits}f} ± {expected[1]:.{digits}f}',
                             Recomputed=f'{actual[0]:.{digits}f} ± {actual[1]:.{digits}f}',
                             **{'Δ mean': actual[0] - expected[0],
                                'Matches': all(abs(a-b) <= .5 * 10**(-digits) + 1e-10
                                               for a, b in zip(actual, expected))}))
    return pd.DataFrame(rows).set_index(INDEX + ['Metric'])


def plot(runs, mean, sd, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none'})
    groups = list(dict.fromkeys((r[0], r[1]) for r in mean.index))
    figure, axes = plt.subplots((len(groups)+1)//2, 2, figsize=(10.4, 2.2*((len(groups)+1)//2)),
                                layout='constrained', squeeze=False)
    for axis, group in zip(axes.flat, groups):
        methods = mean.xs(group, level=['Dataset','Model']).index.tolist()
        for position, method in enumerate(methods):
            key = (*group, method)
            color = '#226b91' if method == 'GW' else '#73808a'
            selected = runs[(runs.Dataset == group[0]) & (runs.Model == group[1]) & (runs.Method == method)]
            axis.scatter(selected['ECE (%)'], position + np.linspace(-.12,.12,len(selected)),
                         color=color, alpha=.4, s=15, linewidths=0)
            axis.errorbar(mean.loc[key, 'ECE (%)'], position, xerr=sd.loc[key, 'ECE (%)'],
                          fmt='D', color=color, markersize=5, capsize=3)
        axis.set_yticks(range(len(methods)), methods)
        axis.invert_yaxis()
        axis.set_xlim(left=0)
        axis.set_xlabel('ECE (%) ↓')
        axis.set_title(f'{group[0]} · {group[1]}', loc='left', fontsize=12)
        axis.spines[['top','right','left']].set_visible(False)
        axis.grid(axis='x', color='#e5eaee', linewidth=.6)
        axis.set_axisbelow(True)
        axis.tick_params(axis='y', length=0)
    for axis in list(axes.flat)[len(groups):]:
        axis.set_visible(False)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches='tight')
    return figure


def settings(config):
    rows = []
    for record in config['runs']:
        c = record['settings']
        match=re.search(r'p(\d+)_l([\d.]+)_(?:r|rho)(\d+)',c.get('method',''))
        parameters={} if match is None else {'L':float(match[2]),'p':int(match[1])/100,'ρ':int(match[3])/100}
        rows.append(dict(zip(INDEX, [record['dataset'],record['model'],record['method']]),
                         Seed=record['seed'], Images=record['samples'],
                         **{key: c.get(key, '') for key in ['image_size','epochs','warmup_epochs','batch_size',
                              'lr','weight_decay']},**parameters))
    return pd.DataFrame(rows).set_index(INDEX + ['Seed']).rename(columns={
        'image_size':'Input size','epochs':'Supervised epochs','warmup_epochs':'Warm-up epochs',
        'batch_size':'Batch size','lr':'Learning rate','weight_decay':'Weight decay'})


def settings_html(config):
    """Readable presentation only; absent parameters stay absent in the data."""
    return settings(config).to_html(border=0, na_rep='—', sparsify=False,
                                    float_format=lambda value: f'{value:g}')

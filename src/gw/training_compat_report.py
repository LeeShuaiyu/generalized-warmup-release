"""Combine explicit checkpoint evaluations with the remaining archived records."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from . import classification_report as report
from .metrics import classification_metrics
from .presentation import sha256, COLUMNS, METRIC_KEYS
from .warmup import parse_method


def calculate(root, directory=None):
    root = Path(root)
    config = report.read(root, 'training_compat')
    runs, _, _ = report.calculate(root, config)
    directory = root/config.get('saved_evaluation_directory',
        'results/training_compat/gpu_predictions') if directory is None else Path(directory)
    if not (directory/'run.json').exists():
        return report.calculate(root, config)
    state = json.loads((directory/'run.json').read_text())
    if state.get('status') != 'complete' or state['experiment'] != 'training_compat':
        raise ValueError('Only complete compatibility evaluation output is accepted')
    manifest = json.loads((root/'artifacts/manifest.json').read_text())['source_files']
    checked = directory/'run.json'
    if checked.resolve().is_relative_to(root.resolve()):
        relative = checked.resolve().relative_to(root.resolve()).as_posix()
        if relative in manifest and sha256(checked) != manifest[relative]['sha256']:
            raise ValueError('Evaluation record changed')
    ids = [r['id'] for r in state['records']]
    if len(set(ids)) != len(ids) or set(ids) != set(state['selected_ids']):
        raise ValueError('Incomplete or duplicate evaluated identities')
    lookup = {r['id']:r for r in config['runs']}
    for identity in state['records']:
        record = lookup[identity['id']]
        if (identity['seed'], identity['samples'], identity['checkpoint_sha256']) != (
                record['seed'], record['samples'], record['checkpoint_sha256']):
            raise ValueError('Checkpoint evaluation identity differs')
        path = directory/f'{record["id"]}.npz'
        if sha256(path) != identity['predictions_sha256']:
            raise ValueError('Prediction hash differs')
        with np.load(path, allow_pickle=False) as predictions:
            logits, labels = predictions['logits'], predictions['labels']
        if logits.shape != (record['samples'], record['classes']) or labels.shape != (record['samples'],):
            raise ValueError('Prediction dimensions differ')
        metrics = classification_metrics(logits, labels, n_bins=config['ece_bins'])
        selected = (runs.Dataset==record['dataset']) & (runs.Model==record['model']) & (
            runs.Method==record['method']) & (runs.Seed==record['seed'])
        if selected.sum() != 1:
            raise ValueError('Evaluation does not bind to exactly one row')
        for i, (column,key) in enumerate(zip(COLUMNS,METRIC_KEYS)):
            runs.loc[selected,column] = metrics[key] * (100 if i<3 else 1)
    groups = runs.groupby(report.INDEX, sort=False)[COLUMNS]
    if not (groups.size() == config['seeds_per_condition']).all():
        raise ValueError('Incomplete seed coverage')
    order = pd.MultiIndex.from_tuples([tuple(r) for r in config['row_order']], names=report.INDEX)
    return runs, groups.mean().reindex(order), groups.std(ddof=1).reindex(order)


def coverage(root, directory=None):
    root = Path(root)
    config = report.read(root,'training_compat')
    directory = root/config.get('saved_evaluation_directory',
        'results/training_compat/gpu_predictions') if directory is None else Path(directory)
    evaluated = set()
    if (directory/'run.json').exists():
        state = json.loads((directory/'run.json').read_text())
        if state.get('status') != 'complete':
            raise ValueError('Evaluation did not complete')
        evaluated = {r['id'] for r in state['records']}
    rows = []
    for key in config['row_order']:
        selected = [r for r in config['runs'] if [r['dataset'],r['model'],r['method']]==key]
        count = sum(r['id'] in evaluated for r in selected)
        rows.append(dict(Method=key[2], **{'Checkpoint inference':f'{count}/{len(selected)}',
                                         'Saved evaluations':f'{len(selected)-count}/{len(selected)}'}))
    return pd.DataFrame(rows).set_index('Method')


def settings(root):
    """Display the resolved training settings, including condition-specific losses."""
    config = report.read(Path(root), 'training_compat')
    rows = []
    for record in config['runs']:
        c = record['training_settings']
        spec = parse_method(c['method'])
        warmup = spec.family != 'none'
        rows.append(dict(Method=record['method'], Seed=record['seed'], **{
            'Input size': c['image_size'], 'Supervised epochs': c['epochs'],
            'Warm-up epochs': c['warmup_epochs'] if warmup else 0,
            'Batch size': c['batch_size'], 'Learning rate': c['lr'],
            'Weight decay': c['weight_decay'], 'Label smoothing': c['label_smoothing'],
            'MMCE strength': c['mmce_alpha'],
            'L': spec.shape if warmup else np.nan,
            'p': spec.probability if warmup else np.nan,
            'ρ': spec.rho if warmup else np.nan,
            'Selection': c['checkpoint_metric'].replace('val_nll', 'Validation NLL')}))
    return pd.DataFrame(rows).set_index(['Method', 'Seed'])

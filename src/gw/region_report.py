"""Readable checks of completed, full homogeneous-region reconstructions."""
from pathlib import Path
import json
import pandas as pd

from .assets import file_hash
from .distribution_fitting import rebuilt_input
from .supporting_reports import checked


def summary(root, directories=None):
    root = Path(root)
    config = json.loads(checked(root, 'configs/fit.json').read_text())
    directories = directories or {d['id']: root / 'results/fit/regions' / d['id'] for d in config['datasets']}
    if set(directories) != {d['id'] for d in config['datasets']}:
        raise ValueError('Supply one complete run per dataset')
    rows = []
    for item in config['datasets']:
        directory = Path(directories[item['id']])
        if directory.resolve() == (root / 'results/fit/regions' / item['id']).resolve():
            for name in ['run.json', 'bank.npz', 'records.json', 'image_audit.json', 'comparison.json', 'implementation.py']:
                checked(root, f'results/fit/regions/{item["id"]}/{name}')
            state = json.loads((directory / 'run.json').read_text())
            if file_hash(directory / 'implementation.py') != state['code_sha256']:
                raise ValueError('Selection execution-source snapshot changed')
        bank, records = rebuilt_input(root, item, directory)
        state = json.loads((directory / 'run.json').read_text())
        if len(records) != state['selected'] or len(state['input_files']) != state['processed_images']:
            raise ValueError('Selection coverage differs from run record')
        rows.append({'Dataset': item['label'], 'Train images': state['processed_images'],
                     'Blocks': len(records), 'Groups': len({r['group_id'] for r in records}), 'Matches': '✓'})
    return pd.DataFrame(rows).set_index('Dataset')

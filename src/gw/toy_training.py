"""One-seed replay of the archived NumPy random-label warm-up experiment."""
from pathlib import Path
import csv
from datetime import datetime, timezone
import json
import platform
import shutil

import numpy as np

from .assets import file_hash
from .toy_model import (ToyModel, init_model, train_random_labels, confidence_metrics,
                        projected_response_metrics, unit_directions)


def load_model(path):
    dimensions = (2, 10, 10, 2)
    required = {f'{kind}_{i}' for kind in ['weight', 'bias'] for i in range(3)}
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != required:
            raise ValueError('Unexpected toy model arrays')
        weights = [data[f'weight_{i}'].copy() for i in range(3)]
        biases = [data[f'bias_{i}'].copy() for i in range(3)]
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        if weight.shape != (dimensions[i], dimensions[i+1]) or bias.shape != (dimensions[i+1],):
            raise ValueError('Toy model dimensions changed')
        if any(a.dtype != np.float64 or not np.isfinite(a).all() for a in (weight, bias)):
            raise ValueError('Toy model requires finite float64 arrays')
    return ToyModel(weights, biases)


def same_model(left, right):
    return all(np.array_equal(a, b) for a, b in
               zip(left.weights + left.biases, right.weights + right.biases))


def configuration(root):
    config = json.loads((root / 'configs/toy1_training.json').read_text())
    expected = dict(seed_offset=1, steps=12000, snapshot_steps=[0, 250, 1000, 3000, 6000, 12000],
                    batch_size=2048, inputs='fresh_uniform_minus10_plus10',
                    labels='exactly_balanced_random_hard_labels', optimizer='Adam',
                    learning_rate=.0008, beta1=.9, beta2=.999, epsilon=1e-8)
    evaluation = dict(point_seed=20260709, points=10000, domain=[-10., 10.],
                      directions=32, response_epsilon=1e-12)
    if (config['experiment'] != 'toy1' or config['dimensions'] != [2, 10, 10, 2]
            or config['seeds'] != list(range(100, 130)) or config['warmup'] != expected
            or config['evaluation'] != evaluation or config['backend'] != 'numpy_float64_cpu'):
        raise ValueError('Toy training protocol changed')
    if file_hash(root / 'src/gw/toy_model.py') != config['implementation_sha256']:
        raise ValueError('Toy implementation identity changed')
    return config


def measure(model, evaluation):
    points = np.random.default_rng(evaluation['point_seed']).uniform(
        *evaluation['domain'], size=(evaluation['points'], 2))
    return {**confidence_metrics(model, points),
            **projected_response_metrics(model, points, unit_directions(evaluation['directions']),
                                         epsilon=evaluation['response_epsilon'])}


def train(root, seed, *, steps=None, destination=None):
    """Default to the full 12,000 updates; a shorter explicit budget is recorded."""
    root = Path(root).resolve()
    config = configuration(root)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed not in config['seeds']:
        raise ValueError('Choose one of the thirty registered seeds')
    warmup = config['warmup']
    actual_steps = warmup['steps'] if steps is None else steps
    if (not isinstance(actual_steps, int) or isinstance(actual_steps, bool)
            or not 0 <= actual_steps <= warmup['steps']):
        raise ValueError('Update budget must be an integer from 0 to 12000')
    relative = f'artifacts/toy1/{seed}/initial.npz'
    initial_file = root / relative
    catalog = json.loads((root / 'artifacts/manifest.json').read_text())['source_files']
    if file_hash(initial_file) != catalog[relative]['sha256']:
        raise ValueError('Original initialization file identity changed')
    initial = init_model(seed, tuple(config['dimensions']))
    if not same_model(initial, load_model(initial_file)):
        raise ValueError('Initialization differs from the original saved state')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target = Path(destination) if destination is not None else root / 'output/toy1' / stamp
    ancestor = target.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < 8*1024**3 + 1024**2:
        raise RuntimeError('Keep at least 8 GiB free before starting')
    target.mkdir(parents=True, exist_ok=False)
    run = dict(status='running', experiment='toy1', seed=seed, backend=config['backend'],
               configuration=config, steps=actual_steps, full_budget=actual_steps==warmup['steps'],
               initial_file_sha256=file_hash(initial_file), historical_results_replaced=False,
               config_sha256=file_hash(root/'configs/toy1_training.json'),
               environment=dict(python=platform.python_version(), numpy=np.__version__))
    (target / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    try:
        snapshot_steps = tuple(sorted({step for step in warmup['snapshot_steps'] if step <= actual_steps}
                                      | {actual_steps}))
        snapshots = train_random_labels(initial, seed=seed + warmup['seed_offset'], steps=actual_steps,
                                        batch_size=warmup['batch_size'], learning_rate=warmup['learning_rate'],
                                        beta1=warmup['beta1'], beta2=warmup['beta2'], epsilon=warmup['epsilon'],
                                        snapshot_steps=snapshot_steps)
        rows, files = [], []
        for step, model in snapshots.items():
            path = target / f'step_{step:05d}.npz'
            arrays = {f'weight_{i}': a for i, a in enumerate(model.weights)}
            arrays.update({f'bias_{i}': a for i, a in enumerate(model.biases)})
            with path.open('xb') as output:
                np.savez(output, **arrays)
            reloaded = load_model(path)
            if not same_model(model, reloaded):
                raise ValueError('Saved weights differ after reload')
            values = measure(reloaded, config['evaluation'])
            rows.append(dict(seed=seed, step=step, **values))
            files.append(dict(file=path.name, sha256=file_hash(path), reload_exact=True))
        with (target / 'metrics.csv').open('x', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        run.update(status='complete', models=files, snapshot_steps=list(snapshot_steps),
                   metrics_sha256=file_hash(target/'metrics.csv'))
        (target / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    except Exception as error:
        run.update(status='failed', error=f'{type(error).__name__}: {error}')
        (target / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
        raise
    return target

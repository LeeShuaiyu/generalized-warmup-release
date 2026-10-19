"""Evaluate mapped training-calibration checkpoints on the fixed test split."""
from pathlib import Path
import datetime
import json
import platform
import shutil

import numpy as np

from .assets import file_hash
from .classification_data import load_manifest, manifest_sha256, ManifestDataset
from .classification_models import build_model
from .data_assets import prepare
from .training_stages import collect_logits


def run(root, config, *, device='auto', run_ids=None, available_only=False):
    import torch
    root = Path(root).resolve()
    ids = None if run_ids is None else set(run_ids)
    if ids is not None and not ids.issubset({r['id'] for r in config['runs']}):
        raise ValueError('Unknown run IDs')
    jobs = [r for r in config['runs'] if ids is None or r['id'] in ids]
    missing = [r['id'] for r in jobs if 'checkpoint' not in r]
    if missing and not available_only:
        raise ValueError(f'Checkpoint mappings missing: {missing}; select mapped IDs explicitly')
    jobs = [r for r in jobs if 'checkpoint' in r]
    if not jobs:
        raise ValueError('No mapped checkpoints selected')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device not in {'cpu', 'cuda'} or device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested device unavailable')
    if config['evaluation_split'] != 'test':
        raise ValueError('This entry evaluates only the fixed test split')
    rows = load_manifest(root / 'data/manifests/opensar.csv')
    split_hash = manifest_sha256(root / 'data/manifests/opensar.csv')
    checked = prepare(root, 'opensar', check=True, split='test')
    if checked['missing'] or checked['invalid']:
        raise ValueError(f'Prepare the fixed test files: {checked}')
    for record in jobs:
        if record['dataset_key'] != 'opensar' or record['split_sha256'] != split_hash:
            raise ValueError('Dataset identity differs')
        if file_hash(root / record['checkpoint']) != record['checkpoint_sha256']:
            raise ValueError(f'Checkpoint identity differs: {record["id"]}')
    if shutil.disk_usage(root).free < 8 * 1024**3 + 10_000_000:
        raise RuntimeError('Keep 8 GiB available')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output = root / 'output/training_compat' / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', experiment='training_compat', selected_ids=[r['id'] for r in jobs],
                 unmapped_ids=missing, device=device, split_sha256=split_hash, records=[],
                 config_sha256=file_hash(root/'configs/training_compat.json'),
                 code_sha256={name:file_hash(Path(__file__).parent / name) for name in
                              ['training_compat_inference.py', 'classification_data.py', 'classification_models.py', 'training_stages.py']},
                 environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                                  cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0) if device=='cuda' else None))
    try:
        for i, record in enumerate(jobs, 1):
            c = record['training_settings']
            dataset = ManifestDataset(rows, 'test', image_size=c['image_size'], data_root=root/'data/processed/opensar')
            if len(dataset) != record['samples']:
                raise ValueError('Sample count differs')
            loader = torch.utils.data.DataLoader(dataset, batch_size=c['batch_size'], shuffle=False, num_workers=0)
            model = build_model(c['architecture'], record['classes'], 'none', c['image_size'])
            model.load_state_dict(torch.load(root / record['checkpoint'], map_location='cpu', weights_only=True), strict=True)
            model = model.to(device)
            logits, labels = collect_logits(model, loader, device, torch)
            expected_labels = np.array([int(r['class_id']) for r in dataset.rows], dtype=np.int64)
            if not np.array_equal(labels, expected_labels) or not np.isfinite(logits).all():
                raise ValueError('Prediction order or finiteness differs')
            path = output / f'{record["id"]}.npz'
            np.savez_compressed(path, logits=logits, labels=labels)
            state['records'].append(dict(id=record['id'], seed=record['seed'], samples=len(labels), device=device,
                                        checkpoint_sha256=record['checkpoint_sha256'], predictions_sha256=file_hash(path)))
            (output / 'run.json').write_text(json.dumps(state, indent=2) + '\n')
            print(f'Inference {i}/{len(jobs)} · {record["method"]} · {record["seed"]}', flush=True)
            del model
        state['status'] = 'complete'
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (output / 'run.json').write_text(json.dumps(state, indent=2) + '\n')
    return output

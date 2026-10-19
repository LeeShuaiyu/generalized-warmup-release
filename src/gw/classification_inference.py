"""Evaluate fixed classification checkpoints without altering sources."""
from pathlib import Path
from contextlib import contextmanager
import datetime
import json
import shutil
import numpy as np

from .assets import file_hash as sha256
from .classification_data import load_manifest, manifest_sha256, ManifestDataset
from .classification_models import build_model
from .data_assets import verify_inputs
from .assets import relative_path


def preflight(root, config, run_ids=None):
    """Validate identities and input bytes before importing or constructing models."""
    root = Path(root).resolve()
    ids = [r['id'] for r in config['runs']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate configured run IDs')
    requested = None if run_ids is None else set(run_ids)
    if requested is not None and not requested.issubset(set(ids)):
        raise ValueError('Unknown run IDs')
    jobs = [r for r in config['runs'] if requested is None or r['id'] in requested]
    if not jobs:
        raise ValueError('No selected runs')
    datasets, checked = {}, {}
    for r in jobs:
        if 'checkpoint' not in r:
            raise ValueError(f"Checkpoint mapping required: {r['id']}")
        path = root / relative_path(r['checkpoint'])
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != r['checkpoint_sha256']:
            raise ValueError(f"Checkpoint identity mismatch: {r['id']}")
        ds = r['dataset_key']
        relative_path(ds)
        manifest = root / relative_path(r.get('manifest', f'data/manifests/{ds}.csv'))
        if manifest_sha256(manifest) != r['split_sha256']:
            raise ValueError(f'Split identity mismatch: {ds}')
        if ds not in datasets:
            rows = datasets[ds] = load_manifest(manifest)
            checked[ds] = verify_inputs(root, ds, rows, config['evaluation_split'],
                                        root / 'data/processed' / ds)
            if r.get('inference_protocol') == 'legacy_all_splits':
                checked[ds]['context_splits'] = [verify_inputs(root, ds, rows, split,
                    root / 'data/processed' / ds) for split in ['train', 'val']]
        if checked[ds]['samples_verified'] != r['samples']:
            raise ValueError('Evaluation sample count mismatch')
    output_base = root / 'output' / relative_path(config['experiment_id'])
    if not output_base.resolve().is_relative_to(root):
        raise ValueError('Inference output resolves outside the project')
    if shutil.disk_usage(root).free < 8 * 1024**3 + sum(r['samples'] * (r['classes'] * 4 + 8) for r in jobs):
        raise RuntimeError('Keep at least 8 GiB available after inference')
    return jobs, datasets, checked


def evaluation_dataset(root, record, rows, split='test'):
    """Retain the replay runner's quantization and all-split batch boundaries."""
    c = record['settings']
    data_root = Path(root) / 'data/processed' / record['dataset_key']
    protocol = record.get('inference_protocol', 'test_split')
    if split not in {'train', 'val', 'test'}:
        raise ValueError('Unknown evaluation split')
    if protocol == 'test_split':
        dataset = ManifestDataset(rows, split, image_size=c['image_size'], data_root=data_root)
        return dataset, None
    if protocol != 'legacy_all_splits':
        raise ValueError('Unknown checkpoint inference protocol')
    from .imagenet_data import ImageDataset
    from torch.utils.data import ConcatDataset
    parts = [ImageDataset(rows, split, image_size=c['image_size'], data_root=data_root)
             for split in ['train', 'val', 'test']]
    ordered = [row for part in parts for row in part.rows]
    return ConcatDataset(parts), np.array([r['split'] == split for r in ordered])


@contextmanager
def precision_settings(torch, precision):
    """Explicit IEEE FP32 for sensitive calibration; restore caller settings."""
    if precision not in {'default', 'fp32'}:
        raise ValueError('Unknown inference precision')
    previous = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    try:
        if precision == 'fp32':
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        yield dict(precision=precision, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                   matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous


def run(root, config, device='auto', run_ids=None):
    if device not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('Unknown device')
    root = Path(root).resolve()
    precision = config.get('inference_precision', 'default')
    if precision not in {'default', 'fp32'}:
        raise ValueError('Unknown inference precision')
    jobs, datasets, checked = preflight(root, config, run_ids)
    import torch
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output = root / 'output' / config['experiment_id'] / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True,exist_ok=False)
    records = []
    for i,r in enumerate(jobs,1):
        c = r['settings']
        dataset, test_mask = evaluation_dataset(root, r, datasets[r['dataset_key']], config['evaluation_split'])
        loader = torch.utils.data.DataLoader(dataset,batch_size=c.get('evaluation_batch_size', c['batch_size']),shuffle=False,num_workers=0)
        model = build_model(c['architecture'],r['classes'],'none',c['image_size'])
        model.load_state_dict(torch.load(root/r['checkpoint'],map_location='cpu',weights_only=True),strict=True)
        model = model.to(device).eval()
        logits,labels = [],[]
        with precision_settings(torch, precision) as backend, torch.no_grad():
            for images,target in loader:
                logits.append(model(images.to(device)).cpu().numpy())
                labels.append(target.numpy())
        predicted,actual = np.concatenate(logits),np.concatenate(labels)
        if test_mask is not None:
            predicted, actual = predicted[test_mask], actual[test_mask]
        with np.load(root/r['predictions'],allow_pickle=False) as saved:
            if not np.array_equal(actual,saved[r['labels_key']]):
                raise ValueError(f"Label order mismatch: {r['id']}")
        destination = output / f"{r['id']}.npz"
        np.savez_compressed(destination,**{r['logits_key']:predicted,r['labels_key']:actual})
        records.append({'id':r['id'],'seed':r['seed'],'device':device,'samples':len(actual),
                        'evaluation_split':config['evaluation_split'], 'backend':backend,
                        'torch_version':torch.__version__, 'cuda_version':torch.version.cuda,
                        'checkpoint_sha256':r['checkpoint_sha256'],'predictions_sha256':sha256(destination),
                        'checkpoint_kind':r.get('checkpoint_kind','historical_checkpoint'),
                        'inference_protocol':r.get('inference_protocol','test_split')})
        (output/'run.json').write_text(json.dumps({'experiment':config['experiment_id'],
            'input_checks':checked, 'records':records},indent=2)+'\n')
        print(f"Inference {i}/{len(jobs)} · {r['dataset']} · {r['model']} · {r['method']} · {r['seed']}",flush=True)
        del model
    return output

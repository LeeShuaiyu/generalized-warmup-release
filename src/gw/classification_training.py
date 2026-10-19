"""Train one fixed classification configuration into a new output directory."""
from pathlib import Path
import csv
import datetime
import json
import platform
import shutil
import numpy as np

from .assets import file_hash, relative_path
from .classification_data import load_manifest, manifest_sha256, ManifestDataset, balanced_sample_weights
from .classification_models import build_model
from .data_assets import prepare
from .metrics import classification_metrics
from .training_stages import (collect_logits, estimate_image_stats, set_warmup_trainability,
                              supervised_epoch, warmup_epoch)
from .warmup import parse_method


def resolve_training_run(root, experiment, run):
    """Use verified original settings or explicitly identified replay settings."""
    if 'training_settings' in run:
        return run
    if experiment != 'pretrained':
        raise ValueError('No verified training settings for this run')
    root = Path(root)
    names = ['pretrained_training.json']
    if run.get('checkpoint_kind') == 'retrained_checkpoint':
        names.insert(0, 'pretrained_replay_training.json')
    for name in names:
        config = json.loads((root/'configs'/name).read_text())
        matches = [r for r in config['runs'] if r['id'] == run['id']]
        if len(matches) > 1:
            raise ValueError('Duplicate verified training bindings')
        if matches:
            return matches[0]
    raise ValueError('No verified training entry for the selected checkpoint')


def train(root, experiment, run, *, device='auto', evaluate_test=False, limits=None):
    import torch
    root = Path(root).resolve()
    if 'training_settings' not in run:
        raise ValueError('This run requires a different training entry point')
    c = dict(run['training_settings'])
    mmce_alpha = float(c.get('mmce_alpha', 0.0))
    if not np.isfinite(mmce_alpha) or mmce_alpha < 0:
        raise ValueError('Invalid MMCE strength')
    if mmce_alpha and (c['label_smoothing'] or c['geometric_augmentation']):
        raise ValueError('MMCE protocol requires no augmentation or label smoothing')
    if c['validation_calibration_split']:
        raise ValueError('Validation/calibration partition entry point required')
    if c['checkpoint_metric'] not in {'val_nll', 'val_accuracy'}:
        raise ValueError('Unknown checkpoint selection metric')
    if c['epochs'] < 1 or c['batch_size'] < 1 or c['warmup_epochs'] < 0:
        raise ValueError('Invalid epoch or batch count')
    spec = parse_method(c['method'])
    if not 0 <= spec.probability <= 1 or spec.shape <= 0 or spec.rho < 0:
        raise ValueError('Invalid warm-up parameters')
    legacy_protocol = c.get('sampling_protocol') == 'sar_imagenet_v1'
    if c.get('sampling_protocol') not in {None, 'sar_imagenet_v1'}:
        raise ValueError('Unknown sampling protocol')
    if legacy_protocol and c['pretrained'] != 'imagenet':
        raise ValueError('The archived ImageNet protocol requires ImageNet initialization')
    dataset_key = run['dataset_key']
    catalog = json.loads((root / 'data/files.json').read_text())
    manifest = root / relative_path(catalog[dataset_key]['manifest'])
    if manifest_sha256(manifest) != run['split_sha256']:
        raise ValueError('Data split changed')
    rows = load_manifest(manifest)
    splits = ('train', 'val', 'test') if evaluate_test else ('train', 'val')
    # Verify only requested splits; validation-only training never opens test images.
    for split in splits:
        checked = prepare(root, dataset_key, check=True, split=split)
        if checked['missing'] or checked['invalid']:
            raise ValueError(f'Prepare {dataset_key}/{split} before training: {checked}')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device not in {'cuda', 'cpu'} or device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested device unavailable')
    if shutil.disk_usage(root).free < 8*1024**3:
        raise RuntimeError('Keep 8 GiB available before starting training')
    seed = c['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if legacy_protocol:
        from .imagenet_warmup import configure_determinism
        configure_determinism(torch)
    dataset_type = ManifestDataset
    if legacy_protocol:
        from .imagenet_data import ImageDataset
        dataset_type = ImageDataset
    datasets = {s:dataset_type(rows, s, image_size=c['image_size'],
                               data_root=root/'data/processed'/dataset_key,
                               limit=(limits or {}).get(s), seed=seed) for s in splits}
    if any(not len(x) for x in datasets.values()):
        raise ValueError('Empty data split')
    classes = 1+max(int(r['class_id']) for r in rows)
    if classes != run['classes']:
        raise ValueError('Class count changed')
    loaders = {}
    for split, dataset in datasets.items():
        if legacy_protocol:
            from .imagenet_warmup import make_loader
            loaders[split] = make_loader(dataset, split, c['batch_size'], seed, torch,
                balanced=c['class_balanced_sampler'], num_workers=c['num_workers'], pin_memory=device=='cuda')
            continue
        generator = torch.Generator().manual_seed(seed+{'train':201, 'val':301, 'test':401}[split])
        sampler = None
        if split == 'train' and c['class_balanced_sampler']:
            sampler = torch.utils.data.WeightedRandomSampler(balanced_sample_weights(dataset.rows), len(dataset),
                                                           replacement=True, generator=generator)
        loaders[split] = torch.utils.data.DataLoader(dataset, batch_size=c['batch_size'],
            shuffle=split=='train' and sampler is None, sampler=sampler, num_workers=c['num_workers'],
            persistent_workers=c['num_workers']>0, pin_memory=device=='cuda', generator=generator)
    first = datasets['train'][0][0]
    stats_options = {'generator': torch.Generator().manual_seed(seed + 101)} if legacy_protocol else {}
    stats_loader = torch.utils.data.DataLoader(datasets['train'], batch_size=c['batch_size'], shuffle=False,
        num_workers=c['num_workers'], persistent_workers=c['num_workers']>0, **stats_options)
    mean, std = estimate_image_stats(stats_loader, device, torch)
    model = build_model(c['architecture'], classes, c['pretrained'], c['image_size']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    output = root/'output'/f'train_{experiment}'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    state = {'experiment':experiment, 'source_run':run['id'], 'seed':seed, 'settings':c,
             'device':device, 'split_sha256':run['split_sha256'], 'test_evaluated':evaluate_test,
             'limits':limits, 'status':'running', 'split_counts':{s:len(d) for s,d in datasets.items()},
             'training_image_mean':mean, 'training_image_std':std,
             'code_sha256':{name:file_hash(Path(__file__).parent/name) for name in
                            ['classification_training.py', 'training_stages.py', 'warmup.py',
                             'classification_data.py', 'classification_models.py', 'metrics.py']},
             'environment':{'python':platform.python_version(), 'torch':torch.__version__,
                            'cuda':torch.version.cuda, 'cudnn':torch.backends.cudnn.version(),
                            'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None}}
    if 'training_record_kind' in run:
        state['training_record_kind'] = run['training_record_kind']
        state['source_code_sha256'] = run['source_code_sha256']
        state['replay_config_sha256'] = run['replay_config_sha256']
    (output/'run.json').write_text(json.dumps(state, indent=2)+'\n')
    if mmce_alpha:
        state['code_sha256']['mmce.py'] = file_hash(Path(__file__).parent/'mmce.py')
    if legacy_protocol:
        state['code_sha256']['imagenet_warmup.py'] = file_hash(Path(__file__).parent/'imagenet_warmup.py')
        state['code_sha256']['imagenet_data.py'] = file_hash(Path(__file__).parent/'imagenet_data.py')
    history = []
    try:
        if spec.family != 'none':
            if c['pretrained'] == 'imagenet':
                set_warmup_trainability(model, True)
                optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                              lr=c['lr'], weight_decay=c['weight_decay'])
            for epoch in range(c['warmup_epochs']):
                if legacy_protocol:
                    from .imagenet_warmup import epoch as imagenet_warmup_epoch
                    result = imagenet_warmup_epoch(model, spec, len(loaders['train']), c['batch_size'],
                        c['image_size'], classes, optimizer, device, torch, mean, std,
                        c['gaussian_std_multiplier'])
                else:
                    result = warmup_epoch(model, spec, len(loaders['train']), c['batch_size'], c['image_size'],
                                          classes, optimizer, device, torch, mean, std, first.shape[0])
                history.append(dict(stage='warmup', epoch=epoch+1, **result))
                print(f'Warm-up {epoch+1}/{c["warmup_epochs"]}', flush=True)
            set_warmup_trainability(model, False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
        best = best_state = best_epoch = None
        for epoch in range(c['epochs']):
            if mmce_alpha:
                from .mmce import epoch as mmce_epoch
                result = mmce_epoch(model, loaders['train'], optimizer, device, torch, alpha=mmce_alpha,
                                    geometric_augmentation=c['geometric_augmentation'], label_smoothing=c['label_smoothing'])
            else:
                selected_epoch = supervised_epoch
                if legacy_protocol:
                    from .imagenet_warmup import supervised_epoch as selected_epoch
                result = selected_epoch(model, loaders['train'], optimizer, device, torch,
                                          c['geometric_augmentation'], c['label_smoothing'])
            logits, labels = collect_logits(model, loaders['val'], device, torch)
            validation = classification_metrics(logits, labels)
            history.append(dict(stage='supervised', epoch=epoch+1, train_loss=result['loss'],
                                train_accuracy=result['accuracy'], **{f'val_{k}':v for k,v in validation.items()}))
            score = validation['nll'] if c['checkpoint_metric']=='val_nll' else -validation['accuracy']
            if not np.isfinite(score):
                raise ValueError('Non-finite validation score')
            if best is None or score < best:
                best, best_epoch = score, epoch+1
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(f'Training {epoch+1}/{c["epochs"]} · validation NLL {validation["nll"]:.6f}', flush=True)
        model.load_state_dict(best_state)
        outputs, metrics = {}, {}
        for split in splits:
            loader = loaders[split]
            if split == 'train':
                loader = torch.utils.data.DataLoader(datasets['train'], batch_size=c['batch_size'], shuffle=False,
                    num_workers=c['num_workers'], persistent_workers=c['num_workers']>0, pin_memory=device=='cuda')
            logits, labels = collect_logits(model, loader, device, torch)
            prefix = 'train_eval' if split=='train' else split
            outputs[f'{prefix}_logits'], outputs[f'{prefix}_labels'] = logits, labels
            metrics[prefix] = classification_metrics(logits, labels)
        torch.save(best_state, output/'best_model.pt')
        np.savez_compressed(output/'predictions.npz', **outputs)
        (output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        (output/'selection.json').write_text(json.dumps({'split':'val', 'metric':c['checkpoint_metric'],
                                                        'best_epoch':best_epoch, 'best_score':best}, indent=2)+'\n')
        state.update(status='complete', checkpoint_sha256=file_hash(output/'best_model.pt'),
                     predictions_sha256=file_hash(output/'predictions.npz'))
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        with (output/'history.csv').open('w', newline='') as stream:
            fields = sorted({k for r in history for k in r})
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
        (output/'run.json').write_text(json.dumps(state, indent=2)+'\n')
    return output

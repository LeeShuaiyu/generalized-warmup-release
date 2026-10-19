"""Evaluate EuroSAT RGB checkpoints with five fixed target-noise realizations."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil

import numpy as np
from .assets import verify
from .presentation import sha256


def dataset(root, download=False):
    import torchvision
    root = Path(root)
    manifest = json.loads((root/'artifacts/manifest.json').read_text())['source_files']
    for name in ['image_files.json','split_manifest.json']:
        path = root/'data/eurosat'/name
        verify(path,manifest[path.relative_to(root).as_posix()])
    if download:
        if shutil.disk_usage(root).free < 8*1024**3+500_000_000:
            raise RuntimeError('Keep 8 GiB free before downloading EuroSAT')
        torchvision.datasets.EuroSAT(str(root/'data'),download=True)
    base = torchvision.datasets.EuroSAT(str(root/'data'),download=False)
    files = json.loads((root/'data/eurosat/image_files.json').read_text())
    identities = {r['path']:r for r in files}
    context = json.loads((root/'data/eurosat/split_manifest.json').read_text())
    if base.classes != context['class_names'] or len(base) != len(identities):
        raise ValueError('EuroSAT class/file coverage differs')
    digest = hashlib.sha256()
    for path,label in base.samples:
        relative = Path(path).relative_to(root/'data').as_posix()
        item = identities['data/'+relative]
        value = sha256(path)
        if value != item['sha256']:
            raise ValueError(f'EuroSAT image differs: {relative}')
        digest.update(relative.encode()); digest.update(str(int(label)).encode('ascii')); digest.update(value.encode('ascii'))
    if digest.hexdigest() != context['dataset_content_sha256']:
        raise ValueError('EuroSAT ordered dataset identity differs')
    context['base'] = base
    return context


def run(root, config, device='auto', run_ids=None, workers=2):
    import torch
    from . import noise_core as core
    root = Path(root)
    if device=='auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device not in ['cpu','cuda'] or (device=='cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested device unavailable')
    if not isinstance(workers,int) or not 0<=workers<=8:
        raise ValueError('Worker count must be 0–8')
    selected = [r for r in config['runs'] if run_ids is None or r['id'] in run_ids]
    if not selected or (run_ids is not None and set(run_ids)!={r['id'] for r in selected}):
        raise ValueError('Invalid EuroSAT run selection')
    manifest = json.loads((root/'artifacts/manifest.json').read_text())
    for record in selected:
        if sha256(root/record['checkpoint']) != record['checkpoint_sha256']:
            raise ValueError('EuroSAT checkpoint identity differs')
        verify(root/record['predictions'],manifest['source_files'][record['predictions']])
    if shutil.disk_usage(root).free < 8*1024**3+len(selected)*1_000_000:
        raise RuntimeError('Keep 8 GiB free during inference')
    context = dataset(root)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    directory = root/'output'/config['experiment_id']/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory.mkdir(parents=True,exist_ok=False)
    config_path = root/f'configs/{config["experiment_id"]}.json'
    state = dict(experiment=config['experiment_id'],operation='checkpoint_test_inference',status='running',
                 config_sha256=sha256(config_path),device=device,workers=workers,torch_version=torch.__version__,
                 cuda_version=torch.version.cuda,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                 conv_tf32=torch.backends.cudnn.allow_tf32,dataset_content_sha256=context['dataset_content_sha256'],
                 code_sha256={name:sha256(root/'src/gw'/name) for name in ['noise_core.py','noise_inference.py']},records=[])
    def save():
        (directory/'run.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        for number,record in enumerate(selected,1):
            settings = record['settings']
            model = core.build_model(10).to(device)
            model.load_state_dict(torch.load(root/record['checkpoint'],map_location='cpu',weights_only=True),strict=True)
            parts = []
            for realization in range(settings['test_realizations']):
                ds = core.EuroSATDistributionDataset(context['base'],context['test_indices'],settings['noise_domain'],
                    context['mean'],context['std'],settings['alpha'],settings['looks'],record['seed'],False,
                    noise_seed=settings['noise_seed'],realization=realization)
                loader = torch.utils.data.DataLoader(ds,batch_size=settings['batch_size'],shuffle=False,num_workers=workers)
                logits,labels,indices = core.collect_logits(model,loader,torch.device(device))
                if not np.array_equal(indices,np.asarray(context['test_indices'])):
                    raise ValueError('EuroSAT test order differs')
                parts.append((logits,labels,indices))
            logits,labels,indices = [np.concatenate([p[i] for p in parts]) for i in range(3)]
            if logits.shape!=(record['samples'],record['classes']) or not np.isfinite(logits).all():
                raise ValueError('Invalid EuroSAT model predictions')
            with np.load(root/record['predictions'],allow_pickle=False) as original:
                if not np.array_equal(labels,original[record['labels_key']]):
                    raise ValueError('EuroSAT prediction labels differ')
            target = directory/f'{record["id"]}.npz'
            np.savez_compressed(target,logits=logits,labels=labels,indices=indices)
            state['records'].append(dict(id=record['id'],seed=record['seed'],domain=settings['noise_domain'],
                method=settings['method'],samples=len(labels),realizations=settings['test_realizations'],
                checkpoint_sha256=record['checkpoint_sha256'],predictions_sha256=sha256(target)))
            save()
            print(f'EuroSAT {number}/{len(selected)}: {settings["noise_domain"]} · {settings["method"]}',flush=True)
            del model
        state['status']='complete'; save()
    except Exception as exc:
        state['status']='failed'; state['error']=str(exc); save(); raise
    return directory

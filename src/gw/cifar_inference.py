"""Evaluate fixed CIFAR-100 MLP checkpoints on the official test split."""
from pathlib import Path
import datetime
import json
import shutil

import numpy as np

from .assets import verify
from .presentation import sha256


def dataset(root, train=False, download=False):
    import torchvision
    root = Path(root)
    manifest = json.loads((root/'artifacts/manifest.json').read_text())['source_files']
    identity = root/'data/cifar/files.json'
    if sha256(identity) != manifest['data/cifar/files.json']['sha256']:
        raise ValueError('CIFAR dataset identity record changed')
    records = json.loads(identity.read_text())
    if download:
        if shutil.disk_usage(root).free < 8*1024**3+200_000_000:
            raise RuntimeError('Keep 8 GiB free before downloading CIFAR')
        torchvision.datasets.CIFAR100(str(root/'data/cifar'),download=True,train=train)
    for name in [('train' if train else 'test'),'meta']:
        verify(root/records[name]['path'],records[name])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((.5071,.4867,.4408),(.2675,.2565,.2761))])
    return torchvision.datasets.CIFAR100(str(root/'data/cifar'),train=train,download=False,transform=transform)


def run(root, config, device='auto', run_ids=None):
    import torch
    from .cifar_model import LinearNetwork
    root = Path(root)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device not in ['cpu','cuda'] or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested device unavailable')
    selected = [r for r in config['runs'] if run_ids is None or r['id'] in run_ids]
    if not selected or (run_ids is not None and set(run_ids) != {r['id'] for r in selected}):
        raise ValueError('Invalid CIFAR run selection')
    manifest = json.loads((root/'artifacts/manifest.json').read_text())
    for record in selected:
        if sha256(root/record['checkpoint']) != record['checkpoint_sha256']:
            raise ValueError('CIFAR checkpoint identity mismatch')
        if sha256(root/record['predictions']) != manifest['source_files'][record['predictions']]['sha256']:
            raise ValueError('CIFAR reference prediction identity mismatch')
    if shutil.disk_usage(root).free < 8*1024**3+len(selected)*4_100_000:
        raise RuntimeError('Keep 8 GiB free during inference')
    images = dataset(root)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    directory = root/'output'/config['experiment_id']/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory.mkdir(parents=True,exist_ok=False)
    config_path = root/f'configs/{config["experiment_id"]}.json'
    state = {'experiment':config['experiment_id'],'operation':'checkpoint_test_inference','status':'running',
             'config_sha256':sha256(config_path),'device':device,'torch_version':torch.__version__,
             'cuda_version':torch.version.cuda,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
             'conv_tf32':torch.backends.cudnn.allow_tf32,
             'code_sha256':{name:sha256(root/'src/gw'/name) for name in ['cifar_model.py','cifar_inference.py']},
             'data_files':json.loads((root/'data/cifar/files.json').read_text()),'records':[]}
    def save():
        (directory/'run.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        for i,record in enumerate(selected,1):
            payload = torch.load(root/record['checkpoint'],map_location='cpu',weights_only=True)
            if payload['metadata']['seed'] != record['seed'] or payload['method'] != record['settings']['method']:
                raise ValueError('CIFAR checkpoint seed/method mismatch')
            model = LinearNetwork().to(device)
            model.load_state_dict(payload['state_dict'],strict=True)
            model.eval()
            loader = torch.utils.data.DataLoader(images,batch_size=record['settings']['evaluation_batch_size'],shuffle=False,num_workers=0)
            logits,labels = [],[]
            with torch.inference_mode():
                for x,y in loader:
                    logits.append(model(x.to(device)).cpu().numpy())
                    labels.append(y.numpy())
            logits,labels = np.concatenate(logits),np.concatenate(labels)
            if logits.shape != (record['samples'],record['classes']) or not np.isfinite(logits).all():
                raise ValueError('Invalid CIFAR model predictions')
            with np.load(root/record['predictions'],allow_pickle=False) as original:
                if not np.array_equal(labels,original[record['labels_key']]):
                    raise ValueError('CIFAR test order differs')
            target = directory/f'{record["id"]}.npz'
            np.savez_compressed(target,logits=logits,labels=labels)
            state['records'].append({'id':record['id'],'seed':record['seed'],'samples':len(labels),
                                     'checkpoint_sha256':record['checkpoint_sha256'],'predictions_sha256':sha256(target)})
            save()
            print(f'Inference {i}/{len(selected)} · {record["method"]} · {record["seed"]}',flush=True)
            del model
        state['status'] = 'complete'
        save()
    except Exception as exc:
        state.update(status='failed',error=str(exc))
        save()
        raise
    return directory

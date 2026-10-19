"""Source-bound family and duration experiments, with validation-only selection."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import platform
import random
import shutil

import numpy as np

from .classification_data import ManifestDataset, load_manifest, manifest_sha256
from .classification_models import build_model
from .data_assets import prepare
from .assets import file_hash
from .metrics import classification_metrics
from .training_stages import supervised_epoch, collect_logits
from .warmup import energy_standardize, uniform_soft_loss


def seed_all(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generators(seed, device, torch):
    return tuple(torch.Generator(device=device).manual_seed(seed+offset)
                 for offset in (0, 100000, 200000))


def validate_settings(run):
    experiment = run['experiment']
    if experiment not in {'s4_family', 's7_length'}:
        raise ValueError('Unsupported ablation experiment')
    families = {'loggamma', 'normal_residual', 'uniform_residual', 'laplace_residual'}
    if run['family'] not in (families if experiment=='s4_family' else {'erm', 'loggamma'}):
        raise ValueError('Unsupported ablation family')
    for key in ['shape', 'p', 'rho', 'learning_rate', 'weight_decay']:
        if not math.isfinite(run[key]):raise ValueError(f'Nonfinite {key}')
    if run['shape'] <= 0 or not 0 <= run['p'] <= 1 or run['rho'] < 0:
        raise ValueError('Invalid distribution parameters')
    if run['learning_rate'] <= 0 or run['weight_decay'] < 0:
        raise ValueError('Invalid optimizer settings')
    for key in ['batch_size', 'image_size', 'num_classes', 'supervised_epochs', 'warmup_epochs', 'seed']:
        if not isinstance(run[key], int) or isinstance(run[key], bool) or run[key] < 0:
            raise ValueError(f'Invalid {key}')
    if min(run['batch_size'], run['image_size'], run['supervised_epochs']) < 1 or run['num_classes'] < 2:
        raise ValueError('Empty training budget')
    if (run['family']=='erm') != (run['warmup_epochs']==0):
        raise ValueError('ERM is the zero-warm-up condition')
    norm = run['normalization']
    if not math.isfinite(norm['mean']) or not math.isfinite(norm['std']) or norm['std'] <= 0:
        raise ValueError('Invalid normalization')
    if run['normalize_real_inputs'] != (experiment=='s7_length'):
        raise ValueError('Input normalization protocol changed')


def sample(run, dims, streams, device, torch):
    """Keep the three independent streams and each experiment's exact operations."""
    z = torch.randn(dims, device=device, generator=streams[0])
    family = run['family']
    if family=='erm':raise ValueError('ERM does not generate warm-up inputs')
    if run['experiment']=='s7_length':
        z = energy_standardize(z)
        mask = (torch.rand((dims[0],1,1,1), device=device, generator=streams[1]) < run['p']).to(z.dtype)
        g = torch._standard_gamma(torch.full(dims, float(run['shape']), device=device), generator=streams[2])
        k = torch.tensor(float(run['shape']), device=device)
        q = (g.clamp_min(1e-12).log()-torch.special.digamma(k))/torch.sqrt(torch.polygamma(1,k))
        return (z+mask*run['rho']*q)/torch.sqrt(1+mask*run['rho']**2)
    mask = torch.rand((dims[0],1,1,1), device=device, generator=streams[1]) < run['p']
    if family=='normal_residual':
        q = torch.randn(dims, device=device, generator=streams[2])
    elif family=='uniform_residual':
        q = math.sqrt(3)*(2*torch.rand(dims, device=device, generator=streams[2])-1)
    elif family=='laplace_residual':
        u = torch.rand(dims, device=device, generator=streams[2]).clamp(1e-7,1-1e-7)-.5
        q = -torch.sign(u)*torch.log1p(-2*u.abs())/math.sqrt(2)
    elif family=='loggamma':
        k = torch.tensor(float(run['shape']), device=device)
        g = (torch._standard_gamma(torch.full(dims, float(run['shape']), device=device), generator=streams[2])/k).clamp_min(1e-12)
        q = (g.log()-(torch.special.digamma(k)-k.log()))/torch.sqrt(torch.polygamma(1,k))
    else:
        raise ValueError('Unsupported residual')
    norm = run['normalization']
    return norm['mean']+norm['std']*(z+mask*run['rho']*q)


def state_hash(model):
    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class Inputs:
    def __init__(self, base, run):
        self.base, self.run = base, run
    def __len__(self):return len(self.base)
    def __getitem__(self, index):
        x, y = self.base[index]
        if self.run['normalize_real_inputs']:
            norm = self.run['normalization']
            x = (x-norm['mean'])/norm['std']
        return x, y


def train(root, run, *, device='auto', limits=None):
    """Run one fixed condition; test images are never constructed or opened."""
    validate_settings(run)
    if device not in {'auto', 'cpu', 'cuda'}:raise ValueError('Unknown device')
    root = Path(root).resolve()
    manifest = root/'data/manifests/opensarship2.csv'
    if manifest_sha256(manifest) != run['manifest_sha256']:raise ValueError('Fixed split changed')
    rows = [r for r in load_manifest(manifest) if r['split'] in {'train', 'val'}]
    for split in ['train', 'val']:
        checked = prepare(root, 'opensar', check=True, split=split)
        if checked['missing'] or checked['invalid']:raise ValueError(f'Prepare opensar/{split}: {checked}')
    if shutil.disk_usage(root).free < 8*1024**3:raise RuntimeError('Keep at least 8 GiB available')
    import torch
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    datasets = {s:Inputs(ManifestDataset(rows,s,image_size=run['image_size'],data_root=root/'data/processed/opensar',
                    limit=(limits or {}).get(s),seed=run['seed']),run) for s in ['train', 'val']}
    if any(not len(ds) for ds in datasets.values()):raise ValueError('Empty split')
    if limits is None and [len(datasets[s]) for s in ['train','val']] != [4396,954]:
        raise ValueError('Training/validation count changed')
    if 1+max(int(r['class_id']) for r in rows) != run['num_classes']:raise ValueError('Class count changed')
    seed_all(run['seed'],torch)
    model = build_model(run['architecture'],run['num_classes'],'none',run['image_size']).to(device)
    initial = state_hash(model)
    if initial != run['initial_state_sha256']:raise ValueError('Original initialization identity mismatch')
    loaders = {s:torch.utils.data.DataLoader(datasets[s],batch_size=run['batch_size'],shuffle=s=='train',num_workers=0,
        generator=torch.Generator().manual_seed(run['seed']+(201 if s=='train' else 301))) for s in datasets}
    estimate = sum(v.numel()*v.element_size() for v in model.state_dict().values())*3 + len(datasets['val'])*32
    if shutil.disk_usage(root).free < 8*1024**3+estimate:raise RuntimeError('Keep 8 GiB after training outputs')
    output = root/'output'/f"train_{run['experiment']}"/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True,exist_ok=False)
    metadata = dict(status='running',settings=run,device=device,initial_state_sha256=initial,test_constructed=False,
        limits=limits,split_counts={s:len(d) for s,d in datasets.items()},
        code_sha256={name:file_hash(Path(__file__).parent/name) for name in
            ['ablation_training.py','classification_data.py','classification_models.py','training_stages.py','warmup.py','metrics.py']},
        environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda))
    history=[]
    def save():
        (output/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
        (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    save()
    try:
        streams=generators(run['seed']+10001,device,torch)
        optimizer=torch.optim.AdamW(model.parameters(),lr=run['learning_rate'],weight_decay=run['weight_decay'])
        dims=(run['batch_size'],1,run['image_size'],run['image_size'])
        for epoch in range(1,run['warmup_epochs']+1):
            model.train();losses=[]
            for _ in range(len(loaders['train'])):
                x=sample(run,dims,streams,device,torch)
                optimizer.zero_grad(set_to_none=True)
                loss=uniform_soft_loss(model(x),torch);loss.backward();optimizer.step()
                losses.append(float(loss.detach()))
            history.append(dict(stage='warmup',epoch=epoch,loss=float(np.mean(losses))));save()
            print(f"Warm-up {epoch}/{run['warmup_epochs']}",flush=True)
        seed_all(run['seed']+30001,torch)
        optimizer=torch.optim.AdamW(model.parameters(),lr=run['learning_rate'],weight_decay=run['weight_decay'])
        best=math.inf;best_state=None;best_epoch=None
        for epoch in range(1,run['supervised_epochs']+1):
            result=supervised_epoch(model,loaders['train'],optimizer,device,torch,False,0.)
            logits,labels=collect_logits(model,loaders['val'],device,torch)
            validation=classification_metrics(logits,labels)
            history.append(dict(stage='supervised',epoch=epoch,train=result,val=validation))
            if not math.isfinite(validation['nll']):raise ValueError('Nonfinite validation NLL')
            if validation['nll'] < best:
                best=validation['nll'];best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            save();print(f"Train {epoch}/{run['supervised_epochs']}",flush=True)
        model.load_state_dict(best_state)
        logits,labels=collect_logits(model,loaders['val'],device,torch)
        torch.save(best_state,output/'best_model.pt')
        np.savez_compressed(output/'validation_predictions.npz',logits=logits,labels=labels)
        metadata.update(status='complete',best_epoch=best_epoch,validation=classification_metrics(logits,labels),
            checkpoint_sha256=file_hash(output/'best_model.pt'),predictions_sha256=file_hash(output/'validation_predictions.npz'))
        save()
    except Exception as exc:
        metadata.update(status='failed',error=str(exc));save();raise
    return output

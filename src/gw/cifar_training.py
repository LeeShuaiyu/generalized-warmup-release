"""Single-run CIFAR training with the original fixed split and BP protocol."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import random
import shutil

import numpy as np

from .presentation import sha256


def state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape,dtype=np.int64).tobytes()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def split_indices(targets, settings):
    targets = np.asarray(targets)
    if targets.shape != (50000,):
        raise ValueError('Official CIFAR training split required')
    order = np.random.default_rng(settings['split_seed']).permutation(50000)
    train_pool,validation = order[:45000],order[45000:]
    selected = []
    rng = np.random.default_rng(settings['subset_seed'])
    for label in range(100):
        members = train_pool[targets[train_pool] == label]
        selected.extend(rng.choice(members,settings['train_samples_per_class'],replace=False).tolist())
    return np.asarray(selected),validation,hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest()


def run(root, record, device='auto', epochs=None, warmup_steps=None, train_limit=None, validation_limit=None):
    # Must be set before initializing CUDA when strict deterministic matmul is requested.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from .cifar_model import LinearNetwork
    from .cifar_inference import dataset
    from .metrics import classification_metrics
    root = Path(root)
    settings,seed = record['settings'],record['seed']
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    if device not in ['cuda','cpu'] or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested device unavailable')
    epoch_count = settings['epochs'] if epochs is None else epochs
    step_count = settings['warmup_steps'] if warmup_steps is None else warmup_steps
    if epoch_count < 1 or step_count < 0 or (settings['gaussian_scale'] is None and step_count):
        raise ValueError('Invalid training duration')
    if any(v is not None and v < 2 for v in [train_limit,validation_limit]):
        raise ValueError('Diagnostic subsets need at least two examples')
    if shutil.disk_usage(root).free < 8*1024**3+15_000_000:
        raise RuntimeError('Keep 8 GiB free during training')
    images = dataset(root,train=True)
    selected,validation,split_sha = split_indices(images.targets,settings)
    if split_sha != record['split_sha256']:
        raise ValueError('Training/validation split identity changed')
    full_train,full_validation = len(selected),len(validation)
    if train_limit is not None:
        selected = selected[:train_limit]
    if validation_limit is not None:
        validation = validation[:validation_limit]
    if len(selected)%settings['batch_size'] == 1:
        raise ValueError('Final training batch of one is incompatible with BatchNorm')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    def set_seed(value):
        random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)
    set_seed(seed)
    model = LinearNetwork()
    initial_sha = state_hash(model.state_dict())
    if initial_sha != record['initial_state_sha256']:
        raise ValueError('Initial state differs from the original protocol')
    model = model.to(device)
    directory = root/'output/cifar_training'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory.mkdir(parents=True,exist_ok=False)
    state = {'operation':'training','status':'running','seed':seed,'method':settings['method'],
             'device':device,'torch_version':torch.__version__,'settings':settings,
             'split_sha256':split_sha,'initial_state_sha256':initial_sha,'test_evaluated':False,
             'full_train_samples':full_train,'full_validation_samples':full_validation,
             'train_samples':len(selected),'validation_samples':len(validation),
             'execution':{'epochs':epoch_count,'warmup_steps':step_count,'train_limit':train_limit,'validation_limit':validation_limit},
             'config_sha256':sha256(root/'configs/cifar_training.json'),
             'data_files':json.loads((root/'data/cifar/files.json').read_text()),
             'training_indices_sha256':hashlib.sha256(selected.astype(np.int64).tobytes()).hexdigest(),
             'validation_indices_sha256':hashlib.sha256(validation.astype(np.int64).tobytes()).hexdigest(),
             'code_sha256':{name:sha256(root/'src/gw'/name) for name in ['cifar_model.py','cifar_inference.py','cifar_training.py']},
             'warmup':[],'supervised':[]}
    def save():
        (directory/'run.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        if step_count:
            set_seed(seed+100000)
            optimizer = torch.optim.Adam(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
            model.train()
            losses = []
            for step in range(step_count):
                x = settings['gaussian_scale']*torch.randn(settings['batch_size'],3,32,32,device=device)
                y = torch.randint(0,100,(settings['batch_size'],),device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(x),y)
                loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
                if (step+1)%390 == 0 or step+1 == step_count:
                    state['warmup'].append({'step':step+1,'loss':float(np.mean(losses))}); losses = []; save()
        # Reset Adam moments at the transition to supervised training.
        optimizer = torch.optim.Adam(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
        val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(images,validation.tolist()),batch_size=256,shuffle=False,num_workers=0)
        for epoch in range(1,epoch_count+1):
            order = np.random.default_rng(seed+400000+epoch).permutation(selected)
            set_seed(seed+200000+epoch)
            loader = torch.utils.data.DataLoader(torch.utils.data.Subset(images,order.tolist()),batch_size=settings['batch_size'],shuffle=False,num_workers=0)
            model.train()
            for x,y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(x.to(device)),y.to(device))
                loss.backward(); optimizer.step()
            model.eval(); logits,labels = [],[]
            with torch.inference_mode():
                for x,y in val_loader:
                    logits.append(model(x.to(device)).cpu().numpy()); labels.append(y.numpy())
            logits,labels = np.concatenate(logits),np.concatenate(labels)
            metrics = classification_metrics(logits,labels,15)
            state['supervised'].append({'epoch':epoch,'validation':metrics}); save()
            print(f'Training epoch {epoch}/{epoch_count} · validation ECE {metrics["ece_15"]:.6f}',flush=True)
        target = directory/'final_model.pt'
        torch.save({'state_dict':model.cpu().state_dict(),'metadata':{'seed':seed,'split_sha256':split_sha,'initial_state_sha256':initial_sha},
                    'method':settings['method'],'scale':settings['gaussian_scale']},target)
        predictions = directory/'validation_predictions.npz'
        np.savez_compressed(predictions,logits=logits,labels=labels)
        state.update(status='complete',checkpoint_sha256=sha256(target),validation_predictions_sha256=sha256(predictions))
        save()
    except Exception as exc:
        state.update(status='failed',error=str(exc)); save(); raise
    return directory

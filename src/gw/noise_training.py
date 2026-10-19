"""Train one EuroSAT noise-control configuration; checkpoint choice uses validation NLL."""
from pathlib import Path
from types import SimpleNamespace
import datetime
import json
import shutil

import numpy as np
from .noise_inference import dataset
from .presentation import sha256


def run(root, record, device='auto', *, epochs=None, warmup_steps=None, train_limit=None,
        validation_limit=None, workers=4):
    import torch
    from . import noise_core as core
    from .metrics import classification_metrics
    root=Path(root)
    if device=='auto':
        device='cuda' if torch.cuda.is_available() else 'cpu'
    if device not in ['cpu','cuda'] or (device=='cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested device unavailable')
    if shutil.disk_usage(root).free < 8*1024**3+60_000_000:
        raise RuntimeError('Keep 8 GiB free during training')
    c=record['settings']
    for v in [epochs,train_limit,validation_limit]:
        if v is not None and (not isinstance(v,int) or v<1):
            raise ValueError('Training limits must be positive integers')
    if warmup_steps is not None and (not isinstance(warmup_steps,int) or warmup_steps<0):
        raise ValueError('Warmup steps must be nonnegative')
    if not isinstance(workers,int) or not 0<=workers<=8:
        raise ValueError('Worker count must be 0–8')
    context=dataset(root)
    full_sizes={s:len(context[s+'_indices']) for s in ['train','val','test']}
    for name,limit in [('train',train_limit),('val',validation_limit)]:
        if limit is not None:
            if limit>full_sizes[name]:
                raise ValueError('Sample limit exceeds fixed split')
            context[name+'_indices']=context[name+'_indices'][:limit]
    # Creating datasets/loaders below does not instantiate or evaluate the test split.
    torch.set_num_threads(4)
    core.seed_all(record['seed'],True)
    initial=core.build_model(10).state_dict()
    initial_sha=core.state_digest(initial)
    if initial_sha!=record['initial_state_sha256']:
        raise ValueError('Initial model identity differs from the configured seed')
    core.seed_all(record['seed'],True)
    args=SimpleNamespace(alpha=c['alpha'],target_looks=c['looks'],warmup_looks=c['looks'],
        eval_noise_seed=c['noise_seed'],batch_size=c['batch_size'],num_workers=workers,
        lr=c['lr'],momentum=c['momentum'],weight_decay=c['weight_decay'])
    train_loader,val_loader=core.build_loaders(args,context,c['noise_domain'],record['seed'])
    model=core.build_model(10).to(device);model.load_state_dict(initial,strict=True)
    optimizer=core.build_sgd_optimizer(model,args)
    generators=dict(base=torch.Generator(device=device).manual_seed(record['seed']),
                    structure=torch.Generator(device=device).manual_seed(core.stable_seed('warmup_structure',record['seed'])))
    real_epochs=c['epochs'] if epochs is None else epochs
    steps=c['warmup_epochs']*len(train_loader) if warmup_steps is None else warmup_steps
    if c['method']=='erm':
        steps=0
    directory=root/'output/noise_training'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',operation='single_configuration_training',device=device,
        torch_version=torch.__version__,cuda_version=torch.version.cuda,test_evaluated=False,record=record,
        initial_state_sha256=initial_sha,dataset_content_sha256=context['dataset_content_sha256'],
        full_split_sizes=full_sizes,execution=dict(epochs=real_epochs,warmup_steps=steps,
            train_samples=len(context['train_indices']),validation_samples=len(context['val_indices']),workers=workers),
        code_sha256={name:sha256(root/'src/gw'/name) for name in ['noise_training.py','noise_core.py','noise_inference.py']},
        history=[])
    def save():
        (directory/'run.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        if steps:
            state['warmup_loss']=core.warmup_epoch(model,c['method'],steps,args,10,optimizer,generators,torch.device(device))
            save()
        # Preserve the original SGD momentum across the warmup/supervised boundary.
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=real_epochs)
        best_state=None;best_nll=float('inf');best_epoch=None
        for epoch in range(1,real_epochs+1):
            loss,accuracy=core.train_epoch(model,train_loader,optimizer,torch.device(device),epoch)
            logits,labels,indices=core.collect_logits(model,val_loader,torch.device(device))
            metrics=classification_metrics(logits,labels,n_bins=15)
            if metrics['nll']<best_nll:
                best_nll=metrics['nll'];best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            scheduler.step()
            state['history'].append(dict(epoch=epoch,train_loss=loss,train_accuracy=accuracy,validation=metrics))
            save()
            print(f'Epoch {epoch}/{real_epochs}: validation NLL {metrics["nll"]:.4f}',flush=True)
        model.load_state_dict(best_state,strict=True)
        checkpoint=directory/'checkpoint.pt';torch.save(best_state,checkpoint)
        logits,labels,indices=core.collect_logits(model,val_loader,torch.device(device))
        output=directory/'validation.npz';np.savez_compressed(output,logits=logits,labels=labels,indices=indices)
        state.update(status='complete',best_epoch=best_epoch,best_validation_nll=best_nll,
            checkpoint_sha256=sha256(checkpoint),validation_sha256=sha256(output))
        save()
    except Exception as exc:
        state.update(status='failed',error=str(exc));save();raise
    return directory

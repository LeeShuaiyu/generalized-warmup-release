"""Optional separate prediction environment for the ImageNet notebook."""
from pathlib import Path
import os
import subprocess
import sys


def run(root, config, device='auto', run_ids=None):
    root = Path(root).resolve()
    executable = os.environ.get('GW_INFERENCE_PYTHON')
    if not executable or Path(executable).resolve() == Path(sys.executable).resolve():
        from .classification_inference import run as evaluate
        return evaluate(root, config, device=device, run_ids=run_ids)
    if config['experiment_id'] != 'pretrained':
        raise ValueError('Separate prediction environment supports pretrained only')
    if device not in {'auto','cpu','cuda'}:
        raise ValueError('Unknown device')
    ids = [r['id'] for r in config['runs']]
    requested = ids if run_ids is None else list(run_ids)
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(ids):
        raise ValueError('Invalid selected run identities')
    command = [executable,str(root/'scripts/infer_pretrained.py'),'--no-report',
               '--device',device,'--run-ids',','.join(requested)]
    environment = dict(os.environ)
    environment.pop('GW_INFERENCE_PYTHON',None)
    process = subprocess.Popen(command,cwd=root,env=environment,stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,text=True)
    last = ''
    for line in process.stdout:
        print(line,end='',flush=True)
        if line.strip():
            last = line.strip()
    if process.wait():
        raise RuntimeError('Checkpoint inference failed; inspect the output above')
    destination = Path(last)
    if not destination.is_absolute() or not destination.resolve().is_relative_to(root/'output/pretrained'):
        raise ValueError('Unexpected inference destination')
    if not (destination/'run.json').is_file():
        raise ValueError('Missing inference identity record')
    return destination

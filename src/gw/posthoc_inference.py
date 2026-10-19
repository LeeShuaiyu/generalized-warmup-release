"""Checkpoint → validation/test logits → calibration parameters → held-out results."""
from pathlib import Path
import json

import numpy as np

from . import classification_inference, posthoc
from .supporting_reports import checked
from .presentation import sha256


def prediction_source(root, directory, base, split):
    root,directory = Path(root),Path(directory)
    execution_path = directory/'run.json'
    if not execution_path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Prediction outputs must be inside the prepared project')
    relative_execution = execution_path.resolve().relative_to(root.resolve()).as_posix()
    if relative_execution.startswith('results/'):
        checked(root,relative_execution)
    state = json.loads(execution_path.read_text())
    matches = [r for r in state['records'] if r['id']==base['base_run_id']]
    if len(matches) != 1:
        raise ValueError('Missing or duplicated base-model prediction')
    record = matches[0]
    samples = 295 if split == 'validation' else 292
    if record['checkpoint_sha256'] != base['checkpoint_sha256'] or record['seed'] != base['seed'] or record['samples'] != samples:
        raise ValueError('Model, seed or split identity differs')
    path = directory/f"{base['base_run_id']}.npz"
    if sha256(path) != record['predictions_sha256']:
        raise ValueError('Prediction file differs from execution record')
    logits_key,labels_key = ('logits','labels') if split == 'validation' else ('test_logits','test_labels')
    z,y = posthoc.load_predictions(path,7,logits_key,labels_key)
    original = checked(root,base[split])
    _,reference_labels = posthoc.load_predictions(original,7)
    if not np.array_equal(y,reference_labels):
        raise ValueError('Evaluation label order differs')
    return {'path':path.resolve().relative_to(root.resolve()).as_posix(),'sha256':sha256(path),
            'checkpoint_sha256':base['checkpoint_sha256'],'logits_key':logits_key,'labels_key':labels_key,
            'source_execution':relative_execution,'source_execution_sha256':sha256(execution_path)}


def run(root, device='auto', *, validation_directory=None, test_directory=None):
    root = Path(root)
    config = json.loads(checked(root,'configs/posthoc_refit.json').read_text())
    validation = json.loads(checked(root,'configs/posthoc_validation.json').read_text())
    precision = config.get('inference_precision', 'fp32')
    validation = dict(validation, inference_precision=precision)
    if validation_directory is None:
        validation_directory = classification_inference.run(root,validation,device)
    if test_directory is None:
        random = json.loads((root/'configs/random.json').read_text())
        ids = {r['base_run_id'] for r in config['runs']}
        test = dict(random,experiment_id='posthoc_test',inference_precision=precision,
                    runs=[r for r in random['runs'] if r['id'] in ids])
        test_directory = classification_inference.run(root,test,device)
    sources = {r['id']:{split:prediction_source(root,directory,r,split)
                        for split,directory in [('validation',validation_directory),('test',test_directory)]}
               for r in config['runs']}
    return posthoc.run(root,prediction_sources=sources)


def refit(root, directory):
    """Refit the exact inputs recorded with the displayed model-derived results."""
    root, directory = Path(root).resolve(), Path(directory).resolve()
    path = directory/'run.json'
    if not path.is_relative_to(root):
        raise ValueError('Calibration results must be inside the project')
    relative = path.relative_to(root).as_posix()
    if relative.startswith('results/'):
        checked(root, relative)
    state = json.loads(path.read_text())
    if state['operation'] != 'calibration_refit_from_model_logits':
        raise ValueError('Model-derived calibration results required')
    config = json.loads(checked(root, 'configs/posthoc_refit.json').read_text())
    sources = {}
    for record in config['runs']:
        sources[record['id']] = {}
        for split in ['validation', 'test']:
            matches = [item for item in state['inputs'] if item['split'] == split
                       and item.get('checkpoint_sha256') == record['checkpoint_sha256']]
            if len(matches) != 1:
                raise ValueError('One input per model and split required')
            item = matches[0]
            key = ('logits', 'labels') if split == 'validation' else ('test_logits', 'test_labels')
            sources[record['id']][split] = dict(item, logits_key=item.get('logits_key', key[0]),
                                               labels_key=item.get('labels_key', key[1]))
    return posthoc.run(root, prediction_sources=sources)

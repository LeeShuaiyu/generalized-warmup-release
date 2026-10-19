"""Validation-only temperature/vector fitting and held-out evaluation."""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import classification_metrics
from .presentation import sha256, COLUMNS, METRIC_KEYS
from .supporting_reports import checked
from .classification_report import INDEX


def validated(logits, labels):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    if z.ndim != 2 or not len(z) or z.shape[1] < 2 or not np.isfinite(z).all():
        raise ValueError('Finite nonempty classification logits required')
    if y.shape != (len(z),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError('One integer label per sample required')
    if np.any((y < 0) | (y >= z.shape[1])):
        raise ValueError('Label outside the class range')
    return z, y.astype(np.int64, copy=False)


def _nll(logits, labels):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    shifted = z-z.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(shifted).sum(axis=1))+z.max(axis=1)
    return float(np.mean(log_norm-z[np.arange(len(y)), y]))


@dataclass(frozen=True)
class TemperatureScaling:
    temperature: float
    validation_nll: float

    def transform(self, logits):
        return np.asarray(logits, dtype=np.float64)/self.temperature


@dataclass(frozen=True)
class VectorScaling:
    scale: np.ndarray
    bias: np.ndarray
    validation_nll: float
    l2: float

    def transform(self, logits):
        return np.asarray(logits, dtype=np.float64)*self.scale+self.bias


def fit_temperature_scaling(logits, labels):
    z, y = validated(logits, labels)
    result = minimize(lambda x: _nll(z/np.exp(x[0]), y), np.zeros(1),
                      method='L-BFGS-B', bounds=[(-5.0, 5.0)])
    if not result.success:
        raise RuntimeError(f'Temperature fitting failed: {result.message}')
    temperature = float(np.exp(result.x[0]))
    return TemperatureScaling(temperature, _nll(z/temperature, y))


def fit_vector_scaling(logits, labels, l2=1e-3):
    z, y = validated(logits, labels)
    classes = z.shape[1]
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError('Finite nonnegative L2 required')

    def unpack(x):
        scale, bias = x[:classes], x[classes:]
        return scale, bias-bias.mean()

    def objective(x):
        scale, bias = unpack(x)
        return _nll(z*scale+bias, y)+l2*float(np.mean((scale-1.0)**2)+np.mean(bias**2))

    result = minimize(objective, np.r_[np.ones(classes), np.zeros(classes)],
                      method='L-BFGS-B', options={'maxiter':500, 'ftol':1e-12})
    if not result.success:
        raise RuntimeError(f'Vector fitting failed: {result.message}')
    scale, bias = unpack(result.x)
    return VectorScaling(scale, bias, _nll(z*scale+bias, y), l2)


def load_predictions(path, classes, logits_key='logits', labels_key='labels'):
    with np.load(path, allow_pickle=False) as saved:
        z, y = validated(saved[logits_key], saved[labels_key])
    if z.shape[1] != classes:
        raise ValueError('Unexpected class count')
    return z, y


def run(root, prediction_sources=None):
    root = Path(root)
    config_path = checked(root, 'configs/posthoc_refit.json')
    config = json.loads(config_path.read_text())
    if config['fitting_split'] != 'validation' or config['evaluation_split'] != 'test':
        raise ValueError('Fit on validation; evaluate on test')
    identities = [r['id'] for r in config['runs']]
    if len(identities) != 6 or len(set(identities)) != 6:
        raise ValueError('Six unique base runs required')
    if prediction_sources is not None and set(prediction_sources) != set(identities):
        raise ValueError('Six complete model-prediction sources required')
    directory = root/'output/posthoc'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory.mkdir(parents=True, exist_ok=False)
    execution = {'status':'running', 'operation':'calibration_refit_from_saved_logits' if prediction_sources is None else 'calibration_refit_from_model_logits',
                 'config_sha256':sha256(config_path), 'code_sha256':sha256(Path(__file__)),
                 'records':[], 'fitted_parameters':{}, 'inputs':[]}
    run_path = directory/'run.json'

    def save():
        run_path.write_text(json.dumps(execution, indent=2, allow_nan=False)+'\n')

    def inputs(record, split):
        if prediction_sources is None:
            relative = record[split]
            path = checked(root,relative)
            z,y = load_predictions(path,config['classes'])
            identity = {'split':split,'path':relative,'sha256':sha256(path),'samples':len(y)}
        else:
            item = prediction_sources[record['id']][split]
            relative = item['path']
            path = root/relative
            if not path.resolve().is_relative_to(root.resolve()) or sha256(path) != item['sha256']:
                raise ValueError('Model-prediction source identity changed')
            if item['checkpoint_sha256'] != record['checkpoint_sha256']:
                raise ValueError('Calibration base model differs')
            z,y = load_predictions(path,config['classes'],item['logits_key'],item['labels_key'])
            identity = {'split':split,'path':relative,'sha256':sha256(path),'samples':len(y),
                        'logits_key':item['logits_key'], 'labels_key':item['labels_key'],
                        'checkpoint_sha256':item['checkpoint_sha256'],'source_execution':item['source_execution'],
                        'source_execution_sha256':item['source_execution_sha256']}
        execution['inputs'].append(identity)
        return z,y

    save()
    try:
        fits = {}
        # Finish parameter estimation before loading any test predictions.
        for record in config['runs']:
            z,y = inputs(record,'validation')
            temperature = fit_temperature_scaling(z, y)
            vector = fit_vector_scaling(z, y, config['vector_l2'])
            fits[record['id']] = (temperature, vector)
            execution['fitted_parameters'][record['id']] = {
                'temperature':temperature.temperature, 'temperature_validation_nll':temperature.validation_nll,
                'vector_scale':vector.scale.tolist(), 'vector_bias':vector.bias.tolist(),
                'vector_validation_nll':vector.validation_nll, 'vector_l2':vector.l2}
            save()
        # Parameters are locked. Test labels are used for metrics, never fitting.
        for record in config['runs']:
            z,y = inputs(record,'test')
            temperature, vector = fits[record['id']]
            base = 'GW' if record['arm'] == 'gw' else 'Standard training'
            outputs = [('raw', z)] if record['arm'] == 'gw' else []
            outputs += [('ts', temperature.transform(z)), ('vs', vector.transform(z))]
            for kind, logits in outputs:
                identity = f"{record['id']}_{kind}"
                predictions = directory/f'{identity}.npz'
                np.savez_compressed(predictions, logits=logits, labels=y)
                execution['records'].append({'id':identity, 'seed':record['seed'], 'dataset':'OCTDL',
                    'model':'ResNet-18', 'method':base if kind == 'raw' else base+' + '+kind.upper(),
                    'samples':len(y), 'predictions':predictions.name, 'predictions_sha256':sha256(predictions),
                    'metrics':classification_metrics(logits,y,config['ece_bins'])})
                save()
        execution['status'] = 'complete'
        save()
    except Exception as exc:
        execution['status'] = 'failed'
        execution['error'] = str(exc)
        save()
        raise
    return directory


def summarize(root, directory):
    root, directory = Path(root), Path(directory)
    execution_path = directory/'run.json'
    if execution_path.resolve().is_relative_to(root.resolve()):
        relative = execution_path.resolve().relative_to(root.resolve()).as_posix()
        if relative.startswith('results/'):
            checked(root, relative)
    execution = json.loads(execution_path.read_text())
    if execution['status'] != 'complete' or len(execution['records']) != 15:
        raise ValueError('Fifteen complete calibration outputs required')
    rows = []
    for record in execution['records']:
        path = directory/record['predictions']
        if path.parent.resolve() != directory.resolve() or sha256(path) != record['predictions_sha256']:
            raise ValueError('Calibration output identity changed')
        z, y = load_predictions(path, 7)
        metrics = classification_metrics(z, y, n_bins=15)
        rows.append(dict(zip(INDEX,[record['dataset'],record['model'],record['method']]),
                         Seed=record['seed'], Samples=len(y),
                         **{col:metrics[key]*(100 if i<3 else 1)
                            for i,(col,key) in enumerate(zip(COLUMNS,METRIC_KEYS))}))
    frame = pd.DataFrame(rows)
    if frame.duplicated(INDEX+['Seed']).any():
        raise ValueError('Duplicate calibration run')
    groups = frame.groupby(INDEX,sort=False)[COLUMNS]
    if not (groups.size() == 3).all():
        raise ValueError('Three seeds per calibration condition required')
    order = pd.MultiIndex.from_tuples([('OCTDL','ResNet-18',m) for m in
        ['GW','Standard training + TS','GW + TS','Standard training + VS','GW + VS']], names=INDEX)
    return frame, groups.mean().reindex(order), groups.std(ddof=1).reindex(order)

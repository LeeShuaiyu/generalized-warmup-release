"""MLE and held-out scoring of fixed, grouped homogeneous-region blocks."""
from pathlib import Path
import datetime
import json
import numpy as np
import pandas as pd
import scipy
from scipy import optimize, special, stats

from .assets import file_hash
from .supporting_reports import checked

FAMILIES = {'Gaussian':stats.norm, 'Gamma':stats.gamma, 'Weibull':stats.weibull_min,
            'Lognormal':stats.lognorm, 'Rayleigh':stats.rayleigh, 'Nakagami':stats.nakagami,
            'GeneralizedGamma':stats.gengamma, 'Uniform':stats.uniform, 'Laplace':stats.laplace}
BOUNDS = [(-7, 10), (-7, 7), (-15, 15)]


def ggfit(x, gp, wp):
    """Positive-power generalized Gamma; three bounded MLE starts."""
    lx = np.log(x)
    ml = lx.mean()
    def fun(t):
        a, c, s = np.exp(t)
        v = c*(lx-np.log(s))
        z = np.exp(np.clip(v, -700, 500))
        ez = z.mean()
        val = special.gammaln(a)+a*c*np.log(s)-np.log(c)-(a*c-1)*ml+ez
        grad = np.array([a*(special.digamma(a)+c*(np.log(s)-ml)),
                         a*c*(np.log(s)-ml)-1+np.mean(z*v), a*c-c*ez])
        return val, grad
    starts = [(gp[0], 1, gp[2]), (1, wp[0], wp[2]), (2, 2, .7)]
    fits = [optimize.minimize(fun, np.log(v), jac=True, method='L-BFGS-B',
                             bounds=BOUNDS, options={'maxiter':400}) for v in starts]
    good = [r for r in fits if r.success and np.isfinite(r.fun)]
    if not good:
        raise RuntimeError('Generalized Gamma convergence failed')
    result = min(good, key=lambda r:r.fun)
    a, c, s = np.exp(result.x)
    meta = {'success':True, 'starts':3, 'converged':len(good),
            'boundary':bool(any(abs(result.x[i]-b)<1e-5 for i,bb in enumerate(BOUNDS) for b in bb))}
    return (a, c, 0, s), meta


def grouped_samples(bank, records, seed):
    bank = np.asarray(bank, dtype=float)
    if bank.ndim != 3 or bank.shape[1:] != (16, 16) or len(bank) != len(records):
        raise ValueError('Expected one 16×16 block per record')
    if not np.isfinite(bank).all() or np.any(bank < 0):
        raise ValueError('Positive-distribution fitting requires finite nonnegative pixels')
    flat = bank.reshape(-1, 256)
    if np.any(flat.mean(1) <= 0):
        raise ValueError('Zero-mean block cannot be normalized')
    normalized = flat/flat.mean(1, keepdims=True)
    groups = np.asarray([r['group_id'] for r in records])
    unique = np.unique(groups)
    if len(unique) < 4:
        raise ValueError('At least four groups required')
    ordered = np.random.default_rng(seed).permutation(unique)
    fitting_groups = ordered[:max(2, int(.7*len(unique)))]
    mask = np.isin(groups, fitting_groups)
    train, hold = normalized[mask].ravel(), normalized[~mask].ravel()
    train, hold = train[train > 0], hold[hold > 0]
    if not len(train) or not len(hold):
        raise ValueError('No positive fit or evaluation pixels')
    meta = {'fit_blocks':int(mask.sum()), 'hold_blocks':int((~mask).sum()),
            'fit_groups':len(fitting_groups), 'hold_groups':len(unique)-len(fitting_groups),
            'fit_pixels':len(train), 'hold_pixels':len(hold),
            'zero_fraction':float((normalized==0).mean())}
    return train, hold, meta


def fit_samples(train, hold, families):
    if not families or any(name not in FAMILIES for name in families):
        raise ValueError('Unknown or empty distribution family selection')
    gamma = stats.gamma.fit(train, floc=0)
    weibull = stats.weibull_min.fit(train, floc=0)
    rows = []
    for name in families:
        distribution = FAMILIES[name]
        row = {'family':name}
        try:
            if name == 'Gamma':
                parameters = gamma
            elif name == 'Weibull':
                parameters = weibull
            elif name == 'GeneralizedGamma':
                parameters, row['optimizer'] = ggfit(train, gamma, weibull)
            else:
                parameters = distribution.fit(train, **({'floc':0} if name in {'Lognormal','Rayleigh','Nakagami'} else {}))
            log_density = distribution.logpdf(hold, *parameters)
            outside = ~np.isfinite(log_density)
            row.update(parameters=list(map(float, parameters)),
                       nll=float(-log_density.mean()) if not outside.any() else None,
                       ks=float(stats.kstest(hold, lambda v:distribution.cdf(v, *parameters)).statistic),
                       outside_fraction=float(outside.mean()),
                       status='OUTSIDE_SUPPORT' if outside.any() else 'OK')
        except Exception as error:
            row.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        rows.append(row)
    return rows


def rebuilt_input(root, record, directory):
    """Accept a completed, full region run only after reconciling its bank."""
    from .region_selection import compare_bank
    directory = Path(directory)
    state = json.loads((directory / 'run.json').read_text())
    if (state['status'] != 'complete' or state['dataset'] != record['id'] or
            state['diagnostic_limit'] is not None or not state['comparison']['matches_fixed_bank']):
        raise ValueError('Require a complete matching region-selection run')
    for name, digest in state['output_files'].items():
        if file_hash(directory / name) != digest:
            raise ValueError('Rebuilt region output changed')
    records = json.loads((directory / 'records.json').read_text())
    with np.load(directory / 'bank.npz', allow_pickle=False) as data:
        full_bank = {key: data[key].copy() for key in data.files}
    identity = {'expected_bank': record['bank'], 'expected_records': record['records']}
    if not compare_bank(root, identity, full_bank, records)['matches_fixed_bank']:
        raise ValueError('Rebuilt pixels or identities differ from the fixed bank')
    return {key: full_bank[key] for key in record['coordinates']}, records


def run(root, *, datasets=None, families=None, region_runs=None):
    root = Path(root)
    config = json.loads(checked(root, 'configs/fit.json').read_text())
    if config['fit_group_fraction'] != .7 or config['block_size'] != 16:
        raise ValueError('Unsupported fitting protocol')
    selected = [d for d in config['datasets'] if datasets is None or d['id'] in datasets]
    if not selected:
        raise ValueError('No selected datasets')
    region_runs = region_runs or {}
    if set(region_runs) - {d['id'] for d in selected}:
        raise ValueError('Region run supplied for an unselected dataset')
    families = list(FAMILIES) if families is None else list(families)
    inputs = []
    for record in selected:
        bank_path = checked(root, record['bank'])
        records_path = checked(root, record['records'])
        if record['id'] in region_runs:
            bank, records = rebuilt_input(root, record, region_runs[record['id']])
        else:
            with np.load(bank_path, allow_pickle=False) as z:
                bank = {key:z[key].copy() for key in record['coordinates']}
            records = json.loads(records_path.read_text())
        inputs.append((record, bank, records))
    output = root/'output'/'fit'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    state = {'status':'running', 'config_sha256':file_hash(root/'configs/fit.json'),
             'code_sha256':file_hash(Path(__file__)), 'families':families,
             'environment':{'numpy':np.__version__, 'scipy':scipy.__version__},
             'input_files':{r[key]:file_hash(root/r[key]) for r,_,_ in inputs for key in ['bank','records']},
             'scope':'Refit fixed selected blocks; region selection is not rerun'}
    state['rebuilt_region_runs'] = {key: {'path': str(Path(path)), 'run_sha256': file_hash(Path(path) / 'run.json'),
                                        'bank_sha256': file_hash(Path(path) / 'bank.npz')}
                                    for key, path in region_runs.items()}
    (output/'run.json').write_text(json.dumps(state, indent=2)+'\n')
    rows = []
    try:
        for record, bank, records in inputs:
            for coordinate in record['coordinates']:
                for seed in config['seeds']:
                    train, hold, split = grouped_samples(bank[coordinate], records, seed)
                    for fitted in fit_samples(train, hold, families):
                        rows.append(dict(dataset=record['source_id'], label=record['label'],
                                         coordinate=coordinate, seed=seed, **split, **fitted))
                    (output/'rows.json').write_text(json.dumps(rows, indent=2)+'\n')
                    print(f'Fit · {record["label"]} · {coordinate} · {seed}', flush=True)
        if any(row['status']=='FAILED' for row in rows):
            raise RuntimeError('One or more candidate distributions failed; inspect rows.json')
        state.update(status='complete', rows=len(rows), scores_sha256=file_hash(output/'rows.json'))
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (output/'run.json').write_text(json.dumps(state, indent=2)+'\n')
    return output


def summarize(directory):
    directory = Path(directory)
    state = json.loads((directory/'run.json').read_text())
    if state['status'] != 'complete' or file_hash(directory/'rows.json') != state['scores_sha256']:
        raise ValueError('Incomplete or changed refit scores')
    rows = pd.DataFrame(json.loads((directory/'rows.json').read_text()))
    if rows.duplicated(['dataset','coordinate','family','seed']).any():
        raise ValueError('Duplicate fit scores')
    rows['nll'] = rows['nll'].fillna(np.inf)
    summary = rows.groupby(['label','coordinate','family'])[['nll','ks']].mean()
    return rows, summary.rename(columns={'nll':'NLL ↓', 'ks':'KS ↓'})


def paper_comparison(root, directory):
    from .supporting_reports import fit
    root, directory = Path(root), Path(directory)
    if directory.resolve() == (root/'results/fit/refit').resolve():
        for name in ['run.json','rows.json','implementation.py']:
            checked(root, f'results/fit/refit/{name}')
    _, summary = summarize(directory)
    reference = fit(root)
    records = []
    for (dataset, family), saved in reference.iterrows():
        source_family = 'GeneralizedGamma' if family=='Generalized Gamma' else family
        coordinate = 'stored' if dataset=='FUSAR-Ship' and source_family=='GeneralizedGamma' else 'model_gray'
        fresh = summary.loc[(dataset, coordinate, source_family)]
        row = {'Dataset':dataset, 'Family':family, 'Coordinate':coordinate}
        for metric in ['NLL ↓','KS ↓']:
            row[metric] = fresh[metric]
            row[f'Δ {metric[:-2]}'] = 0.0 if np.isinf(fresh[metric]) and np.isinf(saved[metric]) else fresh[metric]-saved[metric]
        row['Matches'] = all(np.round(fresh[m],4)==np.round(saved[m],4) for m in ['NLL ↓','KS ↓'])
        records.append(row)
    return pd.DataFrame(records).set_index(['Dataset','Family'])


def input_settings(root):
    config = json.loads(checked(root,'configs/fit.json').read_text())
    rows = []
    for item in config['datasets']:
        records = json.loads(checked(root,item['records']).read_text())
        rows.append({'Dataset':item['label'],'Blocks':len(records),
                     'Groups':len({r['group_id'] for r in records}),
                     'Pixels':' / '.join({'model_gray':'Scaled','stored':'Stored'}[c] for c in item['coordinates'])})
    return pd.DataFrame(rows).set_index('Dataset')

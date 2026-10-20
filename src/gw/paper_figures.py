"""Historical component generators; use submitted_figures for current figure numbers."""
from pathlib import Path
import datetime
import json
import shutil
import base64
import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image

from . import classification_report as classification, distribution_fitting as fitting
from . import presentation, supporting_reports as supporting
from .metrics import softmax
from .supporting_reports import checked

BLUE, GREY, GOLD, LIGHT = '#226b91', '#73808a', '#b18a61', '#9faeb9'
REDRAW_SCOPE = {
    2: 'Six quantitative panels: saved confidence grid and paired measurements; archived validation curves; saved CUDA test predictions.',
    3: 'Response panels c,d only: paired directional responses and supervised trajectories. Geometry and density illustrations remain in the manuscript image.',
    4: 'ECE panel only: means and 95% t intervals over ten paired seeds. Illustrative imagery and density drawings remain in the manuscript image.',
    6: 'Fixed block 0, held-out relative-pixel histogram and first-split fits, and five-seed warm-up ECE.',
}


def component_catalog(root):
    config = json.loads(checked(root, 'configs/quantitative_components.json').read_text())
    if [r['number'] for r in config['figures']] != list(range(1, 7)):
        raise ValueError('Expected six manuscript figures in order')
    for record in config['figures']:
        for asset in record['assets']:
            if presentation.sha256(checked(root, asset['path'])) != asset['sha256']:
                raise ValueError('Manuscript figure identity differs')
    return config['figures']


def component_html(root, number, folded=False):
    """Display-only thumbnails; the registered manuscript rasters remain unchanged."""
    record = next(r for r in component_catalog(root) if r['number'] == number)
    width = 600 if len(record['assets']) > 1 else 1400
    images = []
    for asset in record['assets']:
        with Image.open(checked(root, asset['path'])) as original:
            preview = original.copy()
        preview.thumbnail((width, 1800), Image.Resampling.LANCZOS)
        stream = io.BytesIO()
        preview.save(stream, format='PNG', optimize=True)
        encoded = base64.b64encode(stream.getvalue()).decode('ascii')
        minimum = '230px' if len(record['assets']) > 1 else '0'
        images.append(f'<img src="data:image/png;base64,{encoded}" style="max-width:100%;height:auto;min-width:{minimum};flex:1" alt="Manuscript figure {number}">')
    content = '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:12px">' + ''.join(images) + '</div>'
    return '<details><summary>Manuscript image</summary>'+content+'</details>' if folded else content


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 12, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False, 'figure.dpi': 140})


def confidence_grid(root):
    frame = pd.read_csv(checked(root, 'artifacts/figure_inputs/heatmap.csv'))
    if frame.duplicated(['x1', 'x2']).any() or len(frame) != 241**2:
        raise ValueError('Incomplete confidence grid')
    return frame


def reliability(logits, labels, bins=15):
    probability = softmax(logits)
    confidence = probability.max(1)
    correct = probability.argmax(1) == labels
    rows = []
    edges = np.linspace(0, 1, bins+1)
    for i in range(bins):
        mask = (confidence >= edges[i]) & (confidence <= edges[i+1]) if i == 0 else (
            (confidence > edges[i]) & (confidence <= edges[i+1]))
        rows.append(dict(Bin=i, Low=edges[i], High=edges[i+1], Count=int(mask.sum()),
                         Confidence=confidence[mask].mean() if mask.any() else np.nan,
                         Accuracy=correct[mask].mean() if mask.any() else np.nan))
    result = pd.DataFrame(rows)
    if result.Count.sum() != len(labels):
        raise ValueError('Reliability bins do not partition the predictions')
    return result


def mechanism(root):
    grid = confidence_grid(root)
    paired, _ = supporting.toy(root)
    trajectory = pd.read_csv(checked(root, 'artifacts/figure_inputs/cifar_trajectory.csv'))
    config = classification.read(root, 'cifar3')
    classification.calculate(root, config, Path(root)/'results/cifar3/gpu_predictions')
    seeds = sorted({r['seed'] for r in config['runs']})
    if len(seeds) != 3 or sorted(trajectory.seed.unique()) != seeds or len(trajectory) != 180:
        raise ValueError('Expected three complete paired CIFAR validation trajectories')
    if trajectory.duplicated(['seed', 'method', 'epoch']).any():
        raise ValueError('Duplicated validation epoch')
    figure, axes = plt.subplots(2, 3, figsize=(10.8, 6.5), layout='constrained')
    for axis, column, title in zip(axes[0, :2], ['single_initialization_confidence',
            'same_initialization_after_warmup_confidence'], ['a  Initialization', 'b  After warm-up']):
        table = grid.pivot(index='x2', columns='x1', values=column)
        image = axis.imshow(table.to_numpy(), origin='lower', extent=(-10, 10, -10, 10),
                            vmin=.5, vmax=1., cmap='cividis', interpolation='nearest')
        axis.set(xlabel='$x_1$', ylabel='$x_2$')
        axis.set_title(title, loc='left')
    figure.colorbar(image, ax=list(axes[0, :2]), shrink=.8, label='Confidence')
    for axis, i, title in [(axes[0, 2], 0, 'c  Confidence bias'), (axes[1, 0], 1, 'd  Local response')]:
        values = paired.iloc[:, 2*i:2*i+2].to_numpy()
        values = values if i == 0 else np.exp(values)
        for row in values:
            axis.plot([0, 1], row, color=GREY, alpha=.28, linewidth=.7, marker='.', markersize=3)
        center = values.mean(0) if i == 0 else np.exp(paired.iloc[:, 2*i:2*i+2].mean(0))
        axis.plot([0, 1], center, 'o', mfc='white', mec='#253746', mew=1.4, markersize=7)
        axis.set_xticks([0, 1], ['Initialization', 'Warm-up'])
        axis.set_title(title, loc='left')
        axis.set_ylabel('Bias' if i == 0 else 'Response norm')
        if i:
            axis.set_yscale('log')
    curves = []
    for method, color, linestyle in [('ERM', GREY, '--'), ('Gaussian warm-up', BLUE, '-')]:
        selected = trajectory[trajectory.method == method]
        matrix = selected.pivot(index='epoch', columns='seed', values='ece_15') * 100
        if matrix.shape != (30, 3) or matrix.isna().any().any():
            raise ValueError('Incomplete validation curve')
        mean, low, high = matrix.mean(1), matrix.min(1), matrix.max(1)
        axes[1, 1].plot(mean.index, mean, color=color, linestyle=linestyle, label=method)
        axes[1, 1].fill_between(mean.index, low, high, color=color, alpha=.12)
        curves.extend(dict(Method=method, Epoch=epoch, Mean=mean[epoch], Low=low[epoch], High=high[epoch]) for epoch in mean.index)
    axes[1, 1].set(xlabel='Supervised epoch', ylabel='Validation ECE (%) ↓')
    axes[1, 1].set_title('e  CIFAR-100', loc='left')
    axes[1, 1].legend(fontsize=9)
    bins = []
    for method, color, marker in [('No warm-up', GREY, 's'), ('Gaussian warm-up', BLUE, 'o')]:
        logits, labels = [], []
        for record in config['runs']:
            if record['method'] == method:
                with np.load(checked(root, f'results/cifar3/gpu_predictions/{record["id"]}.npz'), allow_pickle=False) as z:
                    logits.append(z['logits']); labels.append(z['labels'])
        table = reliability(np.concatenate(logits), np.concatenate(labels))
        bins.append(table.assign(Method=method))
        observed = table[table.Count > 0]
        axes[1, 2].plot(observed.Confidence, observed.Accuracy, marker=marker, markersize=4,
                         color=color, label=method)
    axes[1, 2].plot([0, 1], [0, 1], '--', color=LIGHT, linewidth=.8)
    axes[1, 2].set(xlabel='Mean confidence', ylabel='Accuracy', xlim=(0, 1), ylim=(0, 1))
    axes[1, 2].set_title('f  Reliability', loc='left')
    axes[1, 2].legend(fontsize=9)
    return figure, {'confidence_grid': grid, 'paired_metrics': paired,
                    'validation_curves': pd.DataFrame(curves), 'reliability_bins': pd.concat(bins)}


def direction_statistics(root):
    frame = pd.read_csv(checked(root, 'artifacts/figure_inputs/direction_trajectory.csv'))
    frame = frame[(frame.architecture == 'linear') & (frame.coverage_v == .1)
                  & frame.method.isin(['targeted_v', 'wrong_w'])].copy()
    if frame.duplicated(['seed', 'method', 'phase', 'phase_step', 'direction']).any():
        raise ValueError('Duplicated directional response')
    if not np.isfinite(frame.R).all() or (frame.R < 0).any():
        raise ValueError('Invalid squared response')
    rows = []
    # Shared bootstrap seed indices preserve pairing across conditions and times.
    indices = np.random.default_rng(20261009).integers(0, 100, size=(10000, 100))
    selected = frame[((frame.phase == 'warmup') & (frame.phase_step == 500)) | (frame.phase == 'supervised')]
    for (phase, step, direction, method), group in selected.groupby(['phase', 'phase_step', 'direction', 'method']):
        group = group.sort_values('seed')
        if group.seed.tolist() != list(range(100)):
            raise ValueError('Expected one response for each of 100 paired seeds')
        logs = np.log(group.R.to_numpy()+1e-12)
        low, high = np.quantile(logs[indices].mean(1), [.025, .975])
        rows.append(dict(Phase=phase, Step=0 if phase == 'warmup' else step,
                         Direction=direction, Method=method, Mean=np.exp(logs.mean()),
                         Low=np.exp(low), High=np.exp(high), Seeds=100))
    return frame, pd.DataFrame(rows)


def directional(root):
    raw, summary = direction_statistics(root)
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), layout='constrained')
    labels = [('wrong_w', 'Targets w', GREY), ('targeted_v', 'Targets v', BLUE)]
    for axis, direction, title in zip(axes[:2], ['w_strong_nuisance', 'v_weak_nuisance'],
                                      ['c  Response along w', 'c  Response along v']):
        points = raw[(raw.phase == 'warmup') & (raw.phase_step == 500) & (raw.direction == direction)]
        paired = points.pivot(index='seed', columns='method', values='R').sort_index()
        for row in paired[['wrong_w', 'targeted_v']].to_numpy():
            axis.plot([0, 1], row+1e-12, color=LIGHT, linewidth=.6, alpha=.32)
        for x, (method, label, color) in enumerate(labels):
            row = summary[(summary.Phase == 'warmup') & (summary.Direction == direction) & (summary.Method == method)].iloc[0]
            axis.errorbar(x, row.Mean, yerr=[[row.Mean-row.Low], [row.High-row.Mean]],
                          fmt='o', mfc='white', mec=color, ecolor=color, capsize=4, markersize=7)
        axis.set_xticks([0, 1], [r[1] for r in labels])
        axis.set_yscale('log'); axis.set_ylabel(r'$R+\epsilon$'); axis.set_title(title, loc='left')
    for method, label, color in labels:
        rows = summary[(summary.Direction == 'v_weak_nuisance') & (summary.Method == method)].sort_values('Step')
        axes[2].plot(rows.Step, rows.Mean, color=color, label=label)
        axes[2].fill_between(rows.Step, rows.Low, rows.High, color=color, alpha=.13)
    axes[2].set(xlabel='Supervised step', ylabel=r'$R+\epsilon$', yscale='log')
    axes[2].set_title('d  Response along v', loc='left'); axes[2].legend(fontsize=9)
    return figure, {'paired_responses': raw, 'response_intervals': summary}


def noise_statistics(root):
    config = classification.read(root, 'noise')
    runs, _, _ = classification.calculate(root, config, Path(root)/'results/noise/gpu_predictions')
    grouped = runs.groupby(['Dataset', 'Method'])['ECE (%)']
    mean, sd, count = grouped.mean(), grouped.std(ddof=1), grouped.count()
    if not (count == 10).all():
        raise ValueError('Ten paired seeds required in each noise condition')
    margin = stats.t.ppf(.975, 9) * sd / np.sqrt(10)
    return runs, pd.DataFrame({'Mean': mean, 'Low': mean-margin, 'High': mean+margin, 'Seeds': count})


def optical_noise(root):
    raw, summary = noise_statistics(root)
    figure, axis = plt.subplots(figsize=(6.6, 3.5), layout='constrained')
    for i, (method, color, hatch) in enumerate([('Gaussian', GREY, ''), ('GW', BLUE, '//')]):
        keys = [(domain, method) for domain in ['Gaussian', 'LogGamma']]
        rows = summary.loc[keys]
        axis.bar(np.arange(2)+(i-.5)*.28, rows.Mean, width=.26, color=color,
                 hatch=hatch, yerr=rows.High-rows.Mean, capsize=4, label=method)
    axis.set_xticks([0, 1], ['Gaussian', 'LogGamma'])
    axis.set(xlabel='Input noise domain', ylabel='Test ECE (%) ↓', ylim=(0, .65))
    axis.legend(ncol=2, loc='upper right')
    return figure, {'per_seed_ece': raw, 'ece_intervals': summary}


def distribution_panels(root):
    config = json.loads(checked(root, 'configs/fit.json').read_text())
    source = next(r for r in config['datasets'] if r['id'] == 'opensar')
    records = json.loads(checked(root, source['records']).read_text())
    with np.load(checked(root, source['bank']), allow_pickle=False) as z:
        bank = z['model_gray'].copy()
    _, hold, _ = fitting.grouped_samples(bank, records, 91501)
    for filename in ['run.json', 'rows.json']:
        checked(root, f'results/fit/refit/{filename}')
    rows, _ = fitting.summarize(Path(root)/'results/fit/refit')
    fits = rows[(rows.dataset == source['source_id']) & (rows.coordinate == 'model_gray') & (rows.seed == 91501)]
    x = np.linspace(.001, 4, 400)
    curves = pd.DataFrame({'Relative intensity': x})
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.3), layout='constrained',
                                gridspec_kw={'width_ratios': [.75, 1.4, 1.2]})
    axes[0].imshow(bank[0], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[0].axis('off'); axes[0].set_title('Selected block', loc='left')
    histogram, edges = np.histogram(hold, bins=np.linspace(0, 4, 61))
    density = histogram / (len(hold)*np.diff(edges))
    axes[1].bar(edges[:-1], density, width=np.diff(edges), align='edge', color='#dce4e8', label='Held-out pixels')
    for family, color, linestyle in [('GeneralizedGamma', BLUE, '-'), ('Gaussian', GREY, '--'),
                                      ('Uniform', LIGHT, ':'), ('Laplace', GOLD, '-.')]:
        row = fits[fits.family == family].iloc[0]
        curves[family] = fitting.FAMILIES[family].pdf(x, *row.parameters)
        axes[1].plot(x, curves[family], color=color, linestyle=linestyle,
                     label='Generalized Gamma' if family == 'GeneralizedGamma' else family)
    axes[1].set(xlabel='Pixel / block mean', ylabel='Density', xlim=(0, 4))
    axes[1].legend(fontsize=8, ncol=2, loc='upper right')
    family_config = presentation.read_experiment(root, 's4_family')
    metrics = presentation.load_predictions(root, family_config, Path(root)/'results/s4_family/gpu_predictions')
    mean, sd = presentation.summarize(metrics, family_config)
    axes[2].bar(range(len(mean)), mean['ECE (%)'], yerr=sd['ECE (%)'], capsize=3,
                 color=[BLUE, GREY, LIGHT, GOLD])
    axes[2].set_xticks(range(len(mean)), mean.index, rotation=25, ha='right')
    axes[2].set(ylabel='Validation ECE (%) ↓', ylim=(0, 10))
    return figure, {'fit_parameters': fits, 'density_curves': curves,
                    'heldout_histogram': pd.DataFrame({'Low': edges[:-1], 'High': edges[1:], 'Count': histogram, 'Density': density}),
                    'per_seed_ece': metrics, 'block_identity': pd.DataFrame([dict(Index=0, **records[0])])}


GENERATORS = {2: mechanism, 3: directional, 4: optical_noise, 6: distribution_panels}


def run(root, number, mode='quantitative'):
    root = Path(root)
    record = next((r for r in component_catalog(root) if r['number'] == number), None)
    if record is None or mode not in ['paper', 'quantitative']:
        raise ValueError('Unknown figure or rendering mode')
    if mode == 'quantitative' and number not in GENERATORS:
        raise ValueError('This figure is schematic; use paper mode for its saved artwork')
    destination = root/'output/figures'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')/f'figure{number}'
    destination.mkdir(parents=True, exist_ok=False)
    state = dict(figure=number, mode=mode, status='running', model_inference_executed=False,
                 code_sha256=presentation.sha256(Path(__file__)),
                 input_manifest_sha256=presentation.sha256(root/'artifacts/manifest.json'),
                 environment=dict(numpy=np.__version__, matplotlib=matplotlib.__version__),
                 scope='Original raster files copied byte-for-byte' if mode == 'paper' else REDRAW_SCOPE[number])
    state['statistics'] = {
        2: {'paired_seeds': 30, 'curve_seeds': 3, 'curve_band': 'min/max',
            'reliability_bins': 15, 'reliability_aggregation': 'pooled three-seed test predictions',
            'heatmap_targets': 'balanced shuffled hard labels'},
        3: {'paired_seeds': 100, 'central_value': 'exp(mean(log(R+1e-12)))',
            'bootstrap_draws': 10000, 'bootstrap_seed': 20261009, 'band': 'pointwise 95% percentile'},
        4: {'paired_seeds': 10, 'interval': 'mean ± t(0.975,9)*sample_SD/sqrt(10)'},
        6: {'fit_seed': 91501, 'block_index': 0, 'histogram': 'all held-out positive pixels as denominator',
            'warmup_seeds': 5, 'error_bars': 'sample SD'},
    }.get(number, {}) if mode == 'quantitative' else {}
    try:
        if mode == 'paper':
            for asset in record['assets']:
                shutil.copyfile(checked(root, asset['path']), destination/Path(asset['path']).name)
        else:
            style()
            figure, sources = GENERATORS[number](root)
            for extension in ['png', 'svg', 'pdf']:
                figure.savefig(destination/f'figure{number}.{extension}', dpi=220, bbox_inches='tight')
            plt.close(figure)
            for name, table in sources.items():
                table.to_csv(destination/f'{name}.csv', index=True)
        state.update(status='complete', output_files={p.name:presentation.sha256(p) for p in destination.iterdir() if p.is_file()})
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (destination/'run.json').write_text(json.dumps(state, indent=2)+'\n')
    return destination

# Compatibility names refer to historical generator components only.
catalog = component_catalog
paper_html = component_html

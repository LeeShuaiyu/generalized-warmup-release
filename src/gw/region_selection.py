"""Pixel-only region rules and fixed training-image bank reconstruction."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import json
import shutil

import numpy as np
from PIL import Image
from scipy import ndimage

from .assets import file_hash, relative_path
from .classification_data import load_manifest, manifest_sha256
from .supporting_reports import checked

OCT_PARAMS = dict(block=16, context=48, margin=16, gap=24, x_fraction=(.1, .9), sigma=2,
                  threshold_floor=.15, min_component_width=.4, min_component_area=.01,
                  mean=(.03, .20), std_min=.01, cv_max=1, zmax=6, saturation_max=.1,
                  quadrant_max=.5, fine_max=2, context_range_max=2)


def sar_pixels(path):
    """Keep the original native-resolution float operations, before resizing."""
    path = Path(path)
    if path.suffix == '.npy':
        source = np.load(path, allow_pickle=False)
        if np.iscomplexobj(source):
            raise ValueError('Complex input is unsupported')
        raw = source.astype(np.float32)
        lo, hi = np.percentile(raw[np.isfinite(raw)], [1, 99])
        gray = np.clip((raw - lo) / max(hi - lo, 1e-30), 0, 1)
    else:
        with Image.open(path) as image:
            raw = np.asarray(image.convert('L'), dtype=np.float32)
        gray = raw / 255
    if gray.ndim != 2 or not np.isfinite(gray).all():
        raise ValueError('Expected finite 2D image')
    return raw, gray


def select_sar(gray):
    gray = np.asarray(gray)
    if gray.ndim != 2 or not np.isfinite(gray).all() or gray.min() < 0 or gray.max() > 1:
        raise ValueError('Expected finite 2D gray in [0,1]')
    h, w = gray.shape
    hh, ww = h // 16, w // 16
    if not hh or not ww:
        return None
    patches = gray[:hh * 16, :ww * 16].reshape(hh, 16, ww, 16).transpose(0, 2, 1, 3).reshape(-1, 16, 16)
    mean, std = patches.mean((1, 2)), patches.std((1, 2))
    scale = np.maximum(std, 1e-20)
    quadrant = patches.reshape(-1, 2, 8, 2, 8).mean((2, 4)).reshape(-1, 4)
    fine = patches.reshape(-1, 4, 4, 4, 4).mean((2, 4)).reshape(-1, 16)
    score = np.ptp(quadrant, axis=1) / scale
    valid = ((mean >= .03) & (mean <= .97) & (std >= .01) &
             (((patches <= 0) | (patches >= 1)).mean((1, 2)) < .1) &
             (std / np.maximum(mean, 1e-20) <= 1) &
             ((patches.max((1, 2)) - mean) / scale <= 6) &
             (np.ptp(fine, axis=1) / scale <= 2) & (score <= .5))
    indices = np.flatnonzero(valid)
    if not len(indices):
        return None
    index = int(indices[np.argmin(score[indices])])
    return {'y': (index // ww) * 16, 'x': (index % ww) * 16, 'score': float(score[index])}


def otsu(array):
    histogram, edges = np.histogram(array, bins=256, range=(0, 1))
    probability = histogram.astype(float) / histogram.sum()
    cumulative = np.cumsum(probability)
    centers = (edges[:-1] + edges[1:]) / 2
    mean = np.cumsum(probability * centers)
    between = (mean[-1] * cumulative - mean)**2 / np.maximum(cumulative * (1 - cumulative), 1e-20)
    return float(centers[np.argmax(between)])


def select_background(array):
    """Original OCT image-only background rule; not semantic independence proof."""
    array = np.asarray(array, dtype=float)
    if array.ndim != 2 or not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError('Expected finite 2D gray in [0,1]')
    h, w = array.shape
    if min(h, w) < 96:
        return None, dict(reason='too_small'), None
    smooth = ndimage.gaussian_filter(array, OCT_PARAMS['sigma'])
    threshold = max(OCT_PARAMS['threshold_floor'], otsu(smooth))
    mask = ndimage.binary_closing(smooth >= threshold, structure=np.ones((5, 5)))
    labels, _ = ndimage.label(mask)
    regions = ndimage.find_objects(labels)
    areas = np.bincount(labels.ravel())
    eligible = [i for i, sl in enumerate(regions, 1) if sl is not None and
                sl[1].stop - sl[1].start >= w * .4 and areas[i] >= h * w * .01]
    if not eligible:
        return None, dict(reason='no_broad_tissue_proxy', threshold=threshold), None
    component = max(eligible, key=lambda i: areas[i])
    ys, xs = np.nonzero(labels == component)
    lo, hi = int(xs.min()), int(xs.max())
    top = np.full(w, np.nan)
    for x in np.unique(xs):
        top[x] = ys[xs == x].min()
    ok = np.flatnonzero(np.isfinite(top))
    top[lo:hi + 1] = np.interp(np.arange(lo, hi + 1), ok, top[ok])
    counts = dict(total=0, geometry=0, brightness=0, homogeneity=0, context=0)
    best = None
    for y in range(0, h - 15, 16):
        for x in range(0, w - 15, 16):
            counts['total'] += 1
            x0, x1, y0, y1 = x - 16, x + 32, y - 16, y + 32
            if x0 < max(16, w * .1, lo) or x1 > min(w - 16, w * .9, hi + 1) or y0 < 16 or y1 > h - 16:
                continue
            tissue = top[x0:x1]
            if not np.isfinite(tissue).all() or y1 > tissue.min() - 24:
                continue
            counts['geometry'] += 1
            patch = array[y:y + 16, x:x + 16]
            mean, std = patch.mean(), patch.std()
            if not (.03 <= mean <= .20 and std >= .01):
                continue
            counts['brightness'] += 1
            quadrant = patch.reshape(2, 8, 2, 8).mean((1, 3))
            fine = patch.reshape(4, 4, 4, 4).mean((1, 3))
            score = float(np.ptp(quadrant) / std)
            if (std / mean > 1 or (patch.max() - mean) / std > 6 or
                    np.mean((patch <= 0) | (patch >= 1)) >= .1 or score > .5 or np.ptp(fine) / std > 2):
                continue
            counts['homogeneity'] += 1
            context = array[y0:y1, x0:x1]
            context_mean = context.reshape(3, 16, 3, 16).mean((1, 3))
            context_range = float(np.ptp(context_mean) / std)
            if context_range > 2 or smooth[y0:y1, x0:x1].max() >= threshold:
                continue
            counts['context'] += 1
            record = dict(x=x, y=y, mean=float(mean), std=float(std), score=score,
                          context_range=context_range, tissue_threshold=threshold,
                          distance_to_tissue=float(tissue.min() - (y + 16)))
            if best is None or score < best['score']:
                best = record
    return best, dict(reason='selected' if best else 'no_valid_background',
                      counts=counts, threshold=threshold), top


def _registered_inputs(root, dataset, source_dir=None):
    config = json.loads(checked(root, 'configs/region_selection.json').read_text())
    if config['split'] != 'train' or config['block_size'] != 16:
        raise ValueError('Unsupported region-selection protocol')
    matching = [r for r in config['datasets'] if r['id'] == dataset]
    if len(matching) != 1:
        raise ValueError('Unknown dataset')
    item = matching[0]
    manifest = root / relative_path(item['manifest'])
    if manifest_sha256(manifest) != item['semantic_sha256']:
        raise ValueError('Fixed image split changed')
    all_rows = load_manifest(manifest)
    rows = [r for r in all_rows if r['split'] == 'train']
    if not rows:
        raise ValueError('No training images')
    catalog = json.loads(checked(root, config['files']).read_text())[dataset]
    files = {r['sample_id']: r for r in catalog['files']}
    if len(files) != len(catalog['files']) or set(files) != {r['sample_id'] for r in all_rows}:
        raise ValueError('Duplicate or missing image identity')
    source_root = (Path(source_dir) if source_dir is not None else root / item['source_directory']).resolve()
    inputs = []
    for row in rows:
        record = files[row['sample_id']]
        if row['data_path'] != record['path'] or row['split'] != record['split']:
            raise ValueError('Image path or split differs from catalog')
        path = source_root / relative_path(record['path'])
        if not path.resolve().is_relative_to(source_root):
            raise ValueError('Image path resolves outside source directory')
        if not path.is_file() or file_hash(path) != record['sha256']:
            raise ValueError(f'Training image missing or inconsistent: {row["sample_id"]}')
        if not row.get('group_id'):
            raise ValueError('Missing acquisition or patient group')
        inputs.append((row, record, path))
    return item, inputs


def compare_bank(root, item, bank, records):
    expected_records = json.loads(checked(root, item['expected_records']).read_text())
    keys = ['sample_id', 'group_id', 'x', 'y']
    identities_equal = [{k: r[k] for k in keys} for r in records] == expected_records
    values = {}
    with np.load(checked(root, item['expected_bank']), allow_pickle=False) as expected:
        for key in expected.files:
            actual = bank.get(key)
            values[key] = bool(actual is not None and actual.dtype == expected[key].dtype and
                               np.array_equal(actual, expected[key]))
    return {'selected_identities_and_coordinates_equal': identities_equal,
            'pixels_and_dtype_equal': values,
            'matches_fixed_bank': identities_equal and all(values.values())}


def run(root, dataset, *, source_dir=None, limit=None):
    """Re-select training-image regions; never overwrite the paper's fixed banks."""
    root = Path(root).resolve()
    item, inputs = _registered_inputs(root, dataset, source_dir)
    total = len(inputs)
    if limit is not None:
        if limit <= 0:
            raise ValueError('Limit must be positive')
        inputs = inputs[:limit]
    parent = root / 'output/regions'
    if not parent.resolve().is_relative_to(root):
        raise ValueError('Output resolves outside the project')
    parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(parent).free < 8 * 1024**3 + 2 * 16 * 16 * 8 * len(inputs):
        raise RuntimeError('Keep at least 8 GiB available')
    output = parent / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir()
    state = {'status': 'running', 'dataset': dataset, 'rule': item['rule'],
             'registered_train_images': total, 'processed_images': len(inputs), 'diagnostic_limit': limit,
             'manifest_semantic_sha256': item['semantic_sha256'],
             'code_sha256': file_hash(Path(__file__)), 'config_sha256': file_hash(root / 'configs/region_selection.json'),
             'model_inference_executed': False, 'distribution_fitting_executed': False,
             'selection_uses_class_labels': False, 'input_files': [], 'numpy': np.__version__}
    chosen, image_audit = {}, []
    try:
        for i, (row, registered, path) in enumerate(inputs):
            if item['rule'] == 'oct_background':
                with Image.open(path) as image:
                    native = np.asarray(image.convert('L'), dtype=float) / 255
                selection, metadata, _ = select_background(native)
                if selection is not None:
                    y, x = selection['y'], selection['x']
                    raw_block = native[y:y + 16, x:x + 16].copy()
                    lo, hi = np.percentile(native, [1, 99])
                    if hi <= lo:
                        raise ValueError('Constant OCT image cannot be scaled')
                    scaled_block = np.clip((raw_block - lo) / (hi - lo), 0, 1)
                group_key = row['group_id']
            elif item['rule'] == 'sar_homogeneous':
                native, scaled = sar_pixels(path)
                selection = select_sar(scaled)
                metadata = {'reason': 'selected' if selection else 'no_valid_region'}
                if selection is not None:
                    y, x = selection['y'], selection['x']
                    raw_block = native[y:y + 16, x:x + 16].copy()
                    scaled_block = scaled[y:y + 16, x:x + 16].copy()
                group_key = row['sample_id']
            else:
                raise ValueError('Unsupported selector rule')
            image_audit.append(dict(sample_id=row['sample_id'], group_id=row['group_id'], **metadata))
            state['input_files'].append(dict(sample_id=row['sample_id'], path=row['data_path'], sha256=registered['sha256']))
            if selection is not None:
                # Attach identities after the pixel-only rule returns.
                record = dict(selection, sample_id=row['sample_id'], group_id=row['group_id'])
                previous = chosen.get(group_key)
                if previous is None or record['score'] < previous[0]['score']:
                    chosen[group_key] = (record, raw_block, scaled_block)
            if (i + 1) % 100 == 0:
                print(f'Regions · {dataset} · {i + 1}/{len(inputs)} · {len(chosen)} selected', flush=True)
        if not chosen:
            raise ValueError('No selected regions; thresholds are unchanged')
        records = [value[0] for value in chosen.values()]
        bank = {'stored': np.stack([value[1] for value in chosen.values()]),
                'model_gray': np.stack([value[2] for value in chosen.values()])}
        np.savez_compressed(output / 'bank.npz', **bank)
        (output / 'records.json').write_text(json.dumps(records, indent=2) + '\n')
        (output / 'image_audit.json').write_text(json.dumps(image_audit, indent=2) + '\n')
        comparison = compare_bank(root, item, bank, records) if limit is None else {'scope': 'diagnostic subset; no full-bank claim'}
        (output / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
        state.update(status='complete', selected=len(records),
                     selected_groups=len({r['group_id'] for r in records}),
                     reasons=dict(Counter(r['reason'] for r in image_audit)), comparison=comparison,
                     output_files={name: file_hash(output / name) for name in
                                   ['bank.npz', 'records.json', 'image_audit.json', 'comparison.json']})
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        error.add_note(f'Region-selection record: {output / "run.json"}')
        raise
    finally:
        (output / 'run.json').write_text(json.dumps(state, indent=2) + '\n')
    return output

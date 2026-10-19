"""Convert provider files to the registered, immutable classification inputs."""
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import importlib.metadata
import json
import os
import re
import shutil
import tempfile
import zipfile

import numpy as np
from PIL import Image

from .assets import file_hash, relative_path
from .classification_data import load_manifest, manifest_sha256


def registered_rows(root, dataset):
    root = Path(root).resolve()
    catalog = json.loads((root / 'data/files.json').read_text())
    if dataset not in {'sar', 'opensar', 'oct'} or dataset not in catalog:
        raise ValueError('Unknown dataset')
    item = catalog[dataset]
    manifest = root / relative_path(item['manifest'])
    if manifest_sha256(manifest) != item['semantic_sha256']:
        raise ValueError('Data split changed')
    rows = load_manifest(manifest)
    files = {r['sample_id']: r for r in item['files']}
    if len(files) != len(item['files']) or set(files) != {r['sample_id'] for r in rows}:
        raise ValueError('Sample identities differ between manifest and file catalog')
    paths, groups = set(), defaultdict(set)
    for row in rows:
        record = files[row['sample_id']]
        path = relative_path(record['path'])
        if str(path) != row['data_path'] or record['split'] != row['split']:
            raise ValueError('Sample path or split differs from file catalog')
        if path in paths:
            raise ValueError('Duplicate processed sample path')
        paths.add(path)
        digest = record['sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid processed file hash')
        group = row.get('group_id')
        if not group:
            raise ValueError('Missing acquisition or patient group')
        groups[group].add(row['split'])
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError('Acquisition or patient group occurs in multiple splits')
    return rows, files, item


def audit(root, dataset):
    rows, _, item = registered_rows(root, dataset)
    counts = Counter((r['split'], r['class_id']) for r in rows)
    return {'dataset': dataset, 'samples': len(rows),
            'groups': len({r['group_id'] for r in rows}),
            'semantic_sha256': item['semantic_sha256'],
            'split_counts': dict(Counter(r['split'] for r in rows)),
            'class_split_counts': {f'{split}/{label}': count
                                   for (split, label), count in sorted(counts.items())},
            'group_split_overlap': 0}


def load_sar_records(path):
    records = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(records, list) and len(records) == 1 and isinstance(records[0], list):
        records = records[0]
    if not isinstance(records, list):
        raise ValueError('SAR provider JSON must contain a record list')
    return records


def decode_patch(payload):
    # Preserve the original TIFF-first decoder and its grayscale fallback.
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError('Install the data extra: pip install -e ".[data]"') from exc
    try:
        array = np.asarray(tifffile.imread(io.BytesIO(payload)))
        if array.ndim == 3:
            array = array.mean(axis=2)
        array = array.astype('float32', copy=False)
        height, width = array.shape[:2]
        return array, width, height, 'tifffile'
    except Exception as exc:
        with Image.open(io.BytesIO(payload)) as image:
            gray = image.convert('L')
            array = np.asarray(gray, dtype='float32') / 255.0
            width, height = gray.size
        return array, width, height, f'Pillow fallback ({type(exc).__name__})'


def npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def source_spec(row, dataset):
    if dataset == 'sar':
        match = re.fullmatch(r'(ship_positives|false_positives|true_negatives)_(\d{6})', row['sample_id'])
        if not match:
            raise ValueError('Invalid SAR sample identity')
        component, index = match.group(1), int(match.group(2))
        labels = {'false_positives': ('lookalike', 0), 'true_negatives': ('ocean_clutter', 1),
                  'ship_positives': ('ship', 2)}
        label, class_id = labels[component]
        if row['label'] != label or int(row['class_id']) != class_id:
            raise ValueError('SAR class mapping changed')
        if Path(row['raw_path']).name != f'{component}.json':
            raise ValueError('SAR source component changed')
        return Path(f'{component}.json'), index
    if dataset == 'opensar':
        archive, separator, member = row['raw_path'].partition('!')
        if not separator:
            raise ValueError('Missing OpenSAR archive member')
        archive_name = Path(archive).name
        relative_path(member)
        if not archive_name.endswith('.zip') or '_GRDH_1SDV_' not in archive_name:
            raise ValueError('Expected OpenSAR GRD/SDV ZIP source')
        if not member.startswith('Patch_Cal/') or Path(member).suffix.lower() not in {'.tif', '.tiff'}:
            raise ValueError('Expected calibrated OpenSAR TIFF member')
        label = row['label']
        mapping = {'Cargo': 0, 'Tanker': 1, 'Other Type': 2}
        if label not in mapping or int(row['class_id']) != mapping[label]:
            raise ValueError('OpenSAR class mapping changed')
        slug = re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_')
        identifier = hashlib.sha1(f'{archive_name}|{member}'.encode()).hexdigest()[:12]
        if row['sample_id'] != f'opensar_grd3_{slug}_{identifier}' or row['group_id'] != archive_name:
            raise ValueError('OpenSAR acquisition identity changed')
        return Path(archive_name), member
    marker = 'data/octdl/images/OCTDL/'
    if not row['data_path'].startswith(marker):
        raise ValueError('Unexpected OCT source path')
    source = relative_path(row['data_path'][len(marker):])
    if source.parts[0] != row['class_name'] or source.stem != row['sample_id']:
        raise ValueError('OCT class or sample identity changed')
    return source, None


def _save_verified(destination, payload, expected_hash):
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_hash:
        raise ValueError(f'Converted bytes differ from registered input: {destination.name}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < 8 * 1024**3 + len(payload):
        raise RuntimeError('Keep at least 8 GiB available after conversion')
    descriptor, temporary = tempfile.mkstemp(prefix='.gw-convert-', dir=destination.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
        os.link(temporary, destination)  # No replacement, including racing writers.
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def convert(root, dataset, raw_dir, *, split=None, limit=None, progress=None):
    """Write a new output tree; do not modify source, manifest, or installed data."""
    root, raw_dir = Path(root).resolve(), Path(raw_dir).resolve()
    rows, files, item = registered_rows(root, dataset)
    packages = {name: importlib.metadata.version(name) for name in ('numpy', 'Pillow')}
    if dataset == 'opensar':
        try:
            packages['tifffile'] = importlib.metadata.version('tifffile')
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError('Install the data extra: pip install -e ".[data]"') from exc
    if split is not None and split not in {'train', 'val', 'test'}:
        raise ValueError('Unknown split')
    selected = [r for r in rows if split is None or r['split'] == split]
    if limit is not None:
        if limit <= 0:
            raise ValueError('Limit must be positive')
        selected = selected[:limit]
    if not selected:
        raise ValueError('No selected samples')
    sources = {}
    plan = []
    estimated_bytes = 0
    for row in selected:
        relative, reference = source_spec(row, dataset)
        source = raw_dir / relative
        if not source.resolve().is_relative_to(raw_dir):
            raise ValueError('Source resolves outside the supplied directory')
        if not source.is_file():
            raise FileNotFoundError(f'Missing provider file: {source}')
        sources.setdefault(source, None)
        plan.append((row, source, reference))
        estimated_bytes += (source.stat().st_size if dataset == 'oct' else
                            256 + 4 * int(row['width']) * int(row['height']))
    output_parent = root / 'output/data_conversion'
    if not output_parent.resolve().is_relative_to(root):
        raise ValueError('Output resolves outside the project')
    output_parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_parent).free < 8 * 1024**3 + estimated_bytes:
        raise RuntimeError('Keep at least 8 GiB available after conversion')
    output = output_parent / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir()
    report = {'status': 'running', 'dataset': dataset, 'selected': len(selected),
              'split': split, 'diagnostic_limit': limit, 'semantic_sha256': item['semantic_sha256'],
              'manifest_sha256': file_hash(root / item['manifest']), 'sources': [], 'samples': [],
              'source_code_sha256': file_hash(Path(__file__)), 'model_inference_executed': False}
    report['packages'] = packages
    try:
        # Group by provider file so only one SAR component list is held in memory.
        by_source = defaultdict(list)
        for entry in plan:
            by_source[entry[1]].append(entry)
        for source, entries in by_source.items():
            original_stat = source.stat()
            report['sources'].append({'path': str(source), 'bytes': original_stat.st_size,
                                      'sha256': file_hash(source)})
            with ExitStack() as stack:
                records = load_sar_records(source) if dataset == 'sar' else None
                archive = stack.enter_context(zipfile.ZipFile(source)) if dataset == 'opensar' else None
                if archive is not None:
                    members = Counter(archive.namelist())
                    if any(members[entry[2]] != 1 for entry in entries):
                        raise ValueError('Missing or duplicate registered archive member')
                for row, _, reference in entries:
                    if dataset == 'sar':
                        record = records[reference]
                        if str(record.get('fullname') or record.get('id') or row['sample_id']) != row['group_id']:
                            raise ValueError('SAR acquisition group differs from manifest')
                        array = np.asarray(record['windowfu'], dtype=np.float32)
                        decoder, payload = 'JSON windowfu float32', npy_bytes(array)
                    elif dataset == 'opensar':
                        raw_payload = archive.read(reference)
                        array, _, _, decoder = decode_patch(raw_payload)
                        payload = npy_bytes(array)
                    else:
                        payload, decoder, array = source.read_bytes(), 'unchanged JPEG', None
                    if array is not None and array.shape != (int(row['height']), int(row['width'])):
                        raise ValueError('Converted image dimensions differ from manifest')
                    record = files[row['sample_id']]
                    destination = output / 'files' / relative_path(record['path'])
                    digest = _save_verified(destination, payload, record['sha256'])
                    report['samples'].append({'sample_id': row['sample_id'], 'split': row['split'],
                                              'class_id': row['class_id'], 'group_id': row['group_id'],
                                              'path': record['path'], 'sha256': digest, 'decoder': decoder})
                    if progress and (len(report['samples']) % 500 == 0):
                        progress(len(report['samples']), len(selected))
            final_stat = source.stat()
            if (original_stat.st_size, original_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
                raise ValueError('Provider file changed during conversion')
        report['status'] = 'complete'
    except Exception as exc:
        report['status'], report['error'] = 'failed', f'{type(exc).__name__}: {exc}'
        exc.add_note(f'Conversion record: {output / "run.json"}')
        raise
    finally:
        (output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    return output

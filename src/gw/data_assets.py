"""Validate or copy the fixed, processed dataset files used by the paper."""
from pathlib import Path
import json
import hashlib
import os
import shutil
import tempfile

from .assets import file_hash, relative_path
from .classification_data import manifest_sha256


def verify_inputs(root, dataset, rows, split, data_root):
    """Check the exact files about to be evaluated, including custom read-only roots."""
    root, data_root = Path(root).resolve(), Path(data_root)
    catalog_path = root / 'data/files.json'
    catalog = json.loads(catalog_path.read_text())
    if dataset not in catalog or split not in {'train', 'val', 'test'}:
        raise ValueError('Unknown dataset or evaluation split')
    item = catalog[dataset]
    if manifest_sha256(root / relative_path(item['manifest'])) != item['semantic_sha256']:
        raise ValueError('Data split changed')
    semantic = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value['sample_id']):
        stable = {key: value for key, value in row.items() if not key.endswith('_path') and key != 'path'}
        semantic.update(json.dumps(stable, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())
        semantic.update(b'\n')
    if semantic.hexdigest() != item['semantic_sha256']:
        raise ValueError('Evaluation rows differ from the registered manifest')
    selected = [row for row in rows if row['split'] == split]
    expected_rows = [row for row in item['files'] if row['split'] == split]
    expected = {row['sample_id']: row for row in expected_rows}
    identities = [row['sample_id'] for row in selected]
    if (not selected or len(expected) != len(expected_rows) or len(identities) != len(set(identities))
            or set(identities) != set(expected)):
        raise ValueError('Evaluation sample identities differ from the registered split')
    for row in selected:
        record = expected[row['sample_id']]
        if row['data_path'] != record['path']:
            raise ValueError('Evaluation input path differs from the registered file')
        path = data_root / relative_path(row['data_path'])
        if not path.is_file():
            raise FileNotFoundError(f'Required evaluation image: {row["sample_id"]}')
        if file_hash(path) != record['sha256']:
            raise ValueError(f'Evaluation image checksum mismatch: {row["sample_id"]}')
    return {'dataset': dataset, 'split': split, 'samples_verified': len(selected),
            'semantic_sha256': item['semantic_sha256'], 'file_catalog_sha256': file_hash(catalog_path)}


def prepare(root, dataset, source_root=None, *, check=False, split=None):
    root = Path(root).resolve()
    catalog = json.loads((root / 'data/files.json').read_text())
    if dataset not in catalog:
        raise ValueError('Unknown dataset')
    item = catalog[dataset]
    if manifest_sha256(root / relative_path(item['manifest'])) != item['semantic_sha256']:
        raise ValueError('Data split changed')
    if split is not None and split not in {'train', 'val', 'test'}:
        raise ValueError('Unknown split')
    files = [r for r in item['files'] if split is None or r['split'] == split]
    if not files:
        raise ValueError('No selected samples')
    destination_root = root / 'data/processed' / dataset
    plan, counts = [], {'verified': 0, 'prepared': 0, 'missing': 0, 'invalid': 0}
    for record in files:
        relative = relative_path(record['path'])
        destination = destination_root / relative
        if destination.is_file():
            # Existing server datasets may be linked to a separately stored
            # provider tree. Read-only hash verification is allowed; no copy
            # operation may follow an escaping parent link.
            if file_hash(destination) != record['sha256']:
                counts['invalid'] += 1
                if not check:
                    raise ValueError(f'Refusing replacement of inconsistent sample: {record["sample_id"]}')
            else:
                counts['verified'] += 1
            continue
        if destination.exists() or destination.is_symlink():
            raise ValueError('Invalid data destination')
        if not destination.resolve().is_relative_to(root):
            raise ValueError('Data destination resolves outside the project')
        if check:
            counts['missing'] += 1
            continue
        if source_root is None:
            raise ValueError('Supply the processed source tree with --from-dir')
        if not destination.parent.resolve().is_relative_to(root):
            raise ValueError('Data destination resolves outside the project')
        source = Path(source_root) / relative
        if not source.is_file() or file_hash(source) != record['sha256']:
            raise ValueError(f'Source file missing or inconsistent: {record["sample_id"]}')
        plan.append((source, destination, record['sha256']))
    for source, destination, expected_hash in plan:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(destination.parent).free < source.stat().st_size + 8 * 1024**3:
            raise RuntimeError('Keep at least 8 GiB available after data preparation')
        descriptor, temporary = tempfile.mkstemp(prefix='.gw-data-', dir=destination.parent)
        temporary = Path(temporary)
        try:
            with source.open('rb') as input_stream, os.fdopen(descriptor, 'wb') as output_stream:
                shutil.copyfileobj(input_stream, output_stream)
            if file_hash(temporary) != expected_hash:
                raise ValueError('Copied data file is inconsistent')
            os.link(temporary, destination)
            counts['prepared'] += 1
        finally:
            temporary.unlink(missing_ok=True)
    return counts

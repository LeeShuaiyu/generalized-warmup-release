"""Prepare fixed inputs from provider files; never redraw dataset splits."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.request
import zipfile

from .assets import file_hash, relative_path
from ._files import install_verified_file
from .data_assets import prepare
from .raw_data import audit, convert, registered_rows, source_spec, _save_verified

# Version-specific provider record, checked against Zenodo on 2026-10-09.
# Processed image SHA256s are checked separately against the fixed file catalog.
OCTDL = {
    'record': 'https://zenodo.org/records/10839556',
    'url': 'https://zenodo.org/api/records/10839556/files/OCTDL.zip/content',
    'file': 'OCTDL.zip', 'bytes': 398564236,
    'md5': 'dec9766fc916202c45a333c4f5a09576',
}
RESERVE = 8 * 1024**3


def verify_archive(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != OCTDL['bytes']:
        raise ValueError('OCTDL archive size or file type differs from the provider record')
    digest = hashlib.md5(usedforsecurity=False)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    if digest.hexdigest() != OCTDL['md5']:
        raise ValueError('OCTDL archive checksum differs from the provider record')
    return file_hash(path)


def download_oct(root):
    """Download one pinned public archive, reusing only a verified cache file."""
    root = Path(root).resolve()
    cache = root / 'data/downloads/octdl-10839556'
    if not cache.resolve().is_relative_to(root):
        raise ValueError('Download cache resolves outside the project')
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / OCTDL['file']
    if target.exists() or target.is_symlink():
        verify_archive(target)
        return target
    if shutil.disk_usage(cache).free < RESERVE + OCTDL['bytes']:
        raise RuntimeError('Keep at least 8 GiB available after download')
    descriptor, temporary = tempfile.mkstemp(prefix='.gw-download-', dir=cache)
    temporary = Path(temporary)
    try:
        request = urllib.request.Request(OCTDL['url'], headers={'User-Agent': 'GW-data-preparation/0.1'})
        with os.fdopen(descriptor, 'wb') as output, urllib.request.urlopen(request, timeout=60) as source:
            # A redirect may stay on the provider, but may not downgrade HTTPS.
            if not source.geturl().startswith('https://zenodo.org/'):
                raise ValueError('Unexpected provider download redirect')
            total = 0
            while chunk := source.read(1024**2):
                total += len(chunk)
                if total > OCTDL['bytes']:
                    raise ValueError('Provider download exceeds its registered size')
                if shutil.disk_usage(cache).free < RESERVE + len(chunk):
                    raise RuntimeError('Keep at least 8 GiB available during download')
                output.write(chunk)
        verify_archive(temporary)
        install_verified_file(temporary, target, reserve_bytes=RESERVE)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def unpack_oct(root, archive_path, *, split=None):
    """Copy only registered JPEG members into a new provider tree, checking SHA256."""
    root = Path(root).resolve()
    if split is not None and split not in {'train', 'val', 'test'}:
        raise ValueError('Unknown split')
    rows, files, item = registered_rows(root, 'oct')
    selected = [row for row in rows if split is None or row['split'] == split]
    if not selected:
        raise ValueError('No selected samples')
    archive_path = Path(archive_path).absolute()
    before = archive_path.stat()
    digest = verify_archive(archive_path)
    parent = root / 'output/provider_data'
    if not parent.resolve().is_relative_to(root):
        raise ValueError('Provider output resolves outside the project')
    # Inspect all metadata before creating an output tree. No extractall().
    with zipfile.ZipFile(archive_path) as archive:
        members = {}
        for info in archive.infolist():
            name = info.filename.rstrip('/') if info.is_dir() else info.filename
            path = relative_path(name)
            if path.as_posix() != name or info.filename in members:
                raise ValueError('Noncanonical or duplicate provider ZIP member')
            kind = stat.S_IFMT(info.external_attr >> 16)
            if kind not in {0, stat.S_IFREG, stat.S_IFDIR} or info.flag_bits & 1:
                raise ValueError('Linked, special or encrypted provider ZIP member')
            members[info.filename] = info
        plan = []
        for row in selected:
            relative, _ = source_spec(row, 'oct')
            # Provider releases may include the OCTDL/ top-level directory.
            matches = [name for name in [f'OCTDL/{relative.as_posix()}', relative.as_posix()]
                       if name in members and not members[name].is_dir()]
            if len(matches) != 1:
                raise ValueError(f'Missing or ambiguous registered OCTDL JPEG: {relative}')
            info = members[matches[0]]
            if not 0 < info.file_size <= 64 * 1024**2:
                raise ValueError('Unexpected provider JPEG size')
            plan.append((row, relative, info))
        parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(parent).free < RESERVE + sum(info.file_size for _, _, info in plan):
            raise RuntimeError('Keep at least 8 GiB available after unpacking')
        output = parent / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output.mkdir()
        report = {'status': 'running', 'dataset': 'oct', 'split': split, 'selected': len(selected),
                  'provider': dict(OCTDL), 'archive_sha256': digest,
                  'semantic_sha256': item['semantic_sha256'], 'samples': [],
                  'source_code_sha256': file_hash(Path(__file__)), 'model_inference_executed': False}
        try:
            for row, relative, info in plan:
                payload = archive.read(info)
                image_digest = _save_verified(output / 'OCTDL' / relative, payload,
                                              files[row['sample_id']]['sha256'])
                report['samples'].append({'sample_id': row['sample_id'], 'split': row['split'],
                                          'member': info.filename, 'sha256': image_digest})
            after = archive_path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Provider archive changed during unpacking')
            report['status'] = 'complete'
        except Exception as exc:
            report['status'], report['error'] = 'failed', f'{type(exc).__name__}: {exc}'
            exc.add_note(f'Provider record: {output / "run.json"}')
            raise
        finally:
            (output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    return output / 'OCTDL'


def prepare_original(root, dataset, *, raw_dir=None, archive=None, download=False,
                     split=None, check=False, progress=None):
    """One command: optional official download, original conversion and installation."""
    root = Path(root).resolve()
    if archive is not None and raw_dir is not None:
        raise ValueError('Supply either --archive or --raw-dir')
    if download and (archive is not None or raw_dir is not None):
        raise ValueError('Use --download alone, or supply existing provider files')
    if (download or archive is not None) and dataset != 'oct':
        raise ValueError('Automatic provider download is available only for OCTDL')
    identity = audit(root, dataset)
    existing = prepare(root, dataset, check=True, split=split)
    if check:
        if download or archive is not None:
            raise ValueError('--check does not download or unpack archives')
        required = sorted({source_spec(row, dataset)[0].as_posix()
                           for row in registered_rows(root, dataset)[0]
                           if split is None or row['split'] == split})
        absent = required if raw_dir is None else [name for name in required
                  if not (Path(raw_dir) / name).is_file()]
        return {'dataset': dataset, 'split': split, 'manifest': identity, 'installed': existing,
                'required_provider_files': required, 'missing_provider_files': absent,
                'model_inference_executed': False}
    if existing['invalid']:
        raise ValueError('Installed inputs differ from registered hashes; no files replaced')
    if not existing['missing']:
        return {'dataset': dataset, 'split': split, 'status': 'complete',
                'action': 'reused verified inputs', 'installed': existing,
                'model_inference_executed': False}
    if download:
        archive = download_oct(root)
    if archive is not None:
        raw_dir = unpack_oct(root, archive, split=split)
    if raw_dir is None:
        raise ValueError('Supply --raw-dir, or use --download/--archive for OCTDL')
    output = convert(root, dataset, raw_dir, split=split, progress=progress)
    imported = prepare(root, dataset, output / 'files', split=split)
    verified = prepare(root, dataset, check=True, split=split)
    if verified['missing'] or verified['invalid']:
        raise ValueError('Prepared inputs are incomplete or inconsistent')
    return {'dataset': dataset, 'split': split, 'status': 'complete',
            'action': 'converted and installed provider inputs',
            'conversion': str(output), 'import': imported, 'installed': verified,
            'model_inference_executed': False}

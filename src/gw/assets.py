"""Prepare registered model files without replacing existing research assets."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError(f'Unsafe asset destination: {value!r}')
    return Path(*path.parts)


def read_catalog(root):
    source = json.loads((Path(root) / 'artifacts/manifest.json').read_text())
    items = source['checkpoints']
    destinations = set()
    for key, item in items.items():
        destination = relative_path(item['destination'])
        if destination in destinations:
            raise ValueError('Duplicate asset destination')
        destinations.add(destination)
        digest = item['sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f'Invalid SHA256: {key}')
        if item.get('bytes') is not None and (not isinstance(item['bytes'], int) or item['bytes'] <= 0):
            raise ValueError(f'Invalid asset size: {key}')
    return items


def select_assets(root, experiment=None):
    items = read_catalog(root)
    selected = {key: item for key, item in items.items()
                if experiment is None or key.split('/')[0] == experiment}
    if not selected:
        raise ValueError(f'No registered assets for experiment: {experiment}')
    return selected


def verify(path, item):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if item.get('bytes') is not None and path.stat().st_size != item['bytes']:
        raise ValueError(f'Asset size mismatch: {path}')
    if file_hash(path) != item['sha256']:
        raise ValueError(f'Asset SHA256 mismatch: {path}')


def inventory(root, experiment=None):
    rows = []
    for key, item in select_assets(root, experiment).items():
        target = Path(root) / relative_path(item['destination'])
        status = 'missing'
        if target.exists() or target.is_symlink():
            try:
                verify(target, item)
                status = 'verified'
            except (FileNotFoundError, ValueError):
                status = 'invalid'
        rows.append({'id': key, 'destination': item['destination'], 'status': status,
                     'bytes': item.get('bytes')})
    return rows


def prepare(root, *, experiment=None, source_root=None, urls=None):
    """Copy from an offline asset tree or download from a versioned URL index.

    An existing valid file is reused. An existing invalid file is never replaced.
    Each new file is verified before its final destination is created.
    """
    root = Path(root).resolve()
    items = select_assets(root, experiment)
    urls = urls or {}
    plan = []
    for key, item in items.items():
        destination = root / relative_path(item['destination'])
        if destination.exists() or destination.is_symlink():
            verify(destination, item)
            plan.append((key, item, destination, None, None))
            continue
        # Do not allow the downloader to write through an external directory link.
        if not destination.parent.resolve().is_relative_to(root):
            raise ValueError('Asset parent resolves outside the project')
        source = Path(source_root) / relative_path(item['destination']) if source_root else None
        url = urls.get(key, item.get('url'))
        if source is not None:
            verify(source, item)
        elif not url:
            raise ValueError(f'No download URL or offline source registered for {key}')
        elif urllib.parse.urlsplit(url).scheme != 'https':
            raise ValueError('Asset downloads require HTTPS')
        plan.append((key, item, destination, source, url))
    records = []
    for key, item, destination, source, url in plan:
        if source is None and url is None:
            records.append({'id': key, 'status': 'reused'})
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = item.get('bytes', 0)
        if shutil.disk_usage(destination.parent).free < expected + 1024**3:
            raise RuntimeError('Insufficient space; keep 1 GiB available')
        descriptor, temporary = tempfile.mkstemp(prefix='.gw-download-', dir=destination.parent)
        temporary = Path(temporary)
        try:
            with os.fdopen(descriptor, 'wb') as output:
                if source is not None:
                    with source.open('rb') as stream:
                        shutil.copyfileobj(stream, output)
                else:
                    with urllib.request.urlopen(url, timeout=60) as stream:
                        if urllib.parse.urlsplit(stream.geturl()).scheme != 'https':
                            raise ValueError('Refusing a non-HTTPS redirect')
                        total = 0
                        while block := stream.read(1024 * 1024):
                            total += len(block)
                            if item.get('bytes') is not None and total > item['bytes']:
                                raise ValueError('Download exceeds registered asset size')
                            output.write(block)
                output.flush()
                os.fsync(output.fileno())
            verify(temporary, item)
            # Atomic, fail-if-exists publication: never replace a concurrent file.
            os.link(temporary, destination)
            records.append({'id': key, 'status': 'prepared'})
        finally:
            temporary.unlink(missing_ok=True)
    return records

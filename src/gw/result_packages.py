"""Verified, non-overwriting import of a published results archive."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.parse
import urllib.request
import zipfile
from ._files import install_verified_file

RESERVE = 8 * 1024**3


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def relative(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Invalid result path')
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ('', '.', '..') for p in value.split('/')):
        raise ValueError('Unsafe result path')
    if path.parts[0] not in ('results', 'artifacts', 'assets'):
        raise ValueError('Result package cannot replace source code or configuration')
    return Path(*path.parts)


def identity(item):
    digest = item.get('sha256', '')
    if (not isinstance(digest, str) or len(digest) != 64 or
            any(c not in '0123456789abcdef' for c in digest) or
            type(item.get('bytes')) is not int or item['bytes'] < 0):
        raise ValueError('Invalid result identity')


def verify(path, item):
    identity(item)
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != item['bytes'] or
            sha256(path) != item['sha256']):
        raise ValueError('Result identity mismatch: ' + path.name)


def download(root, item, *, reserve_bytes=RESERVE):
    name = item['file']
    if not isinstance(name, str) or Path(name).name != name or not name.endswith('.zip'):
        raise ValueError('Invalid result archive filename')
    identity(item)
    cache = root / '.cache/results'
    if not cache.resolve().is_relative_to(root):
        raise ValueError('Result cache resolves outside project')
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / name
    if target.exists() or target.is_symlink():
        verify(target, item)
        return target
    url = item.get('url')
    if not url or urllib.parse.urlsplit(url).scheme != 'https':
        raise ValueError('Provide a local archive or a fixed HTTPS download URL')
    if shutil.disk_usage(cache).free < item['bytes'] + reserve_bytes:
        raise RuntimeError('Insufficient space; keep 8 GiB free')
    descriptor, temporary = tempfile.mkstemp(prefix='.gw-results-', dir=cache)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, 'wb') as output, urllib.request.urlopen(url, timeout=60) as source:
            if urllib.parse.urlsplit(source.geturl()).scheme != 'https':
                raise ValueError('Refusing non-HTTPS redirect')
            size = 0
            while block := source.read(1024 * 1024):
                size += len(block)
                if size > item['bytes']:
                    raise ValueError('Results download exceeds registered size')
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        verify(temporary, item)
        install_verified_file(temporary, target, reserve_bytes=reserve_bytes)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def prepare(root, index, *, archive_path=None, check=False, reserve_bytes=RESERVE):
    root = Path(root).resolve()
    index = json.loads(Path(index).read_text()) if not isinstance(index, dict) else index
    item = index['results']
    identity(item)
    expected = {}
    for entry in item.get('files', []):
        identity(entry)
        path = entry['path']
        relative(path)
        if path in expected:
            raise ValueError('Duplicate result destination')
        expected[path] = entry
    if not expected:
        raise ValueError('Empty results catalog')
    for name in expected:
        if any(p.as_posix() in expected for p in PurePosixPath(name).parents):
            raise ValueError('File/directory collision')
    missing = {}
    for name, entry in expected.items():
        target = root / relative(name)
        if not target.parent.resolve().is_relative_to(root):
            raise ValueError('Result destination escapes project')
        if target.exists() or target.is_symlink():
            verify(target, entry)
        else:
            missing[name] = entry
    if check:
        return dict(verified=len(expected) - len(missing), missing=len(missing))
    if not missing:
        return dict(verified=len(expected), imported=0, reused=len(expected))
    size = sum(entry['bytes'] for entry in missing.values())
    if shutil.disk_usage(root).free < size + reserve_bytes + (item['bytes'] if archive_path is None else 0):
        raise RuntimeError('Insufficient space; keep 8 GiB free')
    archive_path = Path(archive_path) if archive_path is not None else download(root, item, reserve_bytes=reserve_bytes)
    verify(archive_path, item)
    # Validate the entire archive and all content before changing destinations.
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError('Unexpected, missing or duplicate ZIP member')
        for member in members:
            mode = stat.S_IFMT(member.external_attr >> 16)
            entry = expected[member.filename]
            if (member.is_dir() or mode not in (0, stat.S_IFREG) or member.flag_bits & 1 or
                    member.file_size != entry['bytes']):
                raise ValueError('Unsafe ZIP member or unexpected member size')
            digest = hashlib.sha256()
            with archive.open(member) as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != entry['sha256']:
                raise ValueError('Archived result content differs from catalog')
        for name, entry in missing.items():
            target = root / relative(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.parent.resolve().is_relative_to(root):
                raise ValueError('Result destination escapes project')
            if shutil.disk_usage(root).free < entry['bytes'] + reserve_bytes:
                raise RuntimeError('Insufficient space during import')
            descriptor, temporary = tempfile.mkstemp(prefix='.gw-result-', dir=target.parent)
            temporary = Path(temporary)
            try:
                with os.fdopen(descriptor, 'wb') as output, archive.open(name) as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                verify(temporary, entry)
                install_verified_file(temporary, target, reserve_bytes=reserve_bytes)
            finally:
                temporary.unlink(missing_ok=True)
    return dict(verified=len(expected), imported=len(missing), reused=len(expected) - len(missing))

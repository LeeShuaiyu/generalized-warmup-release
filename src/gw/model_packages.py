"""Import only registered model files from verified archive parts."""
from pathlib import Path
import json
import os
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request

from .assets import file_hash, read_catalog, relative_path, select_assets, verify
from ._files import install_verified_file


def load_index(root, index):
    index = json.loads(Path(index).read_text()) if not isinstance(index,dict) else index
    if index.get('status') != 'complete' or not index.get('parts'):
        raise ValueError('Model package index is incomplete')
    registered = {item['destination']:item for item in read_catalog(root).values()}
    names, destinations = set(),set()
    for part in index['parts']:
        name = part['file']
        if relative_path(name).name != name or not name.endswith('.tar') or name in names:
            raise ValueError('Unsafe or duplicate model part filename')
        names.add(name)
        if type(part.get('bytes')) is not int or part['bytes'] <= 0:
            raise ValueError('Invalid model part size')
        digest = part.get('sha256','')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid model part SHA256')
        if not part.get('models'):
            raise ValueError('Empty model part')
        for item in part['models']:
            destination = item['destination']
            relative_path(destination)
            if destination in destinations or destination not in registered:
                raise ValueError('Duplicate or unregistered packaged model')
            destinations.add(destination)
            original = registered[destination]
            if item.get('sha256') != original['sha256'] or item.get('bytes') != original.get('bytes'):
                raise ValueError('Packaged model identity differs from the asset catalog')
    return index


def _part_file(cache, part, url, *, reserve_bytes=8*1024**3):
    target = cache/part['file']
    if target.exists() or target.is_symlink():
        verify(target,part)
        return target
    if not url:
        raise ValueError(f'No model package download URL or local file for {part["file"]}')
    if urllib.parse.urlsplit(url).scheme != 'https':
        raise ValueError('Model package downloads require HTTPS')
    cache.mkdir(parents=True,exist_ok=True)
    descriptor,temporary = tempfile.mkstemp(prefix='.gw-part-',dir=cache)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor,'wb') as output, urllib.request.urlopen(url,timeout=60) as source:
            if urllib.parse.urlsplit(source.geturl()).scheme != 'https':
                raise ValueError('Refusing a non-HTTPS redirect')
            size = 0
            while block := source.read(1024*1024):
                size += len(block)
                if size > part['bytes']:
                    raise ValueError('Model package exceeds its registered size')
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        verify(temporary,part)
        install_verified_file(temporary,target,reserve_bytes=reserve_bytes)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def _check_members(path,part):
    expected = {item['destination']:item for item in part['models']}
    seen = set()
    with tarfile.open(path,'r:') as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError('Unexpected or duplicate archive member')
            if member.size != expected[member.name]['bytes']:
                raise ValueError('Archived model size differs from the asset catalog')
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('Model archive is incomplete')


def prepare(root, index, *, experiment=None, packages_dir=None, urls=None, reserve_bytes=8*1024**3):
    """Verify parts and prepare one experiment with exclusive file creation.

    The asset catalog determines model destinations and expected bytes.
    HTTPS URLs must be supplied explicitly or in the index.
    """
    root = Path(root).resolve()
    index = load_index(root,index)
    items = select_assets(root,experiment)
    missing,records = {},[]
    for key,item in items.items():
        target = root/relative_path(item['destination'])
        if target.exists() or target.is_symlink():
            verify(target,item)
            records.append(dict(id=key,status='reused'))
        else:
            if not target.parent.resolve().is_relative_to(root):
                raise ValueError('Model destination parent resolves outside the project')
            missing[item['destination']] = (key,item)
    if not missing:
        return records
    needed = [part for part in index['parts']
              if any(item['destination'] in missing for item in part['models'])]
    covered = {item['destination'] for part in needed for item in part['models']}
    if not set(missing).issubset(covered):
        raise ValueError('Required models are absent from the package index')
    cache = Path(packages_dir).resolve() if packages_dir else root/'.cache/model-packages'
    if packages_dir is None and not cache.resolve().is_relative_to(root):
        raise ValueError('Package cache resolves outside the project')
    urls = urls or {}
    download_bytes = sum(part['bytes'] for part in needed if not (cache/part['file']).exists())
    model_bytes = sum(item['bytes'] for _,item in missing.values())
    if shutil.disk_usage(root).free < model_bytes+reserve_bytes+(download_bytes if cache.is_relative_to(root) else 0):
        raise RuntimeError('Insufficient space; keep the requested disk reserve')
    cache_ancestor = cache
    while not cache_ancestor.exists():
        cache_ancestor = cache_ancestor.parent
    if download_bytes and shutil.disk_usage(cache_ancestor).free < download_bytes+reserve_bytes:
        raise RuntimeError('Insufficient package-cache space; keep the requested disk reserve')
    for part in needed:
        target = cache/part['file']
        if target.exists() or target.is_symlink():
            verify(target,part)
        elif not urls.get(part['file'],part.get('url')):
            raise ValueError(f'No model package download URL or local file for {part["file"]}')
        elif urllib.parse.urlsplit(urls.get(part['file'],part.get('url'))).scheme != 'https':
            raise ValueError('Model package downloads require HTTPS')
    prepared_parts = []
    for part in needed:
        archive_path = _part_file(cache,part,urls.get(part['file'],part.get('url')),
                                  reserve_bytes=reserve_bytes)
        _check_members(archive_path,part)
        prepared_parts.append((part,archive_path))
    for part,archive_path in prepared_parts:
        expected = {item['destination']:item for item in part['models']}
        seen = set()
        with tarfile.open(archive_path,'r|') as archive:
            for member in archive:
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError('Unexpected or duplicate archive member')
                seen.add(member.name)
                if member.size != expected[member.name]['bytes']:
                    raise ValueError('Archived model size differs from the asset catalog')
                if member.name not in missing:
                    continue
                key,item = missing[member.name]
                target = root/relative_path(member.name)
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.parent.resolve().is_relative_to(root):
                    raise ValueError('Model destination parent resolves outside the project')
                if shutil.disk_usage(root).free < item['bytes']+reserve_bytes:
                    raise RuntimeError('Insufficient space; keep the requested disk reserve')
                descriptor,temporary = tempfile.mkstemp(prefix='.gw-model-',dir=target.parent)
                temporary = Path(temporary)
                try:
                    with os.fdopen(descriptor,'wb') as output, archive.extractfile(member) as source:
                        shutil.copyfileobj(source,output)
                        output.flush()
                        os.fsync(output.fileno())
                    verify(temporary,item)
                    install_verified_file(temporary,target,reserve_bytes=reserve_bytes)
                    records.append(dict(id=key,status='prepared_from_package',package=part['file']))
                finally:
                    temporary.unlink(missing_ok=True)
        if seen != set(expected):
            raise ValueError('Model archive is incomplete')
    return records

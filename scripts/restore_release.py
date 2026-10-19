"""Restore verified code/results packages into a new project (standard library only).

Optional model preparation uses the restored project's registered model catalog.
Original image datasets are obtained separately under their provider's terms.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import zipfile

RESERVE_BYTES = 8 * 1024**3
RECEIPT = '.gw-restore.json'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_path(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Invalid package path')
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ('', '.', '..') for part in value.split('/')):
        raise ValueError('Unsafe package path')
    return Path(*path.parts)


def identity(item):
    digest = item.get('sha256', '')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid package SHA256')
    if type(item.get('bytes')) is not int or item['bytes'] < 0:
        raise ValueError('Invalid package byte count')


def check_file(path, item):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Package must be a regular local file: {path.name}')
    if path.stat().st_size != item['bytes'] or file_hash(path) != item['sha256']:
        raise ValueError(f'Package identity mismatch: {path.name}')


def preflight(index, packages_dir):
    """Check both archives and every member before creating any project files."""
    if index.get('status') not in ('private_candidate', 'complete'):
        raise ValueError('Package index is incomplete')
    packages = index.get('packages', {})
    if set(packages) != {'code', 'results'}:
        raise ValueError('Code and results packages are required')
    all_paths, package_names, plan = set(), set(), []
    for kind in ('code', 'results'):
        package = packages[kind]
        identity(package)
        name = package['file']
        if safe_path(name).name != name or not name.endswith('.zip') or name in package_names:
            raise ValueError('Unsafe or duplicate package filename')
        package_names.add(name)
        entries = package.get('files', [])
        if not isinstance(entries, list) or not entries:
            raise ValueError('Empty package file index')
        expected = {}
        for item in entries:
            identity(item)
            path = item['path']
            safe_path(path)
            if path == RECEIPT or path in expected or path in all_paths:
                raise ValueError('Duplicate, reserved or overlapping package path')
            expected[path] = item
            all_paths.add(path)
        archive_path = packages_dir / name
        check_file(archive_path, package)
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)) or set(names) != set(expected):
                raise ValueError('Unexpected, missing or duplicate ZIP member')
            for member in members:
                mode = stat.S_IFMT(member.external_attr >> 16)
                if member.is_dir() or mode not in (0, stat.S_IFREG) or member.flag_bits & 1:
                    raise ValueError('ZIP links, special files, directories and encryption are not accepted')
                if member.file_size != expected[member.filename]['bytes']:
                    raise ValueError('ZIP member size differs from index')
        plan.append((kind, archive_path, expected))
    # Reject file/directory collisions even across the two separate archives.
    for value in all_paths:
        for parent in PurePosixPath(value).parents:
            if parent.as_posix() in all_paths:
                raise ValueError('Package file/directory collision')
    return plan


def restore(index_path, packages_dir, destination, *, index_sha256=None, reserve_bytes=RESERVE_BYTES):
    index_path, packages_dir, destination = Path(index_path), Path(packages_dir), Path(destination)
    digest = file_hash(index_path)
    if index_sha256 is not None and digest != index_sha256:
        raise ValueError('Package index SHA256 mismatch')
    index = json.loads(index_path.read_text())
    plan = preflight(index, packages_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Choose a new project directory; existing projects are not replaced')
    if not destination.parent.is_dir():
        raise FileNotFoundError('Project parent directory must already exist')
    total_bytes = sum(item['bytes'] for _, _, entries in plan for item in entries.values())
    if shutil.disk_usage(destination.parent).free < total_bytes + reserve_bytes:
        raise RuntimeError('Insufficient space; keep 8 GiB free after restoration')
    destination.mkdir(exist_ok=False)
    destination = destination.resolve()
    receipt = dict(status='restoring', index_sha256=digest, files=0, bytes=0,
                   model_weights_prepared=False, raw_image_datasets_prepared=False,
                   checkpoint_inference_executed=False)
    try:
        for kind, archive_path, expected in plan:
            with zipfile.ZipFile(archive_path) as archive:
                for name, item in expected.items():
                    if shutil.disk_usage(destination).free < item['bytes'] + reserve_bytes:
                        raise RuntimeError('Insufficient space; keep the requested reserve')
                    target = destination / safe_path(name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.parent.resolve().is_relative_to(destination):
                        raise ValueError('Package target escapes the new project')
                    digest = hashlib.sha256()
                    written = 0
                    with archive.open(name) as source, target.open('xb') as output:
                        while block := source.read(1024 * 1024):
                            written += len(block)
                            if written > item['bytes']:
                                raise ValueError('ZIP member exceeds its registered size')
                            digest.update(block)
                            output.write(block)
                    if written != item['bytes'] or digest.hexdigest() != item['sha256']:
                        raise ValueError(f'Restored file identity mismatch: {name}')
                    receipt['files'] += 1
                    receipt['bytes'] += written
            receipt[kind] = dict(file=archive_path.name, sha256=file_hash(archive_path))
        receipt['status'] = 'restored'
    except Exception as error:
        receipt.update(status='failed', error=str(error))
        # Keep a failed new directory and its receipt for diagnosis, never remove data.
        (destination / RECEIPT).write_text(json.dumps(receipt, indent=2) + '\n')
        raise
    (destination / RECEIPT).write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True, type=Path)
    parser.add_argument('--index-sha256', help='Expected SHA256 from the trusted handoff record')
    parser.add_argument('--packages-dir', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path, help='New project directory')
    parser.add_argument('--models-dir', type=Path, help='Optional local directory of verified model TAR parts')
    parser.add_argument('--experiment', help='Only prepare models for this manifest experiment')
    args = parser.parse_args()
    if args.experiment and not args.models_dir:
        parser.error('--experiment requires --models-dir')
    receipt = restore(args.index, args.packages_dir, args.destination, index_sha256=args.index_sha256)
    if args.models_dir:
        root = args.destination.resolve()
        sys.path.insert(0, str(root / 'src'))
        from gw.model_packages import prepare
        try:
            records = prepare(root, root / 'artifacts/model_packages.json',
                              experiment=args.experiment, packages_dir=args.models_dir)
        except Exception as error:
            receipt.update(model_preparation_status='failed', model_preparation_error=str(error))
            (root / RECEIPT).write_text(json.dumps(receipt, indent=2) + '\n')
            raise
        receipt.update(model_weights_prepared=True, model_preparation_status='verified',
                       experiment=args.experiment, verified_models=len(records))
        (root / RECEIPT).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

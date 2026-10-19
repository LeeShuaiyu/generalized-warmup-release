"""Read-only download sizing by experiment; no network, extraction or file changes."""
import json
from pathlib import Path
from .assets import read_catalog, select_assets
from .model_packages import load_index


def plan(root, experiment=None, *, results_only=False, include_results=False):
    root = Path(root).resolve()
    catalog = json.loads((root / 'artifacts/downloads.json').read_text())
    files = []
    model_bytes = model_count = 0
    if not results_only:
        if experiment is None:
            raise ValueError('Select one experiment or results-only explicitly')
        selected = select_assets(root, experiment)
        destinations = {row['destination'] for row in selected.values()}
        packages = load_index(root, root / 'artifacts/model_packages.json')
        urls = json.loads((root / 'artifacts/model_urls.json').read_text())
        needed = [row for row in packages['parts']
                  if any(item['destination'] in destinations for item in row['models'])]
        covered = {item['destination'] for row in needed for item in row['models']}
        if not destinations.issubset(covered):
            raise ValueError('Package index does not cover the selected experiment')
        files.extend(dict(file=row['file'], bytes=row['bytes'], sha256=row['sha256'],
                          url=urls.get(row['file'])) for row in needed)
        model_count = len(selected)
        model_bytes = sum(row['bytes'] for row in selected.values())
    if results_only or include_results:
        row = catalog['results']
        files.insert(0, {key: row.get(key) for key in ('file', 'bytes', 'sha256', 'url')})
    return dict(experiment=experiment, results_only=results_only,
                hosting=catalog.get('hosting', {}), files=files,
                archive_bytes=sum(row['bytes'] for row in files),
                selected_models=model_count, selected_model_bytes=model_bytes,
                disk_reserve_bytes=8 * 1024**3,
                notes='Full required archive parts; release parts are separated by experiment. '
                      'Sizes assume an empty cache. Existing verified files are reused by preparation.')


def experiments(root):
    return sorted({key.split('/')[0] for key in read_catalog(root)})

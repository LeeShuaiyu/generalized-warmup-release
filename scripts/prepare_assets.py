"""Check or prepare the model files used by one experiment."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gw.assets import inventory, prepare
from gw.model_packages import prepare as prepare_packages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', help='Manifest experiment ID, e.g. random or s4_family')
    parser.add_argument('--from-dir', type=Path, help='Offline asset tree with the same relative paths')
    parser.add_argument('--url-index', type=Path, help='JSON mapping asset IDs or package filenames to HTTPS URLs')
    parser.add_argument('--package-index', type=Path, help='Completed model archive index')
    parser.add_argument('--packages-dir', type=Path, help='Directory containing the model archive parts')
    parser.add_argument('--check', action='store_true', help='Verify without copying or downloading')
    args = parser.parse_args()
    if args.package_index and args.from_dir:
        parser.error('--from-dir and --package-index are alternative import modes')
    if args.packages_dir and not args.package_index:
        parser.error('--packages-dir requires --package-index')
    if args.check:
        rows = inventory(ROOT, args.experiment)
        counts = {status: sum(r['status'] == status for r in rows)
                  for status in ['verified', 'missing', 'invalid']}
        print(json.dumps(counts))
        for row in rows:
            if row['status'] != 'verified':
                print(f"{row['status']}: {row['id']}")
        return 0 if counts['verified'] == len(rows) else 2
    urls = json.loads(args.url_index.read_text()) if args.url_index else None
    if args.package_index:
        rows = prepare_packages(ROOT,args.package_index,experiment=args.experiment,
                                packages_dir=args.packages_dir,urls=urls)
    else:
        rows = prepare(ROOT, experiment=args.experiment, source_root=args.from_dir, urls=urls)
    print(f"{len(rows)} model files verified and available")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

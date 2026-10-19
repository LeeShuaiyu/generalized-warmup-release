"""Validate or import the fixed processed sample files without changing splits."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gw.data_assets import prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['sar', 'opensar', 'oct', 'fusar'])
    parser.add_argument('--from-dir', type=Path, help='Processed source tree, as described in data/README.md')
    parser.add_argument('--split', choices=['train', 'val', 'test'])
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    counts = prepare(ROOT, args.dataset, args.from_dir, check=args.check, split=args.split)
    print(json.dumps(counts))
    return 2 if counts['missing'] or counts['invalid'] else 0


if __name__ == '__main__':
    raise SystemExit(main())

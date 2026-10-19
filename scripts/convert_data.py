"""Convert original files while keeping the registered sample lists and splits."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gw.raw_data import audit, convert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['sar', 'opensar', 'oct'])
    parser.add_argument('--raw-dir', type=Path, help='SAR json/, OpenSARShip2 ZIP directory, or OCTDL class directory')
    parser.add_argument('--split', choices=['train', 'val', 'test'])
    parser.add_argument('--limit', type=int, help='Diagnostic subset only; first registered samples')
    parser.add_argument('--audit', action='store_true', help='Check the complete sample identities and fixed splits')
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit(ROOT, args.dataset), indent=2))
        return
    if args.raw_dir is None:
        parser.error('--raw-dir is required for conversion')
    output = convert(ROOT, args.dataset, args.raw_dir, split=args.split, limit=args.limit,
                     progress=lambda done, total: print(f'{done}/{total}', flush=True))
    print(json.dumps({'output': str(output), 'status': 'complete'}))


if __name__ == '__main__':
    main()

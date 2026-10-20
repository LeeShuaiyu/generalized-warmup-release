"""Rebuild homogeneous-region banks from fixed training images."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gw.region_selection import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['sar', 'opensar', 'fusar', 'oct'], required=True)
    parser.add_argument('--source-dir', type=Path, help='Registered processed tree, or extracted FUSAR class folders')
    parser.add_argument('--limit', type=int, help='Number of training images in a diagnostic subset')
    args = parser.parse_args()
    print(run(ROOT, args.dataset, source_dir=args.source_dir, limit=args.limit))


if __name__ == '__main__':
    main()

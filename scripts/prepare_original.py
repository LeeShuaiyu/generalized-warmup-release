"""Download or convert original data and prepare the fixed evaluation inputs."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gw.provider_data import prepare_original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['sar', 'opensar', 'oct'])
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--raw-dir', type=Path, help='Extracted original provider directory')
    sources.add_argument('--archive', type=Path, help='Existing official OCTDL.zip')
    sources.add_argument('--download', action='store_true', help='Download pinned public OCTDL v2')
    parser.add_argument('--split', choices=['train', 'val', 'test'])
    parser.add_argument('--check', action='store_true', help='Inspect installed inputs and required filenames only')
    args = parser.parse_args()
    result = prepare_original(ROOT, args.dataset, raw_dir=args.raw_dir, archive=args.archive,
                              download=args.download, split=args.split, check=args.check,
                              progress=lambda done, total: print(f'{done}/{total}', flush=True))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

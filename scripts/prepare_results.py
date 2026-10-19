"""Download/import the paper's results archive after verifying its catalog."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gw.result_packages import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=ROOT / 'artifacts/downloads.json')
    parser.add_argument('--archive', type=Path, help='Use an already downloaded results ZIP')
    parser.add_argument('--check', action='store_true', help='Check files without downloading or writing')
    args = parser.parse_args()
    result = prepare(ROOT, args.index, archive_path=args.archive, check=args.check)
    print(json.dumps(result))
    return 2 if result.get('missing', 0) else 0


if __name__ == '__main__':
    raise SystemExit(main())

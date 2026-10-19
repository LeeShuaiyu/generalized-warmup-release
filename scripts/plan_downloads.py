"""List the precise archive files and size for one reproduction without downloading."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gw.download_plan import experiments, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--experiment', choices=experiments(ROOT))
    choice.add_argument('--results-only', action='store_true')
    parser.add_argument('--include-results', action='store_true')
    args = parser.parse_args()
    result = plan(ROOT, args.experiment, results_only=args.results_only,
                  include_results=args.include_results)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

"""Evaluate ImageNet-initialized final checkpoints, optionally one complete dataset."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gw.classification_inference import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--dataset', help='Evaluate every method and seed for this dataset')
    parser.add_argument('--run-ids', help='Comma-separated configured run identities')
    parser.add_argument('--no-report', action='store_true', help='Write predictions without notebook dependencies')
    args = parser.parse_args()
    config = json.loads((ROOT/'configs/pretrained.json').read_text())
    if args.dataset:
        selected = [r for r in config['runs'] if r['dataset'] == args.dataset]
        if not selected:
            parser.error('Unknown dataset')
        config = dict(config,runs=selected,row_order=[r for r in config['row_order'] if r[0]==args.dataset])
    directory = run(ROOT, config, device=args.device,
                    run_ids=args.run_ids.split(',') if args.run_ids else None)
    if args.no_report:
        print(directory,flush=True)
        return
    if args.run_ids:
        print(directory,flush=True)
        return
    from gw import classification_report as report
    _, mean, sd = report.calculate(ROOT, config, directory)
    for name in ['main_pretrained', 's5_pretrained']:
        reference = report.paper(ROOT, name)
        reference = report.select_reference(reference, config)
        comparison = report.comparison(mean, sd, reference)
        comparison.to_csv(directory / f'{name}_comparison.csv')
    print(directory, flush=True)


if __name__ == '__main__':
    main()

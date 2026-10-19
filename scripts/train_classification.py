"""Run one listed configuration; keep training separate from notebook display."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from gw.classification_training import train, resolve_training_run

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', choices=['random', 'pretrained', 'training_compat'], required=True)
parser.add_argument('--run-id', required=True)
parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
parser.add_argument('--evaluate-test', action='store_true')
args = parser.parse_args()
config = json.loads((ROOT/f'configs/{args.experiment}.json').read_text())
selected = [r for r in config['runs'] if r['id']==args.run_id]
if len(selected) != 1:
    parser.error('Select one run ID from the experiment configuration')
try:
    run = resolve_training_run(ROOT, args.experiment, selected[0])
except ValueError as exc:
    parser.error(str(exc))
print(train(ROOT, args.experiment, run, device=args.device, evaluate_test=args.evaluate_test), flush=True)

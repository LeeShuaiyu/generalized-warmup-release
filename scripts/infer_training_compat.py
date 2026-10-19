"""Evaluate fixed training-calibration weights; missing mappings are explicit."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gw.assets import file_hash
from gw.training_compat_inference import run

parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
parser.add_argument('--runs', nargs='+')
parser.add_argument('--available-only', action='store_true')
args = parser.parse_args()
path = ROOT / 'configs/training_compat.json'
manifest = json.loads((ROOT / 'artifacts/manifest.json').read_text())
if file_hash(path) != manifest['source_files']['configs/training_compat.json']['sha256']:
    raise ValueError('Configuration identity changed')
config = json.loads(path.read_text())
print(run(ROOT, config, device=args.device, run_ids=args.runs, available_only=args.available_only), flush=True)

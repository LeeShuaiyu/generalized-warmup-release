"""Run one registered initialization-bias toy condition in a new directory."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gw.toy_training import train

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, choices=range(100, 130), default=100)
parser.add_argument('--steps', type=int, help='Explicit shorter check budget; default is 12000')
args = parser.parse_args()
print(train(ROOT, args.seed, steps=args.steps), flush=True)

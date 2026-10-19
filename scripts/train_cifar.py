"""Train one registered CIFAR model without evaluating the test split."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gw import cifar_training

parser = argparse.ArgumentParser()
parser.add_argument('--run',required=True,help='Run ID in configs/cifar_training.json')
parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
parser.add_argument('--epochs',type=int)
parser.add_argument('--warmup-steps',type=int)
parser.add_argument('--train-limit',type=int)
parser.add_argument('--validation-limit',type=int)
args = parser.parse_args()
config = json.loads((ROOT/'configs/cifar_training.json').read_text())
record = next((r for r in config['runs'] if r['id'] == args.run),None)
if record is None:
    raise ValueError('Unknown run ID')
print(cifar_training.run(ROOT,record,args.device,args.epochs,args.warmup_steps,args.train_limit,args.validation_limit))

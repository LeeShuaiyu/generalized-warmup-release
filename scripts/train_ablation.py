"""Train one family or duration condition on its fixed training/validation split."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gw.ablation_training import train

parser=argparse.ArgumentParser()
parser.add_argument('--experiment',choices=['s4_family','s7_length'],required=True)
parser.add_argument('--run-id',required=True)
parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
args=parser.parse_args()
config=json.loads((ROOT/'configs/ablation_training.json').read_text())
selected=[r for r in config['runs'] if r['experiment']==args.experiment and r['id']==args.run_id]
if len(selected)!=1:parser.error('Select one run ID from configs/ablation_training.json')
print(train(ROOT,selected[0],device=args.device),flush=True)

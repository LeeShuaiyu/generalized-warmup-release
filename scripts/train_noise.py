"""Train one registered EuroSAT configuration without evaluating the test split."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gw.noise_training import run
from gw.presentation import sha256

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--epochs',type=int)
    p.add_argument('--warmup-steps',type=int)
    p.add_argument('--train-limit',type=int)
    p.add_argument('--validation-limit',type=int)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    path=ROOT/'configs/noise.json'
    identity=json.loads((ROOT/'artifacts/manifest.json').read_text())['source_files']['configs/noise.json']
    if sha256(path)!=identity['sha256']:
        raise ValueError('Noise configuration identity differs')
    records=[r for r in json.loads(path.read_text())['runs'] if r['id']==args.run]
    if len(records)!=1:
        raise ValueError('Choose exactly one registered run')
    print(run(ROOT,records[0],device=args.device,epochs=args.epochs,warmup_steps=args.warmup_steps,
              train_limit=args.train_limit,validation_limit=args.validation_limit,workers=args.workers))

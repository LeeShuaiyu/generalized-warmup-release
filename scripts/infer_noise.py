"""Run model inference for the EuroSAT target-noise control table."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gw.noise_inference import run
from gw.presentation import sha256

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--runs',nargs='+')
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    path=ROOT/'configs/noise.json'
    identity=json.loads((ROOT/'artifacts/manifest.json').read_text())['source_files']['configs/noise.json']
    if sha256(path)!=identity['sha256']:
        raise ValueError('Noise configuration identity differs')
    print(run(ROOT,json.loads(path.read_text()),device=args.device,run_ids=args.runs,workers=args.workers))

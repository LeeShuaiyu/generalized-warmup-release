"""Download or verify official EuroSAT RGB files for the fixed experiment split."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gw.noise_inference import dataset

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download',action='store_true')
    args=p.parse_args()
    c=dataset(ROOT,download=args.download)
    print(f'EuroSAT verified: {len(c["base"])} RGB images; {len(c["train_indices"])} train, '
          f'{len(c["val_indices"])} validation, {len(c["test_indices"])} test')

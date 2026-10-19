"""Check CIFAR source files; download the official archive only when requested."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gw.cifar_inference import dataset

parser = argparse.ArgumentParser()
parser.add_argument('--download',action='store_true')
args = parser.parse_args()
for train in [False,True]:
    images = dataset(ROOT,train=train,download=args.download)
    print(f'CIFAR-100 {"train" if train else "test"}: {len(images)} images verified')

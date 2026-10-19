"""Run CIFAR inference and print the corresponding paper table."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gw import classification_report as report, cifar_inference

parser = argparse.ArgumentParser()
parser.add_argument('--table',choices=['cifar3','cifar10'],default='cifar10')
parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
args = parser.parse_args()
config = report.read(ROOT,args.table)
directory = cifar_inference.run(ROOT,config,args.device)
_,mean,sd = report.calculate(ROOT,config,directory)
reference = report.paper(ROOT,'s2_'+args.table)
print(report.table_frame(mean,sd,reference).to_string())
print(f'Output: {directory}')

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gw import classification_report as report
from gw.classification_inference import run

parser = argparse.ArgumentParser()
parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
args = parser.parse_args()
config = report.read(ROOT,'random')
directory = run(ROOT,config,device=args.device)
runs,mean,sd = report.calculate(ROOT,config,directory)
for table in ['main_random','s4_main']:
    comparison = report.comparison(mean,sd,report.paper(ROOT,table))
    comparison.to_csv(directory/f'{table}_comparison.csv')
print(directory,flush=True)

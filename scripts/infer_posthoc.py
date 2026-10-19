"""Evaluate six checkpoints and refit validation-only TS/VS."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gw.posthoc_inference import run
from gw.posthoc import summarize
from gw.classification_report import comparison, paper, table_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--validation-output',type=Path)
    parser.add_argument('--test-output',type=Path)
    args = parser.parse_args()
    directory = run(ROOT,args.device,validation_directory=args.validation_output,test_directory=args.test_output)
    _,mean,sd = summarize(ROOT,directory)
    reference = paper(ROOT,'s6_posthoc')
    print(table_frame(mean,sd,reference).to_string())
    print(f'Output: {directory}')

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from gw.distribution_fitting import run, summarize

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', action='append', choices=['opensar','sar','fusar','oct'])
parser.add_argument('--family', action='append', choices=['Gaussian','Gamma','Weibull','Lognormal','Rayleigh','Nakagami','GeneralizedGamma','Uniform','Laplace'])
parser.add_argument('--region-run', action='append', default=[], metavar='DATASET=DIRECTORY',
                    help='Use a completed full bank rebuilt by select_regions.py')
args = parser.parse_args()
region_runs = {}
for value in args.region_run:
    dataset, separator, directory = value.partition('=')
    if not separator or not directory or dataset in region_runs:
        parser.error('Supply each region run once as DATASET=DIRECTORY')
    region_runs[dataset] = Path(directory)
output = run(ROOT, datasets=args.dataset, families=args.family, region_runs=region_runs)
_, table = summarize(output)
table.to_csv(output/'summary.csv')
print(table.round(6).to_string())
print(output, flush=True)

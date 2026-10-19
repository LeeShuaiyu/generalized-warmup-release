"""Refit TS and VS on the registered validation logits, then evaluate test logits."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from gw import posthoc, posthoc_inference
from gw.classification_report import paper, comparison, table_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['models', 'archived'], default='models')
    parser.add_argument('--directory', type=Path,
                        default=ROOT/'results/posthoc_compat/model_refit_fp32')
    args = parser.parse_args()
    directory = (posthoc.run(ROOT) if args.source == 'archived'
                 else posthoc_inference.refit(ROOT, args.directory))
    runs, mean, sd = posthoc.summarize(ROOT,directory)
    reference = paper(ROOT,'s6_posthoc')
    print(table_frame(mean,sd,reference).to_string())
    print(f'Output: {directory}')

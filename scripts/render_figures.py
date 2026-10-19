"""Export manuscript references or regenerate supported quantitative panels."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from gw.submitted_figures import run
from gw.paper_figures import run as run_component

parser = argparse.ArgumentParser(description=__doc__)
selection=parser.add_mutually_exclusive_group(required=True)
selection.add_argument('--figure', type=int, choices=range(1, 5), help='Submitted figure number (1–4)')
selection.add_argument('--component', type=int, choices=[2,3,4,6], help='Historical quantitative generator ID')
parser.add_argument('--mode', choices=['paper', 'quantitative'], default='paper')
arguments = parser.parse_args()
print(run_component(ROOT, arguments.component, 'quantitative') if arguments.component is not None else run(ROOT, arguments.figure, arguments.mode))

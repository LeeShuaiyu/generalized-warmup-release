"""Re-evaluate the five-seed OpenSARShip2 comparison."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gw.inference import run_family
from gw.presentation import load_predictions, paper_comparison, read_experiment, summarize

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
args = parser.parse_args()
config = read_experiment(ROOT, "s4_family")
output = run_family(ROOT, config, device=args.device)
runs = load_predictions(ROOT, config, output)
mean, std = summarize(runs, config)
mean.to_csv(output / "summary.csv")
comparison = paper_comparison(ROOT, config, mean)
comparison.to_csv(output / "paper_comparison.csv", index=False)
print(mean.round(6).to_string())
print(f"Completed runs: {len(runs)}")
print(f"Results: {output}")

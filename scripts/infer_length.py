"""Re-evaluate every warm-up duration and paired seed."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gw.inference import run_experiment
from gw.presentation import read_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    output = run_experiment(ROOT, read_experiment(ROOT, "s7_length"), device=args.device)
    print(f"Predictions: {output}")

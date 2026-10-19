"""Checks for the result table and the portable input manifest."""
from pathlib import Path
import csv
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gw import presentation as report
from gw.inference import split_hash


class FamilyResults(unittest.TestCase):
    def setUp(self):
        self.config = report.read_experiment(ROOT, "s4_family")

    def test_complete_runs(self):
        runs = report.load_predictions(ROOT, self.config)
        self.assertEqual(len(runs), 20)
        self.assertEqual(runs.groupby("Method").size().tolist(), [5, 5, 5, 5])
        for _, group in runs.groupby("Method"):
            self.assertEqual(set(group.Seed), set(self.config["seeds"]))

    def test_paper_values(self):
        runs = report.load_predictions(ROOT, self.config)
        mean, _ = report.summarize(runs, self.config)
        comparison = report.paper_comparison(ROOT, self.config, mean)
        self.assertEqual(len(comparison), 20)
        self.assertTrue(comparison["Matches displayed precision"].all())

    def test_fixed_split(self):
        with (ROOT / "data/manifests/opensarship2.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(split_hash(rows), self.config["manifest_sha256"])
        self.assertEqual(len({row["sample_id"] for row in rows}), len(rows))
        self.assertEqual(sum(row["split"] == "val" for row in rows), 954)
        self.assertTrue(all(not row["raw_path"].startswith("/root/") for row in rows))

    def test_checkpoint_identities(self):
        manifest = json.loads((ROOT / "artifacts/manifest.json").read_text())
        checkpoints = {key: row for key, row in manifest["checkpoints"].items() if key.startswith("s4_family/")}
        self.assertEqual(len(checkpoints), 20)
        self.assertTrue(all(len(row["sha256"]) == 64 for row in checkpoints.values()))

    def test_cuda_results(self):
        directory = ROOT / "results/s4_family/gpu_predictions"
        record = json.loads((directory / "run.json").read_text())
        self.assertEqual(len(record["records"]), 20)
        self.assertEqual({row["device"] for row in record["records"]}, {"cuda"})
        runs = report.load_predictions(ROOT, self.config, directory)
        mean, _ = report.summarize(runs, self.config)
        self.assertTrue(report.paper_comparison(ROOT, self.config, mean)["Matches displayed precision"].all())


if __name__ == "__main__":
    unittest.main()

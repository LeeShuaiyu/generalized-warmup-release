"""Completeness and recomputation checks for the migrated experiments."""
from pathlib import Path
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gw import duration, direction_report, presentation


class MigratedResults(unittest.TestCase):
    def test_duration_completeness(self):
        config = presentation.read_experiment(ROOT, "s7_length")
        runs, summary = duration.summarize(ROOT, config)
        self.assertEqual(len(runs), 33)
        self.assertEqual(list(summary.index), list(range(11)))
        self.assertTrue((summary["count"] == 3).all())
        self.assertEqual(summary["mean"].idxmin(), 4)

    def test_duration_cuda(self):
        config = presentation.read_experiment(ROOT, "s7_length")
        folder = ROOT / "results/s7_length/gpu_predictions"
        record = json.loads((folder / "run.json").read_text())
        self.assertEqual(len(record["records"]), 33)
        self.assertEqual({row["device"] for row in record["records"]}, {"cuda"})
        runs, summary = duration.summarize(ROOT, config, folder)
        self.assertEqual(len(runs), 33)
        self.assertEqual(summary["mean"].idxmin(), 4)

    def test_direction_bootstrap(self):
        runs, summary = direction_report.recompute(ROOT)
        self.assertEqual(len(runs), 100)
        self.assertTrue(direction_report.comparison(ROOT, summary)["Match"].all())
        self.assertEqual(summary["Lower response"].tolist(), [100, 97])

    def test_saved_notebooks(self):
        import nbformat
        for name in ["s4_family", "s7_length", "s2_direction"]:
            notebook = nbformat.read(ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
            nbformat.validate(notebook)
            errors = [output for cell in notebook.cells for output in cell.get("outputs", []) if output.get("output_type") == "error"]
            self.assertEqual(errors, [])
            self.assertTrue(any("image/png" in output.get("data", {}) for cell in notebook.cells for output in cell.get("outputs", [])))


if __name__ == "__main__":
    unittest.main()

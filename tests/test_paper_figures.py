from pathlib import Path
import json
import sys
import unittest

import nbformat
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from gw import paper_figures as figures
from gw.supporting_reports import checked


class PaperFigures(unittest.TestCase):
    def test_manuscript_identity(self):
        records = figures.catalog(ROOT)
        self.assertEqual(len(records), 6)
        self.assertEqual(sum(len(r['assets']) for r in records), 8)
        self.assertEqual([r['number'] for r in records if r['role'] == 'schematic'], [1, 5])

    def test_confidence_grid(self):
        frame = figures.confidence_grid(ROOT)
        for column in ['single_initialization_confidence', 'same_initialization_after_warmup_confidence']:
            self.assertTrue(frame[column].between(.5, 1).all())
        settings = json.loads(checked(ROOT, 'artifacts/figure_inputs/heatmap_settings.json').read_text())
        self.assertIn('hard labels', settings['warmup']['labels'])

    def test_reliability_partition(self):
        # Boundary 0.5, near-certain confidence and an empty bin are retained correctly.
        rows = figures.reliability(np.array([[0., 0.], [1000., 0.], [0., 1.]]), np.array([0, 1, 1]), bins=2)
        self.assertEqual(rows.Count.tolist(), [1, 2])
        self.assertEqual(rows.iloc[0].Accuracy, 1)
        empty = figures.reliability(np.array([[0., 0.]]), np.array([0]), bins=15)
        self.assertTrue(empty[empty.Count == 0].Accuracy.isna().all())

    def test_intervals_include_all_seeds(self):
        raw, summary = figures.noise_statistics(ROOT)
        self.assertEqual(len(raw), 40)
        self.assertTrue((summary.Seeds == 10).all())
        sd = raw.groupby(['Dataset', 'Method'])['ECE (%)'].std(ddof=1)
        np.testing.assert_allclose(summary.High-summary.Mean, stats.t.ppf(.975, 9)*sd/np.sqrt(10))

    def test_generated_sources_and_scope(self):
        for number in [2, 3, 4, 6]:
            base = f'results/figures/figure{number}'
            record = json.loads(checked(ROOT, f'{base}/run.json').read_text())
            self.assertEqual(record['status'], 'complete')
            self.assertFalse(record['model_inference_executed'])
            self.assertEqual(record['figure'], number)
            self.assertTrue(record['scope'].startswith(figures.REDRAW_SCOPE[number]))
            for extension in ['png', 'svg', 'pdf']:
                self.assertGreater(checked(ROOT, f'{base}/figure{number}.{extension}').stat().st_size, 1000)
        response = pd.read_csv(checked(ROOT, 'results/figures/figure3/response_intervals.csv'))
        self.assertTrue((response.Seeds == 100).all())
        self.assertTrue(((response.Low <= response.Mean) & (response.Mean <= response.High)).all())
        for method, rows in response[response.Direction == 'v_weak_nuisance'].groupby('Method'):
            self.assertEqual(sorted(rows.Step.tolist()), list(range(0, 1001, 25)))
        settings = json.loads(checked(ROOT, 'results/figures/figure6/run.json').read_text())['statistics']
        self.assertEqual(settings['error_bars'], 'sample SD')

    def test_saved_gallery(self):
        notebook = nbformat.read(ROOT/'notebooks/quantitative_components.ipynb', as_version=4)
        nbformat.validate(notebook)
        self.assertEqual(notebook.metadata.gw.figures, 6)
        self.assertFalse(notebook.metadata.gw.model_inference_executed)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                self.assertIsNotNone(cell.execution_count)
                self.assertFalse(any(o.output_type == 'error' for o in cell.outputs))


if __name__ == '__main__':
    unittest.main()

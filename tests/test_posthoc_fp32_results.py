"""Verify corrected model outputs and calibration/input lineage, not just prose."""
from pathlib import Path
import json
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from gw import posthoc, posthoc_inference
from gw.classification_report import paper, comparison
from gw.supporting_reports import checked


class CorrectedPosthoc(unittest.TestCase):
    def test_all_models_splits_and_precision(self):
        for directory, split, samples in [('model_validation_fp32', 'val', 295),
                                           ('model_test_fp32', 'test', 292)]:
            state = json.loads(checked(ROOT, f'results/posthoc_compat/{directory}/run.json').read_text())
            self.assertEqual(len(state['records']), 6)
            for record in state['records']:
                self.assertEqual(record['device'], 'cuda')
                self.assertEqual(record['evaluation_split'], split)
                self.assertEqual(record['samples'], samples)
                self.assertEqual(record['backend']['precision'], 'fp32')
                self.assertFalse(record['backend']['cudnn_allow_tf32'])
                self.assertFalse(record['backend']['matmul_allow_tf32'])

    def test_ts_order_and_classification_metrics(self):
        _, mean, sd = posthoc.summarize(ROOT, ROOT/'results/posthoc_compat/model_refit_fp32')
        standard = mean.loc[('OCTDL','ResNet-18','Standard training + TS'), 'ECE (%)']
        gw = mean.loc[('OCTDL','ResNet-18','GW + TS'), 'ECE (%)']
        self.assertLess(standard, gw)
        check = comparison(mean, sd, paper(ROOT, 's6_posthoc'))
        for metric in ['Accuracy (%) ↑', 'Macro-F1 (%) ↑']:
            self.assertTrue(check.xs(metric, level='Metric').Matches.all())

    def test_refit_same_inputs_and_environment_repeatability(self):
        directory = ROOT/'results/posthoc_compat/model_refit_fp32'
        fresh = posthoc_inference.refit(ROOT, directory)
        _, mean, sd = posthoc.summarize(ROOT, directory)
        _, actual_mean, actual_sd = posthoc.summarize(ROOT, fresh)
        # Linux/macOS SciPy solvers can differ in the last reported decimals.
        # The bound is half one ECE reporting unit (0.001 percentage points).
        for actual, expected in [(actual_mean, mean), (actual_sd, sd)]:
            np.testing.assert_allclose(actual.values[:,:3], expected.values[:,:3], rtol=0, atol=0.0005)
            np.testing.assert_allclose(actual.values[:,3:], expected.values[:,3:], rtol=0, atol=0.00005)
        repeated = posthoc_inference.refit(ROOT, directory)
        _, repeated_mean, repeated_sd = posthoc.summarize(ROOT, repeated)
        np.testing.assert_array_equal(actual_mean.values, repeated_mean.values)
        np.testing.assert_array_equal(actual_sd.values, repeated_sd.values)

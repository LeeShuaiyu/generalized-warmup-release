"""Regression checks for split routing, explicit precision and refit lineage."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from gw.classification_inference import evaluation_dataset, precision_settings


class PosthocExecution(unittest.TestCase):
    def test_split_is_forwarded_without_loading_test_images(self):
        record = dict(dataset_key='oct', settings={'image_size':224})
        with patch('gw.classification_inference.ManifestDataset') as dataset:
            evaluation_dataset(ROOT, record, [], 'val')
            self.assertEqual(dataset.call_args.args[1], 'val')
            evaluation_dataset(ROOT, record, [])
            self.assertEqual(dataset.call_args.args[1], 'test')
        with self.assertRaisesRegex(ValueError, 'Unknown evaluation split'):
            evaluation_dataset(ROOT, record, [], 'validation')

    def test_fp32_flags_are_restored_even_on_failure(self):
        torch = SimpleNamespace(backends=SimpleNamespace(
            cudnn=SimpleNamespace(allow_tf32=True),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))))
        with self.assertRaises(RuntimeError):
            with precision_settings(torch, 'fp32') as state:
                self.assertFalse(state['cudnn_allow_tf32'])
                self.assertFalse(state['matmul_allow_tf32'])
                raise RuntimeError('fixture')
        self.assertTrue(torch.backends.cudnn.allow_tf32)
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        with precision_settings(torch, 'default') as state:
            self.assertTrue(state['cudnn_allow_tf32'])

    def test_refit_reuses_the_displayed_model_inputs(self):
        from gw import posthoc_inference, posthoc
        directory = ROOT/'results/posthoc_compat/model_refit'
        with patch('gw.posthoc.run', return_value=directory) as run:
            posthoc_inference.refit(ROOT, directory)
            sources = run.call_args.kwargs['prediction_sources']
        self.assertEqual(len(sources), 6)
        for pair in sources.values():
            self.assertEqual(set(pair), {'validation', 'test'})
            self.assertIn('source_execution_sha256', pair['test'])
            self.assertEqual(pair['test']['logits_key'], 'test_logits')
        with self.assertRaisesRegex(ValueError, 'Model-derived'):
            posthoc_inference.refit(ROOT, ROOT/'results/posthoc_compat/refit')

    def test_actual_val_test_samples_and_labels(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch is not installed')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root/'data/processed/oct'
            base.mkdir(parents=True)
            rows = []
            for index, split in enumerate(['test', 'val', 'train', 'val']):
                np.save(base/f'{index}.npy', np.ones((3,3), dtype=np.float32)*index)
                rows.append(dict(sample_id=str(index), split=split, class_id=str(index),
                                 data_path=f'{index}.npy', data_type='npy_float32'))
            record = dict(dataset_key='oct', settings={'image_size':3})
            val, _ = evaluation_dataset(root, record, rows, 'val')
            test, _ = evaluation_dataset(root, record, rows, 'test')
            self.assertEqual([int(val[i][1]) for i in range(len(val))], [1,3])
            self.assertEqual([int(test[i][1]) for i in range(len(test))], [0])

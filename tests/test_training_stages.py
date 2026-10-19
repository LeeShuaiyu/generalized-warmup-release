import importlib.util
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from gw.warmup import parse_method, energy_standardize, sample, uniform_soft_loss
from gw import training_stages as stages


class MethodSettings(unittest.TestCase):
    def test_registered_training_configurations(self):
        root = Path(__file__).resolve().parents[1]
        for experiment, expected in [('random', 140), ('pretrained', 30)]:
            runs = json.loads((root/f'configs/{experiment}.json').read_text())['runs']
            mapped = [r for r in runs if 'training_settings' in r]
            self.assertEqual(len(mapped), expected)
            for run in mapped:
                settings = run['training_settings']
                self.assertEqual(settings['seed'], run['seed'])
                self.assertTrue(all(settings[k]==v for k,v in run['settings'].items()))
                self.assertIn(settings['checkpoint_metric'], {'val_nll', 'val_accuracy'})
                self.assertFalse(settings['validation_calibration_split'])
                parse_method(settings['method'])

    def test_parameters(self):
        spec = parse_method('loggamma_warmup_p25_l8_r050')
        self.assertEqual((spec.probability, spec.shape, spec.rho), (0.25, 8, 0.5))
        spec = parse_method('addgamma_p50_l1_rho100')
        self.assertEqual((spec.probability, spec.shape, spec.rho), (0.5, 1, 1))
        self.assertEqual(parse_method('erm').family, 'none')
        with self.assertRaises(ValueError):
            parse_method('unknown')


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch environment required')
class TorchStages(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_gaussian_and_sample_gate(self):
        torch = self.torch
        shape = (6, 1, 8, 8)
        torch.manual_seed(42)
        expected = torch.randn(*shape)*0.4+0.2
        torch.manual_seed(42)
        actual = sample(parse_method('gaussian'), shape, torch, 'cpu', 0.2, 0.4)
        self.assertTrue(torch.equal(expected, actual))
        torch.manual_seed(42)
        actual = sample(parse_method('loggamma_warmup_p0_l4_r050'), shape, torch, 'cpu', 0.2, 0.4)
        self.assertTrue(torch.equal(expected, actual))
        actual = sample(parse_method('loggamma_warmup_p50_l4_r050'), shape, torch, 'cpu', 0.2, 0.4)
        self.assertTrue(torch.isfinite(actual).all())

    def test_image_standardization(self):
        torch = self.torch
        for name in ['addgamma_p50_l1_rho100', 'mulgamma_warmup_p50_l4_rho50']:
            x = sample(parse_method(name), (8, 1, 16, 16), torch, 'cpu')
            self.assertLess(float(x.mean((1, 2, 3)).abs().max()), 1e-6)
            self.assertLess(float((x.std((1, 2, 3), unbiased=False)-1).abs().max()), 1e-6)
        self.assertTrue(torch.equal(energy_standardize(torch.ones(2, 1, 8, 8)), torch.zeros(2, 1, 8, 8)))

    def test_uniform_loss(self):
        torch = self.torch
        logits = torch.randn(4, 3, requires_grad=True)
        target = torch.full_like(logits, 1/3)
        expected = -(target*torch.log_softmax(logits, dim=1)).sum(1).mean()
        actual = uniform_soft_loss(logits, torch)
        self.assertAlmostEqual(float(actual.detach()), float(expected.detach()), places=6)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_stage_updates_and_logits(self):
        torch = self.torch
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 3))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        before = model[1].weight.detach().clone()
        result = stages.warmup_epoch(model, parse_method('addgamma_p50_l1_rho50'), 2, 4, 8, 3, optimizer, 'cpu', torch)
        self.assertEqual(result['updates'], 2)
        self.assertFalse(torch.equal(before, model[1].weight))
        dataset = torch.utils.data.TensorDataset(torch.randn(8, 1, 8, 8), torch.arange(8)%3)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        result = stages.supervised_epoch(model, loader, optimizer, 'cpu', torch)
        self.assertGreaterEqual(result['accuracy'], 0)
        logits, labels = stages.collect_logits(model, loader, 'cpu', torch)
        self.assertEqual(logits.shape, (8, 3))
        self.assertEqual(labels.tolist(), (torch.arange(8)%3).tolist())
        with self.assertRaises(ValueError):
            stages.supervised_epoch(model, [], optimizer, 'cpu', torch)

from pathlib import Path
import json
import unittest
import importlib.util
import sys
import csv
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gw.warmup import parse_method
from gw.imagenet_warmup import sample, epoch


class LegacyConfiguration(unittest.TestCase):
    def test_source_verified_training_bindings(self):
        legacy = json.loads((ROOT / 'configs/pretrained_training.json').read_text())
        paper = {r['id']: r for r in json.loads((ROOT / 'configs/pretrained.json').read_text())['runs']}
        self.assertEqual(len(legacy['runs']), 88)
        self.assertEqual({r['id'] for r in legacy['unbound_source_versions']}, {'000', '010'})
        for row in legacy['runs']:
            c = row['training_settings']
            for name, value in paper[row['id']]['settings'].items():
                if name in c:
                    self.assertEqual(c[name], value)
            self.assertEqual(c['pretrained'], 'imagenet')
            self.assertEqual(c['sampling_protocol'], 'sar_imagenet_v1')
            self.assertFalse(c['validation_calibration_split'])
            self.assertEqual(len(row['source_code_sha256']), 2)
            self.assertEqual(len(row['original_config_sha256']), 64)
            self.assertEqual(row['posthoc_validation']['selector_fraction'], 0.7)
            self.assertTrue(row['posthoc_validation']['split_salt'])
            if row['dataset_key'] == 'fusar':
                self.assertTrue(c['class_balanced_sampler'])
                self.assertTrue(c['geometric_augmentation'])
            parse_method(c['method'])

    def test_paper_run_and_prediction_identities_unchanged(self):
        legacy = json.loads((ROOT / 'configs/pretrained_training.json').read_text())
        paper = {r['id']: r for r in json.loads((ROOT / 'configs/pretrained.json').read_text())['runs']}
        for row in legacy['runs']:
            for name in ['seed', 'classes', 'samples', 'predictions', 'logits_key', 'labels_key', 'settings']:
                self.assertEqual(row[name], paper[row['id']][name])

    def test_fusar_catalog_reuses_registered_manifest_and_images(self):
        from collections import Counter
        from gw.classification_data import load_manifest, manifest_sha256
        catalog = json.loads((ROOT / 'data/files.json').read_text())['fusar']
        region_catalog = json.loads((ROOT / 'data/region_files.json').read_text())['fusar']
        self.assertEqual(catalog, region_catalog)
        self.assertEqual(manifest_sha256(ROOT / catalog['manifest']), catalog['semantic_sha256'])
        rows = load_manifest(ROOT / catalog['manifest'])
        self.assertEqual(dict(Counter(r['split'] for r in rows)), {'train': 188, 'val': 48, 'test': 157})
        groups = {split: {r['group_id'] for r in rows if r['split'] == split}
                  for split in ['train', 'val', 'test']}
        self.assertFalse(groups['train'] & groups['val'] | groups['train'] & groups['test'] | groups['val'] & groups['test'])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch is not installed')
class LegacyOperations(unittest.TestCase):
    def test_training_entry_and_weight_reload_without_test_inputs(self):
        import numpy as np
        import torch
        from gw.assets import file_hash
        from gw.classification_data import manifest_sha256, load_manifest
        from gw.imagenet_data import ImageDataset
        from gw.classification_training import train
        from gw.training_stages import collect_logits

        class SmallModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = torch.nn.Conv2d(1, 2, 1)
                self.fc = torch.nn.Linear(2, 3)
            def forward(self, x):
                return self.fc(self.stem(x).mean(dim=(-2, -1)))

        source = json.loads((ROOT / 'configs/pretrained_training.json').read_text())['runs'][0]
        c = dict(source['training_settings'], seed=101, image_size=8, epochs=1,
                 warmup_epochs=1, batch_size=4, num_workers=0,
                 method='loggamma_warmup_p25_l4_r025',
                 class_balanced_sampler=True, geometric_augmentation=True)
        with tempfile.TemporaryDirectory(prefix='gw-training-test-', dir=ROOT.parent) as temporary:
            root = Path(temporary)
            (root / 'data/manifests').mkdir(parents=True)
            images = root / 'data/processed/sar'
            images.mkdir(parents=True)
            rows, files = [], []
            for split, count in [('train', 9), ('val', 6), ('test', 3)]:
                for index in range(count):
                    name = f'{split}-{index}.npy'
                    row = dict(sample_id=f'{split}-{index}', class_id=str(index % 3),
                               split=split, data_path=name, data_type='npy_float32')
                    rows.append(row)
                    # Test files are deliberately absent: training must not open them.
                    if split != 'test':
                        np.save(images / name, np.arange(64, dtype=np.float32).reshape(8, 8) + index)
                    files.append(dict(sample_id=row['sample_id'], split=split, path=name,
                                      sha256=file_hash(images / name) if split != 'test' else '0' * 64))
            manifest = root / 'data/manifests/sar.csv'
            with manifest.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            split_hash = manifest_sha256(manifest)
            (root / 'data/files.json').write_text(json.dumps({'sar': {
                'manifest': 'data/manifests/sar.csv', 'semantic_sha256': split_hash, 'files': files}}))
            run = dict(source, dataset_key='sar', classes=3, split_sha256=split_hash, training_settings=c)
            torch.manual_seed(c['seed'])
            expected_initialization_rng = torch.get_rng_state().clone()
            def factory(architecture, classes, pretrained, size):
                self.assertEqual(pretrained, 'imagenet')
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_initialization_rng))
                return SmallModel()
            with patch('gw.classification_training.build_model', factory):
                output = train(root, 'pretrained', run, device='cpu')
            record = json.loads((output / 'run.json').read_text())
            self.assertEqual(record['status'], 'complete')
            self.assertEqual(record['split_counts'], {'train': 9, 'val': 6})
            self.assertFalse(record['test_evaluated'])
            self.assertIn('imagenet_warmup.py', record['code_sha256'])
            model = SmallModel()
            model.load_state_dict(torch.load(output / 'best_model.pt', map_location='cpu', weights_only=True))
            dataset = ImageDataset(load_manifest(manifest), 'val', image_size=8, data_root=images)
            values, labels = collect_logits(model, torch.utils.data.DataLoader(dataset, batch_size=4), 'cpu', torch)
            with np.load(output / 'predictions.npz') as predictions:
                self.assertTrue(np.array_equal(values, predictions['val_logits']))
                self.assertTrue(np.array_equal(labels, predictions['val_labels']))
                self.assertFalse(any(key.startswith('test') for key in predictions.files))

    def test_matching_native_size_still_quantizes(self):
        import numpy as np
        import torch
        from gw.imagenet_data import ImageDataset
        from gw.classification_data import ManifestDataset, _normalize_array
        with tempfile.TemporaryDirectory(prefix='gw-image-test-', dir=ROOT.parent) as directory:
            root = Path(directory)
            array = np.linspace(0, 1, 441, dtype=np.float32).reshape(21, 21)
            np.save(root / 'image.npy', array)
            rows = [dict(sample_id='a', class_id='0', split='train', data_path='image.npy', data_type='npy_float32')]
            expected = (np.clip(_normalize_array(array), 0, 1) * 255).astype('uint8').astype(np.float32) / 255
            actual, _ = ImageDataset(rows, 'train', image_size=21, data_root=root)[0]
            common, _ = ManifestDataset(rows, 'train', image_size=21, data_root=root)[0]
            self.assertTrue(np.array_equal(actual.numpy()[0], expected))
            self.assertFalse(torch.equal(actual, common))

    def test_balanced_loader_uses_separate_sampler_and_global_base_seed(self):
        import torch
        from gw.imagenet_warmup import make_loader
        class Dataset:
            rows = [{'class_id': str(i % 3)} for i in range(9)]
            def __len__(self): return len(self.rows)
            def __getitem__(self, index): return index, index % 3
        dataset = Dataset()
        torch.manual_seed(611)
        loader = make_loader(dataset, 'train', 4, 101, torch, balanced=True)
        observed = [x for x, _ in loader]
        observed_rng = torch.get_rng_state().clone()
        torch.manual_seed(611)
        generator = torch.Generator().manual_seed(302)
        sampler = torch.utils.data.WeightedRandomSampler([1 / 3] * 9, 9, replacement=True, generator=generator)
        expected = [x for x, _ in torch.utils.data.DataLoader(dataset, batch_size=4, sampler=sampler)]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(observed, expected)))
        self.assertTrue(torch.equal(observed_rng, torch.get_rng_state()))
        self.assertIsNone(loader.generator)

    def test_residual_before_gate_and_clipped_image_range(self):
        import torch
        from gw.warmup import sample as common_sample
        spec = parse_method('loggamma_warmup_p50_l8_r050')
        torch.manual_seed(151)
        x = torch.randn(8, 1, 8, 8) * .5 + .1
        concentration = torch.full_like(x, 8.)
        gamma = torch.distributions.Gamma(concentration, concentration).sample().clamp_min(1e-12)
        looks = torch.tensor(8., dtype=x.dtype)
        q = (torch.log(gamma) - (torch.special.digamma(looks) - torch.log(looks))) / torch.sqrt(torch.polygamma(1, looks).clamp_min(1e-12))
        structured = x + .5 * .5 * q
        mask = torch.rand((8, 1, 1, 1)) < .5
        expected = torch.where(mask, structured, x).clamp(0., 1.)
        state = torch.get_rng_state()
        torch.manual_seed(151)
        actual = sample(spec, (8, 1, 8, 8), torch, 'cpu', .1, .5)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        torch.manual_seed(151)
        common = common_sample(spec, (8, 1, 8, 8), torch, 'cpu', .1, .5)
        self.assertFalse(torch.equal(actual, common))
        self.assertEqual(float(actual.min()), 0.)
        self.assertEqual(float(actual.max()), 1.)

    def test_gaussian_multiplier_and_invalid_shapes(self):
        import torch
        spec = parse_method('gaussian_warmup')
        torch.manual_seed(201)
        expected = (torch.randn(4, 1, 8, 8) * (.3 * 1.25) + .2).clamp(0., 1.)
        torch.manual_seed(201)
        self.assertTrue(torch.equal(sample(spec, (4, 1, 8, 8), torch, 'cpu', .2, .3, 1.25), expected))
        with self.assertRaises(ValueError):
            sample(spec, (4, 3, 8, 8), torch, 'cpu', .2, .3)
        with self.assertRaises(ValueError):
            sample(parse_method('erm'), (4, 1, 8, 8), torch, 'cpu', .2, .3)

    def test_uniform_target_update(self):
        import torch
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 3))
        before = model[1].weight.detach().clone()
        result = epoch(model, parse_method('gaussian'), 2, 4, 8, 3,
                       torch.optim.AdamW(model.parameters(), lr=.001), 'cpu', torch, .1, .5)
        self.assertEqual(result['updates'], 2)
        self.assertGreater(result['loss'], 0.)
        self.assertFalse(torch.equal(before, model[1].weight))
        with self.assertRaises(ValueError):
            epoch(model, parse_method('gaussian'), 0, 4, 8, 3,
                  torch.optim.AdamW(model.parameters()), 'cpu', torch, .1, .5)


if __name__ == '__main__':
    unittest.main()

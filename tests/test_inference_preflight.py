from pathlib import Path
import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gw.classification_data import load_manifest, manifest_sha256
from gw.classification_inference import preflight, run
from gw.data_assets import verify_inputs
from gw.inference import preflight_experiment, run_experiment


class InferenceChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'project'
        (self.root / 'data/manifests').mkdir(parents=True)
        self.rows = [{'sample_id': 'one', 'class_id': '0', 'split': 'test', 'data_path': 'data/one.npy'},
                     {'sample_id': 'two', 'class_id': '1', 'split': 'train', 'data_path': 'data/two.npy'}]
        manifest = self.root / 'data/manifests/opensar.csv'
        with manifest.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        (self.root / 'data/manifests/opensarship2.csv').write_bytes(manifest.read_bytes())
        self.digest = manifest_sha256(manifest)
        self.catalog = {'opensar': {'manifest': 'data/manifests/opensar.csv', 'semantic_sha256': self.digest,
                       'files': [{'sample_id': row['sample_id'], 'split': row['split'], 'path': row['data_path'],
                                  'sha256': hashlib.sha256(row['sample_id'].encode()).hexdigest()}
                                 for row in self.rows]}}
        (self.root / 'data/files.json').write_text(json.dumps(self.catalog))
        self.inputs = self.root / 'data/processed/opensar'
        (self.inputs / 'data').mkdir(parents=True)
        (self.inputs / 'data/one.npy').write_bytes(b'one')
        # The unselected training image is deliberately not installed.
        self.checkpoint = self.root / 'artifacts/checkpoints/random/000.pt'
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b'checkpoint fixture; do not deserialize')
        self.checkpoint_hash = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.config = {'experiment_id': 'random', 'evaluation_split': 'test', 'runs': [
            {'id': '000', 'dataset_key': 'opensar', 'samples': 1, 'classes': 2,
             'split_sha256': self.digest, 'checkpoint': 'artifacts/checkpoints/random/000.pt',
             'checkpoint_sha256': self.checkpoint_hash}]}
        (self.root / 'artifacts/manifest.json').write_text(json.dumps({'checkpoints': {
            's4_family/gw/1': {'sha256': self.checkpoint_hash}}}))
        self.family = {'experiment_id': 's4_family', 'evaluation_split': 'test', 'evaluation_samples': 1,
                       'manifest_sha256': self.digest, 'num_classes': 2,
                       'conditions': [{'id': 'gw', 'label': 'GW'}], 'seeds': [1]}
        family_models = self.root / 'artifacts/checkpoints/s4_family'
        family_models.mkdir()
        (family_models / 'gw_1.pt').write_bytes(self.checkpoint.read_bytes())

    def family_check(self):
        return preflight_experiment(self.root, self.family, data_root=self.inputs)

    def test_valid_checks_do_not_import_models_or_open_training_images(self):
        with patch.dict(sys.modules, {'torch': None}):
            jobs, datasets, checks = preflight(self.root, self.config)
            family = self.family_check()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(checks['opensar']['samples_verified'], 1)
        self.assertEqual(family[3]['samples_verified'], 1)
        self.assertEqual(len(datasets['opensar']), 2)
        self.assertFalse((self.root / 'output').exists())

    def test_family_defaults_use_the_prepared_dataset_directory(self):
        # prepare_data/prepare_original install this canonical directory. No
        # manually created legacy opensarship2 alias should be necessary.
        with patch.dict(sys.modules, {'torch': None}):
            data_root, selected, jobs, checked, base = preflight_experiment(self.root, self.family)
        self.assertEqual(data_root.resolve(), self.inputs.resolve())
        self.assertEqual(checked['samples_verified'], 1)
        self.assertFalse((self.root / 'data/processed/opensarship2').exists())

    def test_changed_image_rejected_before_model_import_or_output(self):
        (self.inputs / 'data/one.npy').write_bytes(b'changed input')
        with patch.dict(sys.modules, {'torch': None}):
            with self.assertRaisesRegex(ValueError, 'image checksum'):
                run(self.root, self.config, device='cpu')
            with self.assertRaisesRegex(ValueError, 'image checksum'):
                run_experiment(self.root, self.family, data_root=self.inputs, device='cpu')
        self.assertFalse((self.root / 'output').exists())

    def test_missing_image_is_not_replaced_with_archived_predictions(self):
        (self.inputs / 'data/one.npy').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'evaluation image'):
            preflight(self.root, self.config)
        with self.assertRaises(FileNotFoundError):
            self.family_check()

    def test_checkpoint_change_rejected(self):
        self.checkpoint.write_bytes(b'wrong checkpoint')
        with self.assertRaisesRegex(ValueError, 'Checkpoint identity'):
            preflight(self.root, self.config)

    def test_unknown_empty_or_duplicate_run_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown run'):
            preflight(self.root, self.config, ['absent'])
        with self.assertRaisesRegex(ValueError, 'No selected'):
            preflight(self.root, self.config, [])
        duplicate = copy.deepcopy(self.config)
        duplicate['runs'] *= 2
        with self.assertRaisesRegex(ValueError, 'Duplicate configured'):
            preflight(self.root, duplicate)

    def test_missing_checkpoint_mapping_or_count_rejected(self):
        config = copy.deepcopy(self.config)
        del config['runs'][0]['checkpoint']
        with self.assertRaisesRegex(ValueError, 'mapping required'):
            preflight(self.root, config)
        config = copy.deepcopy(self.config)
        config['runs'][0]['samples'] = 2
        with self.assertRaisesRegex(ValueError, 'sample count'):
            preflight(self.root, config)

    def test_false_manifest_rows_or_path_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['class_id'] = '1'
        with self.assertRaisesRegex(ValueError, 'rows differ'):
            verify_inputs(self.root, 'opensar', rows, 'test', self.inputs)
        rows = copy.deepcopy(self.rows)
        rows[0]['data_path'] = 'another.npy'
        with self.assertRaisesRegex(ValueError, 'path differs'):
            verify_inputs(self.root, 'opensar', rows, 'test', self.inputs)

    def test_duplicate_catalog_identity_rejected(self):
        self.catalog['opensar']['files'].append(self.catalog['opensar']['files'][0])
        (self.root / 'data/files.json').write_text(json.dumps(self.catalog))
        with self.assertRaisesRegex(ValueError, 'sample identities'):
            preflight(self.root, self.config)

    def test_eight_gib_reserve(self):
        disk = type('Disk', (), {'free': 8 * 1024**3})()
        with patch('gw.classification_inference.shutil.disk_usage', return_value=disk):
            with self.assertRaisesRegex(RuntimeError, '8 GiB'):
                preflight(self.root, self.config)
            with self.assertRaisesRegex(RuntimeError, '8 GiB'):
                self.family_check()
        self.assertFalse((self.root / 'output').exists())

    def test_output_escape_and_invalid_device_rejected(self):
        config = copy.deepcopy(self.config)
        config['experiment_id'] = '../escape'
        with self.assertRaises(ValueError):
            preflight(self.root, config)
        outside = Path(self.temp.name) / 'outside'
        outside.mkdir()
        (self.root / 'output').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'outside'):
            preflight(self.root, self.config)
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.family_check()
        with self.assertRaisesRegex(ValueError, 'Unknown device'):
            run(self.root, self.config, device='other')
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == '__main__':
    unittest.main()

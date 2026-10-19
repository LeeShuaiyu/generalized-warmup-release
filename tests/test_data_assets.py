from pathlib import Path
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gw.classification_data import manifest_sha256
from gw.data_assets import prepare


class DataPreparation(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'project'
        self.source = Path(self.temporary.name) / 'source'
        (self.root / 'data/manifests').mkdir(parents=True)
        self.relative = 'data/images/one.npy'
        source = self.source / self.relative
        source.parent.mkdir(parents=True)
        source.write_bytes(b'fixed pixels')
        manifest = self.root / 'data/manifests/sar.csv'
        with manifest.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['sample_id', 'class_id', 'split', 'data_path'])
            writer.writeheader()
            writer.writerow(dict(sample_id='one', class_id='0', split='test', data_path=self.relative))
        self.item = {'manifest': 'data/manifests/sar.csv', 'semantic_sha256': manifest_sha256(manifest),
                     'files': [{'path': self.relative, 'sample_id': 'one', 'split': 'test',
                                'sha256': hashlib.sha256(b'fixed pixels').hexdigest()}]}
        self.catalog()

    def catalog(self):
        (self.root / 'data/files.json').write_text(json.dumps({'sar': self.item}))

    def test_prepare_and_check(self):
        counts = prepare(self.root, 'sar', self.source, split='test')
        self.assertEqual(counts['prepared'], 1)
        counts = prepare(self.root, 'sar', check=True)
        self.assertEqual(counts['verified'], 1)
        self.assertEqual(prepare(self.root, 'sar', self.source)['verified'], 1)

    def test_missing_does_not_count_as_verified(self):
        counts = prepare(self.root, 'sar', check=True)
        self.assertEqual(counts['missing'], 1)
        self.assertEqual(counts['verified'], 0)

    def test_split_change_is_rejected(self):
        manifest = self.root / self.item['manifest']
        manifest.write_text(manifest.read_text().replace(',0,test,', ',1,test,'))
        with self.assertRaisesRegex(ValueError, 'split changed'):
            prepare(self.root, 'sar', self.source)

    def test_invalid_existing_data_is_not_overwritten(self):
        target = self.root / 'data/processed/sar' / self.relative
        target.parent.mkdir(parents=True)
        target.write_bytes(b'user data')
        with self.assertRaises(ValueError):
            prepare(self.root, 'sar', self.source)
        self.assertEqual(target.read_bytes(), b'user data')

    def test_bad_original_source_is_rejected(self):
        (self.source / self.relative).write_bytes(b'wrong pixels')
        with self.assertRaises(ValueError):
            prepare(self.root, 'sar', self.source)
        self.assertFalse((self.root / 'data/processed/sar' / self.relative).exists())

    def test_eight_gib_reserve(self):
        with patch('gw.data_assets.shutil.disk_usage',
                   return_value=type('Disk', (), {'free': 8 * 1024**3})()):
            with self.assertRaisesRegex(RuntimeError, '8 GiB'):
                prepare(self.root, 'sar', self.source)
        self.assertFalse((self.root / 'data/processed/sar' / self.relative).exists())

    def test_verified_existing_link_is_read_only(self):
        target = self.root / 'data/processed/sar' / self.relative
        target.parent.mkdir(parents=True)
        target.symlink_to(self.source / self.relative)
        self.assertEqual(prepare(self.root, 'sar', check=True)['verified'], 1)
        self.assertEqual(prepare(self.root, 'sar', self.source)['verified'], 1)
        self.assertTrue(target.is_symlink())

    def test_missing_destination_under_external_link_cannot_be_written(self):
        external = self.source / 'outside'
        external.mkdir()
        (self.root / 'data/processed').symlink_to(external)
        with self.assertRaisesRegex(ValueError, 'outside'):
            prepare(self.root, 'sar', self.source)
        self.assertEqual(list(external.iterdir()), [])


if __name__ == '__main__':
    unittest.main()

"""Bootstrap checks do not require notebook, NumPy or Torch dependencies."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/restore_release.py'
spec = importlib.util.spec_from_file_location('gw_restore_release', SCRIPT)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class ReleaseRestoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.destination = self.root / 'project'
        self.index = dict(status='private_candidate', published=False, packages={})
        self.make('code', {'src/method.py': b'print("fixture")\n'})
        self.make('results', {'results/table.csv': b'ECE\n1.0\n'})
        self.index_path = self.root / 'index.json'
        self.save()

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        self.index_path.write_text(json.dumps(self.index))

    def make(self, kind, files, *, member_names=None, link=False):
        name = f'gw-{kind}.zip'
        path = self.root / name
        with zipfile.ZipFile(path, 'w') as archive:
            for key in member_names or list(files):
                info = zipfile.ZipInfo(key)
                info.external_attr = ((stat.S_IFLNK if link else stat.S_IFREG) | 0o644) << 16
                archive.writestr(info, files.get(key, b'unknown'))
        self.index['packages'][kind] = dict(file=name, bytes=path.stat().st_size,
            sha256=bootstrap.file_hash(path), files=[dict(path=key, bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest()) for key, data in files.items()])

    def run_restore(self, **kwargs):
        self.save()
        return bootstrap.restore(self.index_path, self.root, self.destination,
                                 reserve_bytes=0, **kwargs)

    def test_restore_and_receipt(self):
        result = self.run_restore(index_sha256=bootstrap.file_hash(self.index_path))
        self.assertEqual(result['status'], 'restored')
        self.assertEqual(result['files'], 2)
        self.assertFalse(result['checkpoint_inference_executed'])
        self.assertEqual((self.destination / 'results/table.csv').read_bytes(), b'ECE\n1.0\n')
        self.assertEqual(json.loads((self.destination / bootstrap.RECEIPT).read_text()), result)

    def test_existing_project_not_overwritten(self):
        self.destination.mkdir()
        marker = self.destination / 'keep.txt'
        marker.write_text('keep')
        with self.assertRaises(FileExistsError):
            self.run_restore()
        self.assertEqual(marker.read_text(), 'keep')

    def test_broken_destination_link_rejected(self):
        self.destination.symlink_to(self.root / 'missing')
        with self.assertRaises(FileExistsError):
            self.run_restore()
        self.assertTrue(self.destination.is_symlink())

    def test_bad_index_hash(self):
        with self.assertRaisesRegex(ValueError, 'index SHA256'):
            self.run_restore(index_sha256='0' * 64)
        self.assertFalse(self.destination.exists())

    def test_bad_archive_hash_before_write(self):
        self.index['packages']['results']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.run_restore()
        self.assertFalse(self.destination.exists())

    def test_bad_member_hash_retains_failure_record(self):
        self.index['packages']['results']['files'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'file identity mismatch'):
            self.run_restore()
        result = json.loads((self.destination / bootstrap.RECEIPT).read_text())
        self.assertEqual(result['status'], 'failed')
        self.assertTrue((self.destination / 'src/method.py').exists())

    def test_archive_link_rejected(self):
        self.make('results', {'results/table.csv': b'link'}, link=True)
        with self.assertRaisesRegex(ValueError, 'ZIP links'):
            self.run_restore()
        self.assertFalse(self.destination.exists())

    def test_extra_and_duplicate_members_rejected(self):
        for names in (['results/table.csv', 'extra.txt'], ['results/table.csv', 'results/table.csv']):
            with self.subTest(names=names):
                self.make('results', {'results/table.csv': b'1'}, member_names=names)
                with self.assertRaisesRegex(ValueError, 'ZIP member'):
                    self.run_restore()
                self.assertFalse(self.destination.exists())

    def test_missing_member_rejected(self):
        self.make('results', {'results/table.csv': b'1', 'results/other.csv': b'2'},
                  member_names=['results/table.csv'])
        with self.assertRaisesRegex(ValueError, 'ZIP member'):
            self.run_restore()

    def test_unsafe_and_noncanonical_paths(self):
        for value in ('../outside', '/absolute', 'a/../b', 'a\\b', 'a//b', './a', 'C:/a'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bootstrap.safe_path(value)

    def test_cross_package_collision_rejected(self):
        for path in ('src/method.py', 'src/method.py/child'):
            self.make('results', {path: b'bad'})
            with self.assertRaisesRegex(ValueError, 'overlapping|collision'):
                self.run_restore()
            self.assertFalse(self.destination.exists())

    def test_reserved_receipt_rejected(self):
        self.make('results', {bootstrap.RECEIPT: b'bad'})
        with self.assertRaisesRegex(ValueError, 'reserved'):
            self.run_restore()

    def test_index_duplicate_rejected(self):
        files = self.index['packages']['code']['files']
        files.append(files[0].copy())
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.run_restore()

    def test_low_space_rejected_before_write(self):
        with patch.object(bootstrap.shutil, 'disk_usage', return_value=type('Space', (), {'free': 0})()):
            with self.assertRaisesRegex(RuntimeError, 'space'):
                self.run_restore()
        self.assertFalse(self.destination.exists())

    def test_local_archive_symlink_rejected(self):
        actual = self.root / 'source.zip'
        os.rename(self.root / 'gw-code.zip', actual)
        (self.root / 'gw-code.zip').symlink_to(actual)
        with self.assertRaisesRegex(ValueError, 'regular local'):
            self.run_restore()

    def test_wrong_size_and_boolean_size(self):
        for value in (True, -1, 300):
            self.index['packages']['code']['files'][0]['bytes'] = value
            with self.assertRaisesRegex(ValueError, 'byte count|size differs'):
                self.run_restore()


if __name__ == '__main__':
    unittest.main()

from pathlib import Path
import hashlib
import tempfile
import unittest
import zipfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gw.result_packages import prepare


class ResultPackages(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'project'
        self.root.mkdir()
        self.archive = Path(self.temp.name) / 'results.zip'

    def package(self, paths=None):
        paths = paths or {'results/test.json': b'{"value":1}', 'artifacts/toy/weights.npz': b'toy'}
        with zipfile.ZipFile(self.archive, 'w') as archive:
            for path, content in paths.items():
                archive.writestr(path, content)
        item = dict(file=self.archive.name, bytes=self.archive.stat().st_size,
                    sha256=hashlib.sha256(self.archive.read_bytes()).hexdigest(),
                    files=[dict(path=p, bytes=len(c), sha256=hashlib.sha256(c).hexdigest())
                           for p, c in paths.items()])
        return dict(results=item)

    def test_import_reuse_and_check(self):
        index = self.package()
        self.assertEqual(prepare(self.root, index, check=True)['missing'], 2)
        self.assertEqual(prepare(self.root, index, archive_path=self.archive, reserve_bytes=0)['imported'], 2)
        self.assertEqual(prepare(self.root, index, reserve_bytes=0)['imported'], 0)
        self.assertEqual(prepare(self.root, index, check=True)['verified'], 2)

    def test_bad_archive_hash_no_writes(self):
        index = self.package()
        index['results']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            prepare(self.root, index, archive_path=self.archive, reserve_bytes=0)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_content_hash_checked_before_writes(self):
        index = self.package()
        index['results']['files'][-1]['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            prepare(self.root, index, archive_path=self.archive, reserve_bytes=0)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_conflict_not_overwritten(self):
        index = self.package()
        (self.root / 'results').mkdir()
        target = self.root / 'results/test.json'
        target.write_bytes(b'user data')
        with self.assertRaises(ValueError):
            prepare(self.root, index, archive_path=self.archive, reserve_bytes=0)
        self.assertEqual(target.read_bytes(), b'user data')

    def test_path_traversal(self):
        for path in ('../escape', '/results/escape', 'results/../escape', 'src/code.py'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                prepare(self.root, self.package({path:b'x'}), archive_path=self.archive, reserve_bytes=0)

    def test_symlink_destination(self):
        index = self.package()
        outside = Path(self.temp.name) / 'outside'
        outside.mkdir()
        (self.root / 'results').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            prepare(self.root, index, archive_path=self.archive, reserve_bytes=0)
        self.assertEqual(list(outside.iterdir()), [])

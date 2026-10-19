from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gw.assets import inventory, prepare, read_catalog


class AssetPreparation(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root, self.source = self.base / 'project', self.base / 'assets'
        (self.root / 'artifacts').mkdir(parents=True)
        self.relative = 'artifacts/checkpoints/test/001.pt'
        self.body = b'fixed checkpoint bytes'
        self.item = {'destination': self.relative, 'bytes': len(self.body),
                     'sha256': hashlib.sha256(self.body).hexdigest()}
        self.catalog()
        self.original = self.source / self.relative
        self.original.parent.mkdir(parents=True)
        self.original.write_bytes(self.body)

    def catalog(self):
        (self.root / 'artifacts/manifest.json').write_text(json.dumps(
            {'checkpoints': {'test/001': self.item}}))

    def test_offline_prepare_and_idempotence(self):
        self.assertEqual(prepare(self.root, source_root=self.source)[0]['status'], 'prepared')
        self.assertEqual(prepare(self.root)[0]['status'], 'reused')
        self.assertEqual(inventory(self.root)[0]['status'], 'verified')
        self.assertEqual((self.root / self.relative).read_bytes(), self.body)

    def test_existing_corrupt_file_is_preserved(self):
        target = self.root / self.relative
        target.parent.mkdir(parents=True)
        target.write_bytes(b'user data')
        with self.assertRaises(ValueError):
            prepare(self.root, source_root=self.source)
        self.assertEqual(target.read_bytes(), b'user data')

    def test_no_url_is_not_a_fake_success(self):
        with self.assertRaisesRegex(ValueError, 'No download URL'):
            prepare(self.root)
        self.assertFalse((self.root / self.relative).exists())

    def test_bad_source_leaves_no_destination(self):
        self.original.write_bytes(b'bad')
        with self.assertRaises(ValueError):
            prepare(self.root, source_root=self.source)
        self.assertFalse((self.root / self.relative).exists())

    def test_path_traversal_is_rejected(self):
        self.item['destination'] = '../outside.pt'
        self.catalog()
        with self.assertRaises(ValueError):
            read_catalog(self.root)

    def test_external_parent_link_is_rejected(self):
        target = self.root / 'artifacts/checkpoints'
        target.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'outside'):
            prepare(self.root, source_root=self.source)

    def test_https_transfer_verified_before_installation(self):
        import io
        class Response(io.BytesIO):
            def geturl(self): return 'https://example.org/fixed/checkpoint'
        with patch('gw.assets.urllib.request.urlopen', return_value=Response(self.body)):
            rows = prepare(self.root, urls={'test/001': 'https://example.org/fixed/checkpoint'})
        self.assertEqual(rows[0]['status'], 'prepared')
        self.assertEqual(list((self.root / self.relative).parent.glob('.gw-download-*')), [])

    def test_bad_download_is_rejected_and_temporary_removed(self):
        import io
        class Response(io.BytesIO):
            def geturl(self): return 'https://example.org/fixed/checkpoint'
        with patch('gw.assets.urllib.request.urlopen', return_value=Response(b'x' * len(self.body))):
            with self.assertRaises(ValueError):
                prepare(self.root, urls={'test/001': 'https://example.org/fixed/checkpoint'})
        self.assertFalse((self.root / self.relative).exists())
        self.assertEqual(list((self.root / self.relative).parent.glob('.gw-download-*')), [])


if __name__ == '__main__':
    unittest.main()

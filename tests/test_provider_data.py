from pathlib import Path
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gw.classification_data import manifest_sha256
from gw.data_assets import prepare
from gw.provider_data import OCTDL, RESERVE, download_oct, prepare_original, unpack_oct, verify_archive


class ProviderPreparation(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'project'
        (self.root / 'data/manifests').mkdir(parents=True)
        stream = io.BytesIO()
        Image.fromarray(np.arange(9, dtype=np.uint8).reshape(3, 3)).save(stream, format='JPEG')
        self.payload = stream.getvalue()
        self.rows = [{'sample_id': f'amd_{i}', 'class_id': '0', 'class_name': 'AMD',
                      'split': split, 'group_id': f'patient_{i}', 'height': '3', 'width': '3',
                      'raw_path': f'AMD/amd_{i}.jpg',
                      'data_path': f'data/octdl/images/OCTDL/AMD/amd_{i}.jpg'}
                     for i, split in enumerate(['train', 'test'])]
        manifest = self.root / 'data/manifests/oct.csv'
        with manifest.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.item = {'manifest': 'data/manifests/oct.csv', 'semantic_sha256': manifest_sha256(manifest),
                     'files': [{'sample_id': row['sample_id'], 'path': row['data_path'],
                                'split': row['split'], 'sha256': hashlib.sha256(self.payload).hexdigest()}
                               for row in self.rows]}
        (self.root / 'data/files.json').write_text(json.dumps({'oct': self.item}))
        self.archive = Path(self.temporary.name) / 'OCTDL.zip'
        self.make_archive()

    def make_archive(self, extra=None, payload=None, prefix='OCTDL/'):
        with zipfile.ZipFile(self.archive, 'w') as archive:
            for row in self.rows:
                archive.writestr(f'{prefix}AMD/{row["sample_id"]}.jpg',
                                 self.payload if payload is None else payload)
            if extra:
                extra(archive)
        content = self.archive.read_bytes()
        self.pin = {'bytes': len(content), 'md5': hashlib.md5(content).hexdigest()}

    def test_archive_unpack_conversion_install_and_reuse(self):
        original = self.archive.read_bytes()
        with patch.dict(OCTDL, self.pin):
            result = prepare_original(self.root, 'oct', archive=self.archive, split='test')
            self.assertEqual(result['installed']['verified'], 1)
            self.assertEqual(result['import']['prepared'], 1)
            self.assertFalse(result['model_inference_executed'])
            target = self.root / 'data/processed/oct' / self.rows[1]['data_path']
            self.assertEqual(target.read_bytes(), self.payload)
            self.assertFalse((self.root / 'data/processed/oct' / self.rows[0]['data_path']).exists())
            with patch('gw.provider_data.download_oct', side_effect=AssertionError('No duplicate download')):
                again = prepare_original(self.root, 'oct', download=True, split='test')
            self.assertEqual(again['action'], 'reused verified inputs')
            self.assertEqual(len(list((self.root / 'output/provider_data').iterdir())), 1)
        self.assertEqual(self.archive.read_bytes(), original)

    def test_check_lists_required_files_without_writing(self):
        result = prepare_original(self.root, 'oct', check=True, split='test')
        self.assertEqual(result['required_provider_files'], ['AMD/amd_1.jpg'])
        self.assertEqual(result['missing_provider_files'], ['AMD/amd_1.jpg'])
        self.assertEqual(result['installed']['missing'], 1)
        self.assertFalse((self.root / 'output').exists())
        with self.assertRaisesRegex(ValueError, 'does not download'):
            prepare_original(self.root, 'oct', check=True, download=True)

    def test_original_directory_conversion_installs_all(self):
        raw = Path(self.temporary.name) / 'raw'
        (raw / 'AMD').mkdir(parents=True)
        for row in self.rows:
            (raw / 'AMD' / f'{row["sample_id"]}.jpg').write_bytes(self.payload)
        result = prepare_original(self.root, 'oct', raw_dir=raw)
        self.assertEqual(result['installed']['verified'], 2)
        self.assertEqual(prepare_original(self.root, 'oct', raw_dir=raw, check=True)['missing_provider_files'], [])

    def test_input_modes_and_missing_sources_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'either'):
            prepare_original(self.root, 'oct', archive=self.archive, raw_dir=self.root)
        with self.assertRaisesRegex(ValueError, 'download alone'):
            prepare_original(self.root, 'oct', download=True, raw_dir=self.root)
        with self.assertRaisesRegex(ValueError, 'only for OCTDL'):
            prepare_original(self.root, 'sar', download=True)
        with self.assertRaisesRegex(ValueError, 'Supply --raw-dir'):
            prepare_original(self.root, 'oct')

    def test_bad_installed_data_is_not_replaced(self):
        target = self.root / 'data/processed/oct' / self.rows[1]['data_path']
        target.parent.mkdir(parents=True)
        target.write_bytes(b'user data')
        with self.assertRaisesRegex(ValueError, 'no files replaced'):
            prepare_original(self.root, 'oct', archive=self.archive, split='test')
        self.assertEqual(target.read_bytes(), b'user data')

    def test_archive_checksum_and_link_rejected(self):
        with patch.dict(OCTDL, self.pin):
            self.assertEqual(verify_archive(self.archive), hashlib.sha256(self.archive.read_bytes()).hexdigest())
            linked = self.archive.with_name('linked.zip')
            linked.symlink_to(self.archive)
            with self.assertRaises(ValueError):
                verify_archive(linked)
            content = bytearray(self.archive.read_bytes())
            content[0] ^= 1
            self.archive.write_bytes(content)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verify_archive(self.archive)

    def test_wrong_registered_pixels_leave_failed_record(self):
        self.make_archive(payload=b'not the registered JPEG')
        with patch.dict(OCTDL, self.pin), self.assertRaisesRegex(ValueError, 'Converted bytes differ'):
            unpack_oct(self.root, self.archive)
        report = json.loads(next((self.root / 'output/provider_data').rglob('run.json')).read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(any((self.root / 'output').rglob('*.jpg')))

    def test_zip_paths_links_duplicates_and_ambiguity_rejected(self):
        def link(archive):
            info = zipfile.ZipInfo('link')
            info.create_system, info.external_attr = 3, (0o120777 << 16)
            archive.writestr(info, 'outside')
        extras = [lambda archive: archive.writestr('../outside.jpg', b'x'),
                  lambda archive: archive.writestr('OCTDL/./unexpected.jpg', b'x'), link,
                  lambda archive: archive.writestr('AMD/amd_0.jpg', self.payload)]
        for extra in extras:
            with self.subTest(extra=extra):
                self.make_archive(extra)
                with patch.dict(OCTDL, self.pin), self.assertRaises(ValueError):
                    unpack_oct(self.root, self.archive)
        self.assertFalse((self.root / 'output').exists())
        with self.assertWarns(UserWarning):
            self.make_archive(lambda archive: archive.writestr('OCTDL/AMD/amd_0.jpg', self.payload))
        with patch.dict(OCTDL, self.pin), self.assertRaisesRegex(ValueError, 'duplicate'):
            unpack_oct(self.root, self.archive)

    def test_zip_without_top_level_folder_is_supported(self):
        self.make_archive(prefix='')
        with patch.dict(OCTDL, self.pin):
            raw = unpack_oct(self.root, self.archive, split='test')
        self.assertEqual((raw / 'AMD/amd_1.jpg').read_bytes(), self.payload)

    def test_disk_reserve_before_unpacking(self):
        disk = type('Disk', (), {'free': RESERVE})()
        with patch.dict(OCTDL, self.pin), patch('gw.provider_data.shutil.disk_usage', return_value=disk):
            with self.assertRaisesRegex(RuntimeError, '8 GiB'):
                unpack_oct(self.root, self.archive)
        self.assertFalse(any((self.root / 'output').rglob('run.json')))

    def response(self, payload, url=None):
        stream = io.BytesIO(payload)
        stream.geturl = lambda: url or OCTDL['url']
        return stream

    def test_download_fixture_and_verified_reuse(self):
        payload = self.archive.read_bytes()
        with patch.dict(OCTDL, self.pin), patch('gw.provider_data.urllib.request.urlopen',
                                              return_value=self.response(payload)) as opened:
            downloaded = download_oct(self.root)
            self.assertEqual(downloaded.read_bytes(), payload)
            self.assertEqual(download_oct(self.root), downloaded)
            self.assertEqual(opened.call_count, 1)
            self.assertFalse(any(downloaded.parent.glob('.gw-download-*')))

    def test_download_bad_length_checksum_redirect_and_disk(self):
        for payload, url in [(b'truncated', None),
                             (self.archive.read_bytes() + b'x', None),
                             (b'x' * self.pin['bytes'], None),
                             (self.archive.read_bytes(), 'http://zenodo.org/unsecure')]:
            with self.subTest(url=url, length=len(payload)):
                with patch.dict(OCTDL, self.pin), patch('gw.provider_data.urllib.request.urlopen',
                                                      return_value=self.response(payload, url)):
                    with self.assertRaises(ValueError):
                        download_oct(self.root)
        cache = self.root / 'data/downloads/octdl-10839556'
        self.assertFalse((cache / 'OCTDL.zip').exists())
        self.assertFalse(any(cache.glob('.gw-download-*')))
        with patch.dict(OCTDL, self.pin), patch('gw.provider_data.shutil.disk_usage',
                  return_value=type('Disk', (), {'free': RESERVE})()):
            with self.assertRaisesRegex(RuntimeError, '8 GiB'):
                download_oct(self.root)

    def test_external_cache_and_output_links_rejected(self):
        outside = Path(self.temporary.name) / 'outside'
        outside.mkdir()
        (self.root / 'data/downloads').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'outside'):
            download_oct(self.root)
        (self.root / 'output').symlink_to(outside)
        with patch.dict(OCTDL, self.pin), self.assertRaisesRegex(ValueError, 'outside'):
            unpack_oct(self.root, self.archive)


if __name__ == '__main__':
    unittest.main()

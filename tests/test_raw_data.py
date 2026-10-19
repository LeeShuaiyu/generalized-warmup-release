from pathlib import Path
import csv
import hashlib
import io
import importlib.util
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
from gw.raw_data import audit, convert, decode_patch, load_sar_records, npy_bytes, source_spec


class RawDataConversion(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'project'
        self.raw = Path(self.temporary.name) / 'raw'
        self.raw.mkdir()
        (self.root / 'data/manifests').mkdir(parents=True)
        self.array = np.arange(9, dtype=np.float32).reshape(3, 3)
        self.row = {'sample_id': 'ship_positives_000000', 'class_id': '2', 'label': 'ship',
                    'split': 'test', 'group_id': 'scene', 'height': '3', 'width': '3',
                    'raw_path': 'data/ship_positives.json',
                    'data_path': 'data/arrays/ship_positives_000000.npy'}
        (self.raw / 'ship_positives.json').write_text(json.dumps([[{
            'windowfu': self.array.tolist(), 'patchfu': [[0]], 'fullname': 'scene'}]]))
        self.register('sar', [self.row], [npy_bytes(self.array)])

    def register(self, dataset, rows, payloads):
        manifest = self.root / f'data/manifests/{dataset}.csv'
        with manifest.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.item = {'manifest': str(manifest.relative_to(self.root)),
                     'semantic_sha256': manifest_sha256(manifest),
                     'files': [{'sample_id': r['sample_id'], 'path': r['data_path'], 'split': r['split'],
                                'sha256': hashlib.sha256(p).hexdigest()}
                               for r, p in zip(rows, payloads)]}
        (self.root / 'data/files.json').write_text(json.dumps({dataset: self.item}))

    def test_native_window_conversion_and_independent_output(self):
        before = (self.raw / 'ship_positives.json').read_bytes()
        output = convert(self.root, 'sar', self.raw)
        np.testing.assert_array_equal(np.load(output / 'files' / self.row['data_path']), self.array)
        report = json.loads((output / 'run.json').read_text())
        self.assertEqual(report['status'], 'complete')
        self.assertFalse(report['model_inference_executed'])
        self.assertEqual(report['samples'][0]['split'], 'test')
        self.assertEqual((self.raw / 'ship_positives.json').read_bytes(), before)
        self.assertFalse((self.root / 'data/processed').exists())
        self.assertNotEqual(convert(self.root, 'sar', self.raw), output)

    def test_wrong_provider_bytes_fail_with_record(self):
        self.array[0, 0] = 99
        (self.raw / 'ship_positives.json').write_text(json.dumps([
            {'windowfu': self.array.tolist(), 'fullname': 'scene'}]))
        with self.assertRaisesRegex(ValueError, 'Converted bytes differ'):
            convert(self.root, 'sar', self.raw)
        report = json.loads(next((self.root / 'output').rglob('run.json')).read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['samples'], [])
        self.assertFalse(any((self.root / 'output').rglob('*.npy')))

    def test_provider_group_is_checked(self):
        (self.raw / 'ship_positives.json').write_text(json.dumps([
            {'windowfu': self.array.tolist(), 'fullname': 'wrong'}]))
        with self.assertRaisesRegex(ValueError, 'acquisition group'):
            convert(self.root, 'sar', self.raw)

    def test_missing_source_creates_no_run(self):
        with self.assertRaises(FileNotFoundError):
            convert(self.root, 'sar', self.root / 'absent')
        self.assertFalse((self.root / 'output').exists())

    def test_label_change_is_rejected(self):
        row = dict(self.row, class_id='0')
        self.register('sar', [row], [npy_bytes(self.array)])
        with self.assertRaisesRegex(ValueError, 'class mapping'):
            convert(self.root, 'sar', self.raw)

    def test_duplicate_catalog_and_group_overlap_are_rejected(self):
        self.item['files'].append(self.item['files'][0])
        (self.root / 'data/files.json').write_text(json.dumps({'sar': self.item}))
        with self.assertRaisesRegex(ValueError, 'Sample identities'):
            audit(self.root, 'sar')
        row2 = dict(self.row, sample_id='ship_positives_000001', split='train', data_path='data/two.npy')
        self.register('sar', [self.row, row2], [npy_bytes(self.array)] * 2)
        with self.assertRaisesRegex(ValueError, 'multiple splits'):
            audit(self.root, 'sar')

    def test_catalog_path_or_split_change_is_rejected(self):
        self.item['files'][0]['path'] = '../unsafe.npy'
        (self.root / 'data/files.json').write_text(json.dumps({'sar': self.item}))
        with self.assertRaises(ValueError):
            convert(self.root, 'sar', self.raw)

    def test_disk_reserve(self):
        with patch('gw.raw_data.shutil.disk_usage', return_value=type('Disk', (), {'free': 8 * 1024**3})()):
            with self.assertRaisesRegex(RuntimeError, '8 GiB'):
                convert(self.root, 'sar', self.raw)

    def test_symlink_source_escape_rejected(self):
        outside = self.root / 'external.json'
        outside.write_text('[]')
        source = self.raw / 'ship_positives.json'
        source.unlink()
        source.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'outside'):
            convert(self.root, 'sar', self.raw)

    @unittest.skipUnless(importlib.util.find_spec('tifffile'), 'Install the data extra for TIFF conversion tests')
    def test_tiff_decoder_and_registered_archive_conversion(self):
        import tifffile
        stream = io.BytesIO()
        channels = np.stack([self.array, self.array + 6, self.array + 12], axis=2)
        tifffile.imwrite(stream, channels, photometric='rgb')
        payload = stream.getvalue()
        array, width, height, decoder = decode_patch(payload)
        np.testing.assert_array_equal(array, self.array + 6)
        self.assertEqual((width, height, decoder), (3, 3, 'tifffile'))
        name = 'S1A_IW_GRDH_1SDV_example.zip'
        member = 'Patch_Cal/Cargo_x1_y1.tif'
        digest = hashlib.sha1(f'{name}|{member}'.encode()).hexdigest()[:12]
        row = dict(self.row, label='Cargo', class_id='0', group_id=name,
                   sample_id=f'opensar_grd3_Cargo_{digest}', raw_path=f'data/{name}!{member}',
                   data_path=f'data/arrays/{digest}.npy')
        self.register('opensar', [row], [npy_bytes(array)])
        with zipfile.ZipFile(self.raw / name, 'w') as archive:
            archive.writestr(member, payload)
        output = convert(self.root, 'opensar', self.raw)
        np.testing.assert_array_equal(np.load(output / 'files' / row['data_path']), array)
        bad = dict(row, raw_path=f'{name}!../Patch_Cal/Cargo.tif')
        with self.assertRaises(ValueError):
            source_spec(bad, 'opensar')

    @unittest.skipUnless(importlib.util.find_spec('tifffile'), 'Install the data extra for TIFF conversion tests')
    def test_recorded_fallback_preserves_original_gray_scaling(self):
        stream = io.BytesIO()
        Image.fromarray(np.arange(9, dtype=np.uint8).reshape(3, 3)).save(stream, format='PNG')
        array, _, _, branch = decode_patch(stream.getvalue())
        np.testing.assert_array_equal(array, np.arange(9, dtype=np.float32).reshape(3, 3) / 255.0)
        self.assertTrue(branch.startswith('Pillow fallback'))

    def test_oct_copy_is_byte_exact(self):
        stream = io.BytesIO()
        Image.fromarray(np.arange(9, dtype=np.uint8).reshape(3, 3)).save(stream, format='JPEG')
        payload = stream.getvalue()
        row = dict(self.row, sample_id='amd_one', class_name='AMD', class_id='0',
                   data_path='data/octdl/images/OCTDL/AMD/amd_one.jpg')
        self.register('oct', [row], [payload])
        (self.raw / 'AMD').mkdir()
        (self.raw / 'AMD/amd_one.jpg').write_bytes(payload)
        output = convert(self.root, 'oct', self.raw)
        self.assertEqual((output / 'files' / row['data_path']).read_bytes(), payload)

    def test_frozen_manifests(self):
        root = Path(__file__).resolve().parents[1]
        expected = {'sar': 9576, 'opensar': 6301, 'oct': 2064}
        for dataset, count in expected.items():
            result = audit(root, dataset)
            self.assertEqual(result['samples'], count)
            self.assertEqual(result['group_split_overlap'], 0)


if __name__ == '__main__':
    unittest.main()

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gw.region_selection import select_sar, select_background, sar_pixels, compare_bank, _registered_inputs
from gw.distribution_fitting import rebuilt_input


class RegionSelection(unittest.TestCase):
    def test_constant_small_or_invalid_images(self):
        for array in (np.zeros((32, 32)), np.full((32, 32), .5), np.ones((8, 8))):
            self.assertIsNone(select_sar(array))
        for array in (np.zeros((256, 512)), np.full((256, 512), .5), np.zeros((32, 32))):
            self.assertIsNone(select_background(array)[0])
        for array in (np.full((16, 16), np.nan), np.ones((16, 16)) * 2, np.zeros((2, 2, 2))):
            with self.assertRaises(ValueError):
                select_sar(array)
            with self.assertRaises(ValueError):
                select_background(array)

    def test_sar_first_raster_tie_and_native_size(self):
        tile = np.tile(np.array([.1, .2, .3, .4]), (16, 4))
        chosen = select_sar(np.tile(tile, (2, 2)))
        self.assertEqual((chosen['x'], chosen['y']), (0, 0))
        self.assertEqual(chosen['score'], 0)
        self.assertEqual(select_sar(tile[:16, :16]), chosen)

    def test_oct_context_above_tissue(self):
        rng = np.random.default_rng(20260916)
        image = np.clip(.08 + rng.normal(0, .02, (256, 512)), .001, 1)
        image[145:220] = .55
        record, audit, top = select_background(image)
        self.assertEqual(audit['reason'], 'selected')
        self.assertLess(record['mean'], .20)
        self.assertLessEqual(record['y'] + 32, top[record['x'] - 16:record['x'] + 32].min() - 24)
        self.assertGreaterEqual(record['distance_to_tissue'], 40)

    def test_array_loader_keeps_original_arithmetic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'array.npy'
            original = np.random.default_rng(4).gamma(2, 2, (21, 21)).astype(np.float32)
            np.save(path, original)
            raw, gray = sar_pixels(path)
            lo, hi = np.percentile(original, [1, 99])
            expected = np.clip((original - lo) / max(hi - lo, 1e-30), 0, 1)
            np.testing.assert_array_equal(gray, expected)
            self.assertEqual(gray.dtype, expected.dtype)
            np.testing.assert_array_equal(raw, original)
            np.save(path, original.astype(complex) + 1j)
            with self.assertRaisesRegex(ValueError, 'Complex'):
                sar_pixels(path)

    def test_registered_banks_compare_all_pixels_not_just_counts(self):
        config = json.loads((ROOT / 'configs/region_selection.json').read_text())
        for item in config['datasets']:
            with np.load(ROOT / item['expected_bank']) as data:
                bank = {k: data[k].copy() for k in data.files}
            records = json.loads((ROOT / item['expected_records']).read_text())
            result = compare_bank(ROOT, item, bank, records)
            self.assertTrue(result['matches_fixed_bank'])
            bank['model_gray'][0, 0, 0] += .01
            self.assertFalse(compare_bank(ROOT, item, bank, records)['matches_fixed_bank'])

    def test_catalog_checks_before_selection(self):
        config = json.loads((ROOT / 'configs/region_selection.json').read_text())
        self.assertEqual(len(config['datasets']), 4)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'missing or inconsistent'):
                _registered_inputs(ROOT, 'sar', Path(directory))
        with self.assertRaisesRegex(ValueError, 'Unknown dataset'):
            _registered_inputs(ROOT, 'unknown')

    def test_complete_rebuilt_banks_and_saved_summary(self):
        from gw.region_report import summary
        config = json.loads((ROOT / 'configs/fit.json').read_text())
        for item in config['datasets']:
            bank, records = rebuilt_input(ROOT, item, ROOT / 'results/fit/regions' / item['id'])
            self.assertEqual(len(bank['model_gray']), len(records))
        table = summary(ROOT)
        self.assertEqual(table.Blocks.tolist(), [157, 99, 166, 543])
        self.assertEqual(table['Train images'].tolist(), [4396, 6642, 188, 1477])
        self.assertTrue((table.Matches == '✓').all())

    def test_partial_or_changed_region_run_is_rejected(self):
        import shutil
        item = json.loads((ROOT / 'configs/fit.json').read_text())['datasets'][0]
        original = ROOT / 'results/fit/regions' / item['id']
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'regions'
            shutil.copytree(original, directory)
            path = directory / 'run.json'
            state = json.loads(path.read_text())
            state['diagnostic_limit'] = 10
            path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, 'complete matching'):
                rebuilt_input(ROOT, item, directory)
            state['diagnostic_limit'] = None
            path.write_text(json.dumps(state))
            (directory / 'records.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'output changed'):
                rebuilt_input(ROOT, item, directory)


if __name__ == '__main__':
    unittest.main()

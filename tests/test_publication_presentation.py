"""Presentation cleanup must retain results and a runnable comparison path."""
from pathlib import Path
import csv
import json
import re
import unittest
import nbformat

ROOT = Path(__file__).resolve().parents[1]


class PublicationPresentation(unittest.TestCase):
    def test_noise_domain_is_retained_in_figure_data_table(self):
        nb = nbformat.read(ROOT/'notebooks/quantitative_components.ipynb', 4)
        source = next(c.source for c in nb.cells if c.cell_type == 'code'
                      and 'ece_intervals.csv' in c.source)
        self.assertIn("'Noise domain'", source)
        self.assertNotIn("ece_intervals.csv', index_col=0", source)

    def test_saved_outputs_have_no_binary_paper_status(self):
        for path in (ROOT / 'notebooks').glob('*.ipynb'):
            nb = nbformat.read(path, 4)
            for cell in nb.cells:
                for output in cell.get('outputs', []):
                    html = output.get('data', {}).get('text/html', '')
                    self.assertIsNone(re.search(r'>\s*(Different|Matches|Match|Agreement|True|False)\s*<', html), path.name)

    def test_sar_manifest_is_portable(self):
        with (ROOT / 'data/manifests/sar.csv').open() as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 9576)
        for row in rows:
            self.assertNotIn('window_path', json.loads(row['extra']))
            self.assertNotIn('/root/', json.dumps(row))

    def test_public_notebook_identity_and_reference_access(self):
        self.assertTrue((ROOT / 'notebooks/cifar100.ipynb').is_file())
        self.assertFalse((ROOT / 'notebooks/s2_cifar10.ipynb').exists())
        for name in ['main_random', 'main_pretrained', 'main_compatibility']:
            nb = nbformat.read(ROOT / f'notebooks/{name}.ipynb', 4)
            code = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
            self.assertIn('folded_comparison(', code)
        config = json.loads((ROOT / 'configs/pretrained.json').read_text())
        self.assertEqual(config['coverage']['retrained_checkpoints'], 90)
        self.assertEqual(config['coverage']['historical_checkpoints'], 30)


if __name__ == '__main__':
    unittest.main()

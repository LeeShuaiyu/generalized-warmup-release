from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from gw import distribution_fitting as fitting
from gw.supporting_reports import checked


class DistributionFitting(unittest.TestCase):
    def test_saved_refit_and_source_identity(self):
        import json
        from gw.assets import file_hash
        folder = ROOT/'results/fit/refit'
        state = json.loads(checked(ROOT,'results/fit/refit/run.json').read_text())
        rows, table = fitting.summarize(folder)
        self.assertEqual(len(rows),225)
        self.assertEqual(file_hash(checked(ROOT,'results/fit/refit/implementation.py')),state['code_sha256'])
        for relative,digest in state['input_files'].items():
            self.assertEqual(file_hash(checked(ROOT,relative)),digest)
        comparison = fitting.paper_comparison(ROOT,folder)
        self.assertEqual(len(comparison),16)
        self.assertTrue(comparison.Matches.all())
        self.assertLess(comparison['Δ NLL'].abs().max(),1e-12)
        self.assertLess(comparison['Δ KS'].abs().max(),1e-12)

    def test_banks_and_group_splits(self):
        import json
        config = json.loads(checked(ROOT, 'configs/fit.json').read_text())
        self.assertEqual(len(config['datasets']), 4)
        for item in config['datasets']:
            records = json.loads(checked(ROOT, item['records']).read_text())
            with np.load(checked(ROOT, item['bank']), allow_pickle=False) as bank:
                for coordinate in item['coordinates']:
                    a,b,meta = fitting.grouped_samples(bank[coordinate], records, 91501)
                    self.assertTrue((a>0).all() and (b>0).all())
                    self.assertEqual(meta['fit_blocks']+meta['hold_blocks'], len(records))
                    self.assertEqual(meta['fit_groups']+meta['hold_groups'], len({r['group_id'] for r in records}))
                    again = fitting.grouped_samples(bank[coordinate], records, 91501)
                    self.assertTrue(np.array_equal(a, again[0]))
                    self.assertTrue(np.array_equal(b, again[1]))

    def test_invalid_samples(self):
        bank = np.zeros((4,16,16))
        records = [{'group_id':str(i)} for i in range(4)]
        with self.assertRaises(ValueError):
            fitting.grouped_samples(bank, records, 1)
        with self.assertRaises(ValueError):
            fitting.grouped_samples(np.ones((4,16,16)), records[:3], 1)
        with self.assertRaises(ValueError):
            fitting.grouped_samples(-np.ones((4,16,16)), records, 1)

    def test_uniform_support_is_not_repaired(self):
        rows = fitting.fit_samples(np.linspace(.5,1.5,256), np.array([.2,.6,1.2,1.8]), ['Uniform'])
        self.assertEqual(rows[0]['status'], 'OUTSIDE_SUPPORT')
        self.assertIsNone(rows[0]['nll'])
        self.assertEqual(rows[0]['outside_fraction'], .5)

    def test_positive_families_and_generalized_gamma(self):
        train = np.random.default_rng(41).gamma(4,.25,2000)
        hold = np.random.default_rng(42).gamma(4,.25,500)
        rows = fitting.fit_samples(train,hold,['Gamma','Weibull','GeneralizedGamma'])
        self.assertTrue(all(r['status']=='OK' for r in rows))
        self.assertTrue(all(np.isfinite(r['nll']) for r in rows))
        self.assertEqual(rows[-1]['parameters'][2], 0)
        self.assertEqual(rows[-1]['optimizer']['starts'], 3)
        self.assertGreater(rows[-1]['optimizer']['converged'], 0)


if __name__=='__main__':
    unittest.main()

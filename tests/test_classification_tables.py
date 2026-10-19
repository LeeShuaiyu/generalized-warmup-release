from pathlib import Path
import sys
import unittest
import json
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gw import classification_report as report


class PaperTables(unittest.TestCase):
    def test_settings_html_preserves_missing_values_and_config(self):
        for experiment in ('random', 'pretrained'):
            config = report.read(ROOT, experiment)
            before = json.dumps(config, sort_keys=True)
            frame = report.settings(config)
            missing = frame[['L', 'p', 'ρ']].isna()
            html = report.settings_html(config)
            self.assertNotIn('>NaN<', html)
            self.assertEqual(html.count('>—<'), int(missing.to_numpy().sum()))
            self.assertTrue(report.settings(config).equals(frame))
            self.assertEqual(json.dumps(config, sort_keys=True), before)
            self.assertIn('Learning rate', html)
            self.assertIn('0.25', html)
            # Each scrollable seed row retains its dataset/method context.
            self.assertEqual(html.count('>Schwegmann SAR<'),
                             sum(r['dataset'] == 'Schwegmann SAR' for r in config['runs']))

    def test_saved_random_uses_model_results(self):
        config = report.read(ROOT, 'random')
        runs, mean, sd, sources = report.saved_evaluation(ROOT, config)
        expected = report.calculate(ROOT, config, ROOT / 'results/random/gpu_predictions')
        self.assertTrue(runs.equals(expected[0]))
        self.assertTrue(mean.equals(expected[1]))
        self.assertTrue(sd.equals(expected[2]))
        self.assertEqual(int(sources['Model evaluations'].sum()), 140)

    def test_saved_pretrained_discloses_complete_mixed_checkpoint_coverage(self):
        config = report.read(ROOT, 'pretrained')
        runs, _, _, sources = report.saved_evaluation(ROOT, config)
        self.assertEqual(len(runs), 120)
        self.assertEqual(int(sources['Model evaluations'].sum()), 120)
        self.assertEqual(int(sources['Runs'].sum()), 120)
        self.assertEqual(sources.loc[('OCTDL', 'Swin-Tiny'), 'Source'], 'Saved model evaluation')
        self.assertEqual(int((sources.Source == 'Replayed checkpoint evaluation').sum()), 3)

    def test_partial_unknown_or_duplicate_evaluations_fail(self):
        config = report.read(ROOT, 'random')
        cases = [[{'id': config['runs'][0]['id']}], [{'id': 'unknown'}],
                 [{'id': config['runs'][0]['id']}] * 2]
        for records in cases:
            with self.subTest(records=records), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp) / 'results/random/gpu_predictions'
                directory.mkdir(parents=True)
                (directory / 'run.json').write_text(json.dumps({'records': records}))
                with self.assertRaises(ValueError), patch.object(report, 'calculate') as calculate:
                    report.saved_evaluation(Path(tmp), config)
                calculate.assert_not_called()

    def test_random(self):
        config = report.read(ROOT,'random')
        runs,mean,sd = report.calculate(ROOT,config)
        self.assertEqual(len(runs),140)
        for name in ['main_random','s4_main']:
            self.assertTrue(report.comparison(mean,sd,report.paper(ROOT,name)).Matches.all())
        for group in dict.fromkeys((r[0],r[1]) for r in mean.index):
            self.assertLess(mean.loc[(*group,'GW'),'ECE (%)'],mean.loc[(*group,'Standard training'),'ECE (%)'])

    def test_pretrained(self):
        config = report.read(ROOT,'pretrained')
        runs,mean,sd = report.calculate(ROOT,config)
        self.assertEqual(len(runs),120)
        for name in ['main_pretrained','s5_pretrained']:
            self.assertTrue(report.comparison(mean,sd,report.paper(ROOT,name)).Matches.all())
        for group in dict.fromkeys((r[0],r[1]) for r in mean.index):
            self.assertLess(mean.loc[(*group,'GW'),'ECE (%)'],mean.loc[(*group,'Standard fine-tuning'),'ECE (%)'])

    def test_random_cuda_complete_and_differences_retained(self):
        config = report.read(ROOT, 'random')
        runs, mean, sd = report.calculate(ROOT, config, ROOT / 'results/random/gpu_predictions')
        self.assertEqual(len(runs), 140)
        comparison = report.comparison(mean, sd, report.paper(ROOT, 'main_random'))
        self.assertFalse(comparison.Matches.all())
        self.assertLess(comparison['Δ mean'].abs().max(), 0.05)
        for group in dict.fromkeys((r[0], r[1]) for r in mean.index):
            self.assertLess(mean.loc[(*group, 'GW'), 'ECE (%)'], mean.loc[(*group, 'Standard training'), 'ECE (%)'])

    def test_pretrained_oct_cuda(self):
        config = report.select_dataset(report.read(ROOT, 'pretrained'), 'OCTDL')
        runs, mean, sd = report.calculate(ROOT, config, ROOT / 'results/pretrained/gpu_predictions')
        self.assertEqual(len(runs), 30)
        for name in ['main_pretrained', 's5_pretrained']:
            reference = report.select_reference(report.paper(ROOT, name), config)
            self.assertTrue(report.comparison(mean, sd, reference).Matches.all())


if __name__=='__main__':
    unittest.main()

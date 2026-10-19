from pathlib import Path
import json
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gw import posthoc


class Posthoc(unittest.TestCase):
    def test_validation_inputs(self):
        for arm in ['erm','gw']:
            for seed in range(20260921,20260924):
                for split,samples in [('validation',295),('test',292)]:
                    path = posthoc.checked(ROOT,f'artifacts/posthoc/{arm}_{seed}/{split}.npz')
                    z,y = posthoc.load_predictions(path,7)
                    self.assertEqual(z.shape,(samples,7))
                    self.assertEqual(y.shape,(samples,))

    def test_input_validation(self):
        for z,y in [(np.zeros((3,2)),np.array([0.,0.,1.])),
                    (np.zeros((3,2)),np.array([0,0,2])),
                    (np.full((3,2),np.nan),np.zeros(3,dtype=int))]:
            with self.assertRaises(ValueError):
                posthoc.fit_temperature_scaling(z,y)
        with self.assertRaises(ValueError):
            posthoc.fit_vector_scaling(np.zeros((3,2)),np.zeros(3,dtype=int),-1)

    def test_fitting_and_transform(self):
        z,y = posthoc.load_predictions(ROOT/'artifacts/posthoc/erm_20260921/validation.npz',7)
        ts = posthoc.fit_temperature_scaling(z,y)
        vs = posthoc.fit_vector_scaling(z,y)
        self.assertGreater(ts.temperature,0)
        self.assertLess(ts.validation_nll,posthoc._nll(z,y))
        self.assertLess(vs.validation_nll,posthoc._nll(z,y))
        self.assertAlmostEqual(float(vs.bias.mean()),0,places=12)
        np.testing.assert_array_equal(ts.transform(z).argmax(1),z.argmax(1))

    def test_saved_refit(self):
        from gw.classification_report import comparison, paper
        from gw.presentation import sha256
        directory = ROOT/'results/posthoc_compat/refit'
        path = posthoc.checked(ROOT,'results/posthoc_compat/refit/run.json')
        state = json.loads(path.read_text())
        self.assertEqual(state['operation'],'calibration_refit_from_saved_logits')
        self.assertEqual(len(state['fitted_parameters']),6)
        self.assertEqual(len(state['inputs']),12)
        self.assertEqual(sha256(directory/'implementation.py'),state['code_sha256'])
        for item in state['inputs']:
            self.assertEqual(sha256(posthoc.checked(ROOT,item['path'])),item['sha256'])
        runs,mean,sd = posthoc.summarize(ROOT,directory)
        self.assertEqual(len(runs),15)
        self.assertTrue(comparison(mean,sd,paper(ROOT,'s6_posthoc')).Matches.all())
        self.assertEqual(sha256(directory/'config.json'),state['config_sha256'])

    def test_model_derived_sources_and_differences(self):
        from gw.classification_report import comparison, paper
        from gw.posthoc_inference import prediction_source
        from gw.presentation import sha256
        config=json.loads(posthoc.checked(ROOT,'configs/posthoc_refit.json').read_text())
        for base in config['runs']:
            self.assertEqual(base['settings']['image_size'],224)
            self.assertEqual(base['settings']['batch_size'],64)
            for split,relative in [('validation','results/posthoc_compat/model_validation'),('test','results/random/gpu_predictions')]:
                source=prediction_source(ROOT,ROOT/relative,base,split)
                self.assertEqual(source['checkpoint_sha256'],base['checkpoint_sha256'])
            altered=dict(base,checkpoint_sha256='0'*64)
            with self.assertRaises(ValueError):
                prediction_source(ROOT,ROOT/'results/posthoc_compat/model_validation',altered,'validation')
        directory=ROOT/'results/posthoc_compat/model_refit'
        state=json.loads(posthoc.checked(ROOT,'results/posthoc_compat/model_refit/run.json').read_text())
        self.assertEqual(state['operation'],'calibration_refit_from_model_logits')
        self.assertEqual(sha256(directory/'implementation.py'),state['code_sha256'])
        self.assertEqual(sha256(directory/'config.json'),state['config_sha256'])
        _,mean,sd=posthoc.summarize(ROOT,directory)
        check=comparison(mean,sd,paper(ROOT,'s6_posthoc'))
        self.assertEqual(int(check.Matches.sum()),19)
        self.assertAlmostEqual(mean.loc[('OCTDL','ResNet-18','GW'),'ECE (%)'],7.690073,places=5)
        self.assertTrue(check.loc[('OCTDL','ResNet-18','GW + VS')].Matches.all())

    def test_main_and_supplementary_share_posthoc_results(self):
        from gw import compatibility_report
        _,main_mean,main_sd=compatibility_report.calculate(ROOT)
        _,mean,sd=posthoc.summarize(ROOT,ROOT/'results/posthoc_compat/model_refit_fp32')
        np.testing.assert_array_equal(main_mean.loc[mean.index].values,mean.values)
        np.testing.assert_array_equal(main_sd.loc[sd.index].values,sd.values)
        self.assertEqual(len(main_mean),10)


if __name__ == '__main__':
    unittest.main()

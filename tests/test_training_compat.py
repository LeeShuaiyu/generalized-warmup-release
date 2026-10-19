from pathlib import Path
import json
import tempfile
import unittest
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gw import classification_report as report, training_compat_report
from gw.classification_data import manifest_sha256
from gw.presentation import sha256


class CompatibilityBindings(unittest.TestCase):
    def test_displayed_training_parameters(self):
        frame = training_compat_report.settings(ROOT)
        self.assertEqual(len(frame), 15)
        for (method, seed), row in frame.iterrows():
            self.assertEqual(row['Warm-up epochs'], 5 if method.startswith('GW') else 0)
            self.assertEqual(row['Label smoothing'], .1 if 'smoothing' in method else 0.)
            self.assertEqual(row['MMCE strength'], .1 if 'MMCE' in method else 0.)
            if method.startswith('GW'):
                self.assertEqual((row['L'], row['p'], row['ρ']), (4., .25, .25))
            else:
                self.assertTrue(np.isnan(row['L']))
            self.assertEqual(row['Selection'], 'Validation NLL')

    def test_original_training_settings(self):
        config = report.read(ROOT,'training_compat')
        self.assertEqual(len(config['runs']),15)
        for record in config['runs']:
            c = record['training_settings']
            self.assertEqual(c['seed'],record['seed'])
            self.assertEqual((c['architecture'],c['image_size'],c['batch_size']),('resnet18_gray',128,128))
            self.assertEqual(c['checkpoint_metric'],'val_nll')
            self.assertEqual(record['split_sha256'],manifest_sha256(ROOT/'data/manifests/opensar.csv'))
            self.assertEqual(record['dataset_key'],'opensar')
            self.assertEqual(c['mmce_alpha'],.1 if 'MMCE' in record['method'] else 0.)
            self.assertEqual(c['label_smoothing'],.1 if 'smoothing' in record['method'] else 0.)

    def test_saved_checkpoint_coverage(self):
        directory=ROOT/'results/training_compat/gpu_predictions'
        if not (directory/'run.json').exists():
            self.skipTest('New model evaluations not imported yet')
        state=json.loads((directory/'run.json').read_text())
        self.assertEqual(state['status'],'complete')
        self.assertEqual(len(state['records']),9)
        self.assertEqual(set(state['selected_ids']),{f'{i:03d}' for i in range(3,12)})
        config=report.read(ROOT,'training_compat')
        for identity in state['records']:
            record=next(r for r in config['runs'] if r['id']==identity['id'])
            self.assertEqual(identity['checkpoint_sha256'],record['checkpoint_sha256'])
            self.assertEqual(sha256(directory/f'{record["id"]}.npz'),identity['predictions_sha256'])
        runs,mean,sd=training_compat_report.calculate(ROOT)
        self.assertEqual(len(runs),15)
        coverage=training_compat_report.coverage(ROOT)
        self.assertEqual(coverage.loc['GW','Checkpoint inference'],'3/3')
        self.assertEqual(coverage.loc['MMCE','Checkpoint inference'],'3/3')
        self.assertTrue(np.isfinite(mean.to_numpy()).all())

    def test_full_previously_completed_evaluations_are_distinct_from_new_inference(self):
        config=report.read(ROOT,'training_compat')
        directory=ROOT/config['saved_evaluation_directory']
        state=json.loads((directory/'run.json').read_text())
        self.assertEqual(len(state['records']),15)
        self.assertFalse(state['new_model_inference_executed_during_import'])
        self.assertEqual(set(state['selected_ids']),{f'{i:03d}' for i in range(15)})
        self.assertEqual(config['coverage'],{'mapped':15,'total':15})
        self.assertTrue(all('checkpoint' in r for r in config['runs']))

    def test_changed_evaluation_identity_rejected(self):
        path=ROOT/'results/training_compat/gpu_predictions/run.json'
        if not path.exists():
            self.skipTest('New model evaluations not imported yet')
        state=json.loads(path.read_text())
        state['records'][0]['checkpoint_sha256']='0'*64
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            (directory/'run.json').write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError,'Checkpoint evaluation identity'):
                training_compat_report.calculate(ROOT,directory)


class MMCEOperations(unittest.TestCase):
    def test_penalty_gradient_and_corner_cases(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch is not installed')
        from gw.mmce import penalty
        x=torch.tensor([[2.,.1,-1.],[.3,1.7,-.8],[.1,-.2,2.1],[1.3,.1,-.5]],dtype=torch.double,requires_grad=True)
        y=torch.tensor([0,2,2,1])
        self.assertTrue(torch.autograd.gradcheck(lambda z:penalty(z,y),(x,),eps=1e-6,atol=1e-4))
        idx=torch.tensor([3,0,2,1])
        self.assertTrue(torch.allclose(penalty(x,y),penalty(x[idx],y[idx])))
        for labels in (x.argmax(1),(x.argmax(1)+1)%3):
            z=x.detach().clone().requires_grad_()
            loss=penalty(z,labels); loss.backward()
            self.assertEqual(float(loss),0.)
            self.assertTrue(torch.isfinite(z.grad).all())

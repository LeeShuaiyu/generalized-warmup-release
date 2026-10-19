from pathlib import Path
import json
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gw import classification_report as report, cifar_training
from gw.presentation import sha256


class CifarResults(unittest.TestCase):
    def test_complete_cuda_tables(self):
        for name,n,count in [('cifar3',6,8),('cifar10',20,10)]:
            config = report.read(ROOT,name)
            directory = ROOT/f'results/{name}/gpu_predictions'
            receipt = json.loads((directory/'run.json').read_text())
            self.assertEqual(receipt['status'],'complete')
            self.assertEqual(receipt['device'],'cuda')
            self.assertEqual(len(receipt['records']),n)
            for run in config['runs']:
                self.assertEqual(run['settings']['warmup_steps'],0 if run['settings']['method']=='erm' else 1950)
            self.assertEqual(sha256(directory/'config_original.json'),receipt['config_sha256'])
            runs,mean,sd = report.calculate(ROOT,config,directory)
            self.assertEqual(len(runs),n)
            check = report.comparison(mean,sd,report.paper(ROOT,'s2_'+name))
            self.assertEqual(len(check),count)
            self.assertTrue(check.Matches.all())

    def test_shared_seed_predictions(self):
        ten = report.read(ROOT,'cifar10')
        lookup = {(r['seed'],r['settings']['method']):r for r in ten['runs']}
        for r in report.read(ROOT,'cifar3')['runs']:
            source = lookup[(r['seed'],r['settings']['method'])]
            self.assertEqual(r['checkpoint_sha256'],source['checkpoint_sha256'])
            self.assertEqual(sha256(ROOT/f'results/cifar3/gpu_predictions/{r["id"]}.npz'),
                             sha256(ROOT/f'results/cifar10/gpu_predictions/{source["id"]}.npz'))

    def test_training_protocol(self):
        config = report.read(ROOT,'cifar_training')
        self.assertEqual(len(config['runs']),20)
        for run in config['runs']:
            c = run['settings']
            self.assertEqual(c['epochs'],30)
            self.assertEqual(c['train_samples_per_class'],80)
            self.assertEqual(c['warmup_steps'],0 if c['method']=='erm' else 1950)
            self.assertEqual(c['checkpoint_metric'],'final_epoch')
            self.assertEqual(run['split_sha256'],'efeb8f4ad8e49d772d543bee73f8e9408e6e3537fd249f5f124f6b54383d5d71')
        targets = np.repeat(np.arange(100),500)
        train,validation,identity = cifar_training.split_indices(targets,config['runs'][0]['settings'])
        self.assertEqual(identity,config['runs'][0]['split_sha256'])
        self.assertEqual(len(train),8000)
        self.assertEqual(len(validation),5000)
        self.assertEqual(len(set(train)&set(validation)),0)
        np.testing.assert_array_equal(np.bincount(targets[train]),np.full(100,80))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None,'Torch is not installed')
class CifarModel(unittest.TestCase):
    def test_initial_state_identity_and_output(self):
        from gw.cifar_model import LinearNetwork
        r = report.read(ROOT,'cifar_training')['runs'][0]
        np.random.seed(r['seed']); torch.manual_seed(r['seed'])
        model = LinearNetwork().eval()
        self.assertEqual(cifar_training.state_hash(model.state_dict()),r['initial_state_sha256'])
        with torch.inference_mode():
            values = model(torch.zeros(2,3,32,32))
        self.assertEqual(tuple(values.shape),(2,100))
        self.assertTrue(torch.isfinite(values).all())


if __name__ == '__main__':
    unittest.main()

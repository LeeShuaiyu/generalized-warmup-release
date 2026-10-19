from pathlib import Path
import json
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gw import classification_report as report,noise_core as core
from gw.presentation import sha256


class NoiseConfiguration(unittest.TestCase):
    @unittest.skipUnless((ROOT/'results/noise/gpu_predictions/run.json').exists(),'New model inference not imported')
    def test_new_model_outputs_and_interaction(self):
        from gw import supporting_reports
        directory=ROOT/'results/noise/gpu_predictions'
        receipt=json.loads((directory/'run.json').read_text())
        self.assertEqual(receipt['status'],'complete')
        self.assertEqual(receipt['device'],'cuda')
        self.assertEqual(len(receipt['records']),40)
        self.assertEqual(sha256(directory/'config_original.json'),receipt['config_sha256'])
        raw,summary=supporting_reports.interaction(ROOT,directory)
        self.assertEqual(len(raw),10)
        self.assertGreater(summary.loc['Gaussian domain','Mean (pp)'],0)
        self.assertLess(summary.loc['LogGamma domain','Mean (pp)'],0)
        self.assertLess(summary.loc['Interaction','CI high'],0)
        self.assertEqual(summary.loc['Interaction','Negative seeds'],7)

    def test_run_identity_and_weights(self):
        config=report.read(ROOT,'noise')
        manifest=json.loads((ROOT/'artifacts/manifest.json').read_text())
        self.assertEqual(len(config['runs']),40)
        seen=set()
        for r in config['runs']:
            c=r['settings']
            self.assertEqual(r['samples'],4050*5)
            self.assertEqual(c['alpha'],.75)
            self.assertEqual(c['looks'],4)
            self.assertEqual(c['noise_seed'],20260715)
            self.assertIn(c['method'],['gaussian','loggamma'])
            key=(r['seed'],c['noise_domain'],c['method'])
            self.assertNotIn(key,seen);seen.add(key)
            name=Path(r['checkpoint']).parent.name
            self.assertEqual(manifest['checkpoints']['noise/'+name]['sha256'],r['checkpoint_sha256'])

    def test_fixed_data_coverage(self):
        context=json.loads((ROOT/'data/eurosat/split_manifest.json').read_text())
        groups=[context[s+'_indices'] for s in ['train','val','test']]
        self.assertEqual([len(g) for g in groups],[18900,4050,4050])
        all_indices=[i for group in groups for i in group]
        self.assertEqual(sorted(all_indices),list(range(27000)))
        files=json.loads((ROOT/'data/eurosat/image_files.json').read_text())
        self.assertEqual(len({r['path'] for r in files}),27000)
        labels=[]
        for label,name in enumerate(context['class_names']):
            labels.extend([label]*sum(Path(r['path']).parent.name==name for r in files))
        actual=core.stratified_split(labels,20260629,[.7,.15,.15])
        self.assertEqual(actual,groups)


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Torch is not installed')
class NoiseOperations(unittest.TestCase):
    def test_initial_state_and_paired_sampling(self):
        r=report.read(ROOT,'noise')['runs'][0]
        core.seed_all(r['seed'],True)
        model=core.build_model(10)
        self.assertEqual(core.state_digest(model.state_dict()),r['initial_state_sha256'])
        def generators():
            return dict(base=torch.Generator().manual_seed(101),structure=torch.Generator().manual_seed(102))
        a,y=core.make_warmup_batch('gaussian',2,10,.75,4.,generators(),'cpu')
        b,z,fields=core.make_warmup_batch('loggamma',2,10,.75,4.,generators(),'cpu',return_audit=True)
        self.assertTrue(torch.equal(y,z))
        self.assertTrue(torch.equal(a,fields['base']))
        self.assertTrue(torch.equal(b,(a+.75*fields['probe'])/1.25))
        self.assertTrue(torch.equal(y,torch.full((2,10),.1)))
        self.assertEqual(tuple(b.shape),(2,3,64,64))

    def test_target_is_shared_across_training_seeds(self):
        a=core.target_noise_field(168,3,20260715,'loggamma_noisy',4.)
        b=core.target_noise_field(168,3,20260715,'loggamma_noisy',4.)
        self.assertTrue(torch.equal(a,b))
        c=core.target_noise_field(168,4,20260715,'loggamma_noisy',4.)
        self.assertFalse(torch.equal(a,c))
        with self.assertRaises(ValueError):
            core.target_noise_field(168,3,20260715,'unknown',4.)


if __name__=='__main__':
    unittest.main()

from pathlib import Path
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gw.ablation_training import validate_settings, Inputs


class Settings(unittest.TestCase):
    def test_all_53_condition_settings(self):
        runs=json.loads((ROOT/'configs/ablation_training.json').read_text())['runs']
        self.assertEqual(len(runs),53)
        self.assertEqual(sum(r['experiment']=='s4_family' for r in runs),20)
        self.assertEqual(sum(r['experiment']=='s7_length' for r in runs),33)
        for run in runs:
            validate_settings(run)
            self.assertEqual(run['supervised_epochs'],50)
            self.assertEqual(run['batch_size'],128)
            self.assertEqual(len(run['initial_state_sha256']),64)
            self.assertEqual(len(run['foundation_sha256']),5)
            self.assertEqual(run['learning_rate'],.001)
        self.assertEqual({r['shape'] for r in runs if r['condition']=='gw'},{2.373805936071085})

    def test_invalid_settings(self):
        run=json.loads((ROOT/'configs/ablation_training.json').read_text())['runs'][0]
        for changes in [dict(shape=0),dict(p=2),dict(rho=float('nan')),
                        dict(batch_size=0),dict(warmup_epochs=0),dict(normalize_real_inputs=True)]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                validate_settings(dict(run,**changes))

    def test_input_normalization_separate(self):
        base=[(np.array([0.,.5,1.]),2)]
        setting=dict(normalization=dict(mean=.1,std=.2),normalize_real_inputs=False)
        self.assertIs(Inputs(base,setting)[0][0],base[0][0])
        np.testing.assert_array_equal(Inputs(base,dict(setting,normalize_real_inputs=True))[0][0],
                                      (base[0][0]-.1)/.2)


@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch is not installed')
class Operations(unittest.TestCase):
    def test_independent_streams_and_repeatability(self):
        import torch
        from gw.ablation_training import sample,generators
        runs=json.loads((ROOT/'configs/ablation_training.json').read_text())['runs']
        for run in [runs[0],runs[5],runs[10],runs[15],runs[23]]:
            torch.manual_seed(3);before=torch.get_rng_state().clone()
            streams=generators(9,'cpu',torch)
            a=sample(run,(4,1,8,8),streams,'cpu',torch)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            b=sample(run,(4,1,8,8),generators(9,'cpu',torch),'cpu',torch)
            self.assertTrue(torch.equal(a,b));self.assertTrue(torch.isfinite(a).all())
            c=sample(run,(4,1,8,8),streams,'cpu',torch)
            self.assertFalse(torch.equal(a,c))

    def test_training_and_reload_without_test_inputs(self):
        import torch
        from gw.ablation_training import train,state_hash,seed_all
        from gw.classification_data import manifest_sha256,load_manifest,ManifestDataset
        from gw.training_stages import collect_logits
        from gw.assets import file_hash
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.fc=torch.nn.Linear(64,3)
            def forward(self,x):return self.fc(x.flatten(1))
        all_runs=json.loads((ROOT/'configs/ablation_training.json').read_text())['runs']
        for condition in [all_runs[0],next(r for r in all_runs if r['condition']=='w00'),
                          next(r for r in all_runs if r['condition']=='w01')]:
            run=dict(condition,seed=102,batch_size=4,image_size=8,supervised_epochs=1,
                     warmup_epochs=0 if condition['family']=='erm' else 1)
            with self.subTest(experiment=run['experiment'],family=run['family']),tempfile.TemporaryDirectory(prefix='gw-ablation-',dir=ROOT.parent) as temp:
                root=Path(temp);(root/'data/manifests').mkdir(parents=True)
                images=root/'data/processed/opensar';images.mkdir(parents=True)
                rows=[];files=[]
                for split,count in [('train',9),('val',6),('test',3)]:
                    for i in range(count):
                        name=f'{split}-{i}.npy'
                        row=dict(sample_id=f'{split}-{i}',class_id=str(i%3),split=split,data_path=name,data_type='npy_float32')
                        rows.append(row)
                        if split!='test':np.save(images/name,np.arange(64,dtype=np.float32).reshape(8,8)+i)
                        files.append(dict(sample_id=row['sample_id'],split=split,path=name,
                            sha256=file_hash(images/name) if split!='test' else '0'*64))
                manifest=root/'data/manifests/opensarship2.csv'
                with manifest.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                run['manifest_sha256']=manifest_sha256(manifest)
                (root/'data/files.json').write_text(json.dumps(dict(opensar=dict(manifest='data/manifests/opensarship2.csv',
                    semantic_sha256=run['manifest_sha256'],files=files))))
                seed_all(run['seed'],torch);run['initial_state_sha256']=state_hash(Model())
                with patch('gw.ablation_training.build_model',lambda *args:Model()):
                    output=train(root,run,device='cpu',limits=dict(train=9,val=6))
                metadata=json.loads((output/'run.json').read_text())
                self.assertEqual(metadata['status'],'complete');self.assertFalse(metadata['test_constructed'])
                self.assertEqual(metadata['split_counts'],dict(train=9,val=6))
                model=Model();model.load_state_dict(torch.load(output/'best_model.pt',weights_only=True))
                data=Inputs(ManifestDataset(load_manifest(manifest),'val',image_size=8,data_root=images),run)
                logits,labels=collect_logits(model,torch.utils.data.DataLoader(data,batch_size=4),'cpu',torch)
                with np.load(output/'validation_predictions.npz') as saved:
                    np.testing.assert_array_equal(logits,saved['logits']);np.testing.assert_array_equal(labels,saved['labels'])
                history=json.loads((output/'history.json').read_text())
                self.assertEqual(sum(r['stage']=='warmup' for r in history),run['warmup_epochs'])


if __name__=='__main__':unittest.main()

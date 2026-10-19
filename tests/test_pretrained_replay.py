from pathlib import Path
import json
import tempfile
import unittest
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gw import classification_report as report
from gw.classification_inference import evaluation_dataset


class ReplayBindings(unittest.TestCase):
    def test_complete_mapping_retains_two_checkpoint_kinds(self):
        c=report.read(ROOT,'pretrained')
        self.assertEqual(len(c['runs']),120)
        self.assertTrue(all('checkpoint' in r for r in c['runs']))
        replays=[r for r in c['runs'] if r.get('checkpoint_kind')=='retrained_checkpoint']
        self.assertEqual(len(replays),90)
        self.assertTrue(all(r['inference_protocol']=='legacy_all_splits' for r in replays))
        self.assertEqual({r['dataset_key'] for r in replays},{'sar','opensar','fusar'})
        runs,mean,sd,sources=report.saved_evaluation(ROOT,c)
        self.assertEqual(len(runs),120)
        self.assertEqual(int(sources['Model evaluations'].sum()),120)
        self.assertEqual(set(sources.Source),{'Replayed checkpoint evaluation','Saved model evaluation'})
        self.assertTrue(np.isfinite(mean.to_numpy()).all())

    def test_original_predictions_are_not_replaced_with_replay_results(self):
        c=report.read(ROOT,'pretrained')
        archived,_,_=report.calculate(ROOT,c)
        replay,_,_,_=report.saved_evaluation(ROOT,c)
        self.assertEqual(len(archived),len(replay))
        self.assertGreater(np.abs(archived['ECE (%)']-replay['ECE (%)']).max(),.01)

    def test_legacy_input_order_and_quantization(self):
        try:import torch
        except ImportError:self.skipTest('Torch is not installed')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'data/processed/sar';base.mkdir(parents=True)
            rows=[]
            for i,split in enumerate(['test','train','val']):
                name=f'{i}.npy';np.save(base/name,np.arange(9,dtype=np.float32).reshape(3,3)+i)
                rows.append(dict(sample_id=str(i),split=split,class_id=str(i),data_path=name,data_type='npy_float32'))
            record=dict(dataset_key='sar',inference_protocol='legacy_all_splits',settings={'image_size':3})
            dataset,mask=evaluation_dataset(root,record,rows)
            self.assertEqual([int(dataset[i][1]) for i in range(3)],[1,2,0])
            self.assertEqual(mask.tolist(),[False,False,True])
            x,_=dataset[0]
            self.assertTrue(torch.allclose(x*255,torch.round(x*255)))

    def test_unknown_input_protocol_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unknown checkpoint inference protocol'):
            evaluation_dataset(ROOT,dict(dataset_key='sar',inference_protocol='invented',settings={'image_size':3}),[])

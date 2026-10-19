from pathlib import Path
import hashlib
import json
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gw.classification_training import resolve_training_run

class ReplayTrainingBindings(unittest.TestCase):
    def test_all_pretrained_runs_resolve(self):
        paper=json.loads((ROOT/'configs/pretrained.json').read_text())
        for row in paper['runs']:
            bound=resolve_training_run(ROOT,'pretrained',row)
            self.assertEqual(bound['id'],row['id'])
            self.assertEqual(bound['training_settings']['seed'],row['seed'])

    def test_replay_source_and_configuration_evidence(self):
        config=json.loads((ROOT/'configs/pretrained_replay_training.json').read_text())
        legacy=json.loads((ROOT/'configs/pretrained_training.json').read_text())
        self.assertEqual(len(legacy['runs']),88)
        self.assertEqual({r['id'] for r in config['runs']},{'000','010'})
        for row in config['runs']:
            base=ROOT/'results/pretrained/replay_training_records'/row['id']
            for name,key in [('torch_train_config.json','replay_config_sha256'),('run_manifest.json','replay_manifest_sha256')]:
                self.assertEqual(hashlib.sha256((base/name).read_bytes()).hexdigest(),row[key])
            manifest=json.loads((base/'run_manifest.json').read_text())
            self.assertEqual({Path(p).name:h for p,h in manifest['script_hashes'].items()},row['source_code_sha256'])
            actual=json.loads((base/'torch_train_config.json').read_text())
            self.assertEqual(actual['seed'],row['training_settings']['seed'])
            self.assertEqual(row['training_record_kind'],'replayed_training')

    def test_replay_binding_does_not_claim_missing_original(self):
        paper=json.loads((ROOT/'configs/pretrained.json').read_text())
        run=dict(paper['runs'][0],checkpoint_kind='original_checkpoint')
        with self.assertRaisesRegex(ValueError,'No verified training entry'):
            resolve_training_run(ROOT,'pretrained',run)

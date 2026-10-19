from pathlib import Path
import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


@unittest.skipUnless(importlib.util.find_spec('nbformat'),'Notebook tools are not installed')
class Portals(unittest.TestCase):
    def test_default_display_does_not_train(self):
        import nbformat
        import pandas as pd
        for name in ['s4_family','s7_length']:
            nb=nbformat.read(ROOT/f'notebooks/{name}.ipynb',4)
            headers=[i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and c.source.startswith('## Training')]
            self.assertEqual(len(headers),1)
            cell=nb.cells[headers[0]+1]
            self.assertEqual(cell.cell_type,'code')
            ast.parse(cell.source)
            config=json.loads((ROOT/f'configs/{name}.json').read_text())
            shown=[]
            with patch('gw.ablation_training.train',side_effect=AssertionError('Default notebook started training')):
                scope=dict(ROOT=ROOT,config=config,display=shown.append)
                exec(cell.source,scope)
            self.assertFalse(scope['RUN_TRAINING'])
            self.assertEqual(scope['selected_run']['experiment'],name)
            self.assertEqual(len(shown),1)
            self.assertEqual(shown[0].data['Seed'].iloc[0],scope['selected_run']['seed'])
            self.assertFalse(nb.metadata.gw.training_executed)
            self.assertFalse(nb.metadata.gw.inference_executed)

    def test_enabled_cell_calls_automatic_device_and_reports_one_seed(self):
        import nbformat
        for name in ['s4_family','s7_length']:
            nb=nbformat.read(ROOT/f'notebooks/{name}.ipynb',4)
            cell=next(nb.cells[i+1] for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and c.source.startswith('## Training'))
            with tempfile.TemporaryDirectory(prefix='gw-portal-') as temp:
                output=Path(temp)
                (output/'run.json').write_text(json.dumps(dict(best_epoch=3,validation=dict(ece_15=.052,nll=.81))))
                config=json.loads((ROOT/f'configs/{name}.json').read_text());shown=[]
                with patch('gw.ablation_training.train',return_value=output) as train,patch('builtins.print'):
                    scope=dict(ROOT=ROOT,config=config,display=shown.append)
                    exec(cell.source.replace('RUN_TRAINING = False','RUN_TRAINING = True'),scope)
                train.assert_called_once_with(ROOT,scope['selected_run'],device='auto')
                self.assertEqual(len(shown),2)
                self.assertEqual(len(shown[1].data),1)
                self.assertAlmostEqual(shown[1].data['ECE (%)'].iloc[0],5.2)
                self.assertEqual(shown[1].data['Selected epoch'].iloc[0],3)


if __name__=='__main__':unittest.main()

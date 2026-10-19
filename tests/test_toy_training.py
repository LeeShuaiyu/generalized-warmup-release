from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gw import toy_training as training
from gw.toy_model import init_model, train_random_labels


class ToyTraining(unittest.TestCase):
    def test_original_initializations(self):
        config = training.configuration(ROOT)
        for seed in config['seeds']:
            actual = training.load_model(ROOT / f'artifacts/toy1/{seed}/initial.npz')
            self.assertTrue(training.same_model(init_model(seed), actual))

    def test_short_training_saved_reloaded_and_measured(self):
        with tempfile.TemporaryDirectory(prefix='gw-toy-') as temp:
            destination = Path(temp) / 'run'
            output = training.train(ROOT, 100, steps=2, destination=destination)
            run = json.loads((output / 'run.json').read_text())
            self.assertEqual(run['status'], 'complete')
            self.assertEqual(run['snapshot_steps'], [0, 2])
            self.assertFalse(run['full_budget'])
            expected = init_model(100)
            config = training.configuration(ROOT)['warmup']
            train_random_labels(expected, seed=101, steps=2, batch_size=config['batch_size'],
                                learning_rate=config['learning_rate'])
            actual = training.load_model(output / 'step_00002.npz')
            self.assertTrue(training.same_model(expected, actual))
            metrics = np.genfromtxt(output / 'metrics.csv', delimiter=',', names=True)
            values = training.measure(actual, training.configuration(ROOT)['evaluation'])
            self.assertEqual(metrics['confidence_bias'][-1], values['confidence_bias'])
            self.assertEqual(metrics['mean_log_response'][-1], values['mean_log_response'])
            with self.assertRaises(FileExistsError):
                training.train(ROOT, 100, steps=0, destination=destination)

    def test_invalid_settings_and_disk_reserve_before_output(self):
        for seed, steps in [(99, 2), (True, 2), (100, -1), (100, True), (100, 12001)]:
            with self.assertRaises(ValueError):
                training.train(ROOT, seed, steps=steps)
        with tempfile.TemporaryDirectory(prefix='gw-toy-') as temp:
            destination = Path(temp) / 'run'
            with patch('gw.toy_training.shutil.disk_usage') as disk:
                disk.return_value.free = 8*1024**3
                with self.assertRaises(RuntimeError):
                    training.train(ROOT, 100, steps=0, destination=destination)
            self.assertFalse(destination.exists())

    def test_zero_steps_and_invalid_arrays(self):
        with tempfile.TemporaryDirectory(prefix='gw-toy-') as temp:
            destination = Path(temp) / 'run'
            training.train(ROOT, 129, steps=0, destination=destination)
            self.assertTrue(training.same_model(init_model(129), training.load_model(destination/'step_00000.npz')))
            invalid = Path(temp)/'invalid.npz'
            np.savez(invalid, weight_0=np.ones((2, 10)))
            with self.assertRaises(ValueError):
                training.load_model(invalid)

    def test_notebook_default_does_not_train(self):
        import nbformat
        nb=nbformat.read(ROOT/'notebooks/s2_initialization.ipynb',4)
        indices=[i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and c.source.startswith('## Training')]
        self.assertEqual(len(indices),1)
        shown=[]
        with patch('gw.toy_training.train',side_effect=AssertionError('Default started training')):
            scope=dict(ROOT=ROOT,display=shown.append)
            exec(nb.cells[indices[0]+1].source,scope)
        self.assertFalse(scope['RUN_TRAINING'])
        self.assertEqual(len(shown),1)
        self.assertEqual(shown[0].data['Updates'].iloc[0],12000)
        self.assertFalse(nb.metadata.gw.training_executed)

    def test_enabled_notebook_calls_backend(self):
        import nbformat
        nb=nbformat.read(ROOT/'notebooks/s2_initialization.ipynb',4)
        with tempfile.TemporaryDirectory(prefix='gw-toy-portal-') as temp:
            output=Path(temp)
            (output/'metrics.csv').write_text('step,confidence_bias,mean_log_response\n0,0.3,0.2\n12000,0.01,-4.1\n')
            shown=[]
            with patch('gw.toy_training.train',return_value=output) as backend,patch('builtins.print'):
                exec(nb.cells[-1].source.replace('RUN_TRAINING = False','RUN_TRAINING = True'),dict(ROOT=ROOT,display=shown.append))
            backend.assert_called_once_with(ROOT,100,steps=12000)
            self.assertEqual(len(shown),2)
            self.assertEqual(shown[1].data['Updates'].tolist(),[0,12000])


if __name__ == '__main__':
    unittest.main()

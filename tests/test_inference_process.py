from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch,MagicMock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gw import inference_process


class InferenceProcess(unittest.TestCase):
    def test_default_uses_current_runtime(self):
        config={'experiment_id':'pretrained','runs':[{'id':'000'}]}
        with patch.dict(os.environ,{},clear=True),patch('gw.classification_inference.run') as run:
            inference_process.run(ROOT,config,device='cpu',run_ids=['000'])
            run.assert_called_once_with(ROOT.resolve(),config,device='cpu',run_ids=['000'])

    def test_separate_runtime_keeps_identity_and_returns_checked_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'output/pretrained/new';target.mkdir(parents=True)
            (target/'run.json').write_text('{}')
            process=MagicMock(stdout=iter(['Inference 1/1\n',str(target)+'\n']))
            process.wait.return_value=0
            with patch.dict(os.environ,{'GW_INFERENCE_PYTHON':'/runtime/bin/python'}),patch('subprocess.Popen',return_value=process) as popen:
                result=inference_process.run(root,dict(experiment_id='pretrained',runs=[{'id':'000'}]),run_ids=['000'])
                self.assertEqual(result,target)
                command=popen.call_args.args[0]
                self.assertIn('--no-report',command)
                self.assertEqual(command[-1],'000')
                self.assertNotIn('GW_INFERENCE_PYTHON',popen.call_args.kwargs['env'])

    def test_invalid_identity_fails_before_subprocess(self):
        with patch.dict(os.environ,{'GW_INFERENCE_PYTHON':'/runtime/bin/python'}),patch('subprocess.Popen') as popen:
            with self.assertRaises(ValueError):
                inference_process.run(ROOT,dict(experiment_id='pretrained',runs=[{'id':'000'}]),run_ids=['001'])
            popen.assert_not_called()

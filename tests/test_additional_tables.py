from pathlib import Path
import sys
import unittest
import nbformat
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gw import classification_report as report
from gw import supporting_reports as support


class AdditionalTables(unittest.TestCase):
    def test_prediction_tables(self):
        for name,experiment in [('s2_cifar3','cifar3'),('s2_cifar10','cifar10'),
                                ('s3_noise','noise'),('s6_posthoc','posthoc_compat')]:
            _,mean,sd=report.calculate(ROOT,report.read(ROOT,experiment))
            self.assertTrue(report.comparison(mean,sd,report.paper(ROOT,name)).Matches.all(),name)

    def test_training_difference_retained(self):
        _,mean,sd=report.calculate(ROOT,report.read(ROOT,'training_compat'))
        check=report.comparison(mean,sd,report.paper(ROOT,'s6_training'))
        difference=check.loc[~check.Matches]
        self.assertEqual(len(difference),1)
        self.assertEqual(difference.index[0][2],'GW + MMCE')
        self.assertAlmostEqual(difference.iloc[0]['Δ mean'],.030637,places=5)

    def test_interaction(self):
        _,table=support.interaction(ROOT)
        self.assertAlmostEqual(table.loc['Interaction','Mean (pp)'],-.1119,places=4)
        self.assertEqual(table.loc['Interaction','Negative seeds'],7)
        self.assertLess(table.loc['Interaction','CI high'],0)

    def test_model_states(self):
        rows=support.toy_inference(ROOT)
        self.assertEqual(len(rows),60)
        self.assertLess(rows['Max difference'].max(),1e-9)

    def test_fit_identity(self):
        rows=support.fit(ROOT)
        self.assertEqual(len(rows),16)
        self.assertAlmostEqual(rows.loc[('OpenSARShip2','Generalized Gamma'),'NLL ↓'],.8367,places=4)
        self.assertTrue(np.isinf(rows.xs('Uniform',level='Family')['NLL ↓']).all())

    def test_ten_table_portals(self):
        notebooks=[p for p in (ROOT/'notebooks').glob('*.ipynb') if p.stem != 'paper_figures']
        self.assertEqual(len(notebooks),10)
        for path in notebooks:
            notebook=nbformat.read(path,as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type=='code':
                    self.assertIsNotNone(cell.execution_count,path.name)
                    self.assertFalse(any(o.output_type=='error' for o in cell.outputs),path.name)


if __name__=='__main__':
    unittest.main()

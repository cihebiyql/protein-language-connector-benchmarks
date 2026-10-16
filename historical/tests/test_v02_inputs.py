import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bench_v02.build_inputs import model_input

class InputIsolationTests(unittest.TestCase):
    def test_affinity_gold_source_and_raw_columns_never_enter_model_inputs(self):
        row={'sample_id':'transport','panel_id':'A1','task':'affinity','peptide_sequence':'ACDEFG',
             'target_sequence':'ACDEFGHIK','target_sequence_role':'reference',
             'measurement_type':'KD','relation':'=','kd_molar':1e-7,'pkd':7,'label_exact':True,
             'chemical_state':{'topology':'Linear','explicitly_representable':True},
             'experimental_context':{'ph':7.4,'leaked_KD':999},'source_doi':'forbidden_source',
             'raw_metadata':{'Affinity':'forbidden_target'},'eligibility':{'broad_reference_target':True}}
        inputs=model_input(row,'A1')['model_inputs']
        encoded=json.dumps(inputs)
        self.assertNotIn('forbidden',encoded)
        self.assertNotIn('leaked_KD',encoded)
        self.assertNotIn('pkd',inputs)
        self.assertNotIn('kd_molar',inputs)

    def test_rt_std_count_and_upstream_predictions_are_not_features(self):
        row={'sample_id':'transport','panel_id':'A2','task':'retention_time','peptide_sequence':'ACDEFG',
             'peptidoform':'ACDEFG|','modifications':[],'condition_id':'c','rt_value':17.6,'rt_unit':'min',
             'eligibility':{'measurement_valid':True},
             'rawmetadata':{'condition_metadata':{'pH':9},'rt_std':3,'rt_count':99,'predicted_tr':8}}
        inputs=model_input(row,'A2')['model_inputs']
        self.assertEqual(inputs['condition_metadata'],{'pH':9})
        self.assertNotIn('rt_value',inputs)
        self.assertNotIn('predicted_tr',json.dumps(inputs))

if __name__=='__main__':
    unittest.main()

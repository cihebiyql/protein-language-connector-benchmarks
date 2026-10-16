import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bench_v02.contracts import ContractError,numeric_generation,record_reasons,validate_run_manifest
from bench_v02.evaluate import evaluate


class V02ContractTests(unittest.TestCase):
    def manifest(self,channel='generation'):
        return {'benchmark_version':'0.2','channel':channel,'model_id':'fixture_only',
                'seed':17,'checkpoint_sha256':'a'*64,'role':'controlled_interface',
                'source_checkpoints_by_panel':{p:'a'*64 for p in ('A1','A2','B1','B2')}}

    def test_generation_requires_real_finite_json_not_legacy_logits(self):
        self.assertEqual(numeric_generation('{"pKD":7.25}','pKD'),7.25)
        self.assertEqual(numeric_generation('```json\n{"pKD":7}\n```','pKD'),7)
        for text in ('Prediction is 7.2','{"pKD":NaN}','{"pKD":true}','{"pKD":"7.2"}'):
            with self.assertRaises(ContractError):
                numeric_generation(text,'pKD')
        m=self.manifest();m['legacy_alias_logits']=True
        with self.assertRaises(ContractError):
            validate_run_manifest(m,'A1',diagnostic=True)

    def test_single_checkpoint_and_pre_answer_boundaries(self):
        m=self.manifest();m['source_checkpoints_by_panel']['B2']='b'*64
        with self.assertRaises(ContractError):
            validate_run_manifest(m,'B2',diagnostic=True)
        m=self.manifest('readout_head');m['state_boundary']='after_teacher_forced_answer';m['included_answer_tokens']=True
        with self.assertRaises(ContractError):
            validate_run_manifest(m,'A1',diagnostic=True)

    def test_formal_evaluation_requires_phase_zero_lock(self):
        with self.assertRaises(ContractError):
            validate_run_manifest(self.manifest(),'A1')

    def test_all_target_label_uses_count_toward_budget(self):
        m=self.manifest()
        m['adaptation']={'label_budget':2,'label_ids':{'fit':['a'],'calibration':['b'],'validation':['c']},
                         'test_ids':['z'],'parent_source_checkpoint_sha256':'a'*64}
        with self.assertRaises(ContractError):
            validate_run_manifest(m,'A2',diagnostic=True)
        m['adaptation']['label_budget']=3
        validate_run_manifest(m,'A2',diagnostic=True)
        m['adaptation']['test_ids']=['c']
        with self.assertRaises(ContractError):
            validate_run_manifest(m,'A2',diagnostic=True)

    def test_invalid_generation_does_not_disappear_from_primary_score(self):
        gold=[{'sample_id':'x','pkd':7.0,'relation':'=','label_exact':True,'groups':{'target_uniprot':'P'}},
              {'sample_id':'y','pkd':5.0,'relation':'=','label_exact':True,'groups':{'target_uniprot':'P'}}]
        preds=[{'sample_id':'x','panel_id':'A1','channel':'generation','checkpoint_sha256':'a'*64,'raw_text':'{"pKD":7.0}'}]
        r=evaluate(gold,preds,self.manifest(),'A1',diagnostic=True)
        self.assertEqual(r['output_validity'],.5)
        self.assertFalse(r['main_table_eligible'])
        self.assertEqual(r['status'],'incomplete_invalid_outputs')
        self.assertEqual(r['metrics']['mae'],0)
        self.assertEqual(r['metric_scope'],'conditional_on_valid_outputs_only_not_primary_score')

    def test_b1_anonymous_candidates_are_mapped_only_in_evaluator(self):
        gold=[{'sample_id':'x','class_id':'heldout_class'}]
        preds=[{'sample_id':'x','panel_id':'B1','channel':'generation','checkpoint_sha256':'a'*64,
                'raw_text':'{"ranked_candidate_ids":["candidate_1","candidate_0"]}'}]
        r=evaluate(gold,preds,self.manifest(),'B1',diagnostic=True,
                   candidate_to_class={'candidate_0':'seen_class','candidate_1':'heldout_class'})
        self.assertEqual(r['metrics']['class_macro_mrr'],1)
        self.assertFalse(r['main_table_eligible'])

    def test_rt_does_not_pool_conditions_or_accept_unscaled_readout(self):
        gold=[{'sample_id':'x','rt_value':2.0,'rt_unit':'min','condition_id':'pH2'},
              {'sample_id':'y','rt_value':100.0,'rt_unit':'min','condition_id':'pH9'}]
        m=self.manifest('predictor')
        preds=[{'sample_id':r['sample_id'],'panel_id':'A2','channel':'predictor','checkpoint_sha256':'a'*64,
                'prediction':{'retention_time':r['rt_value'],'unit':'min'}} for r in gold]
        r=evaluate(gold,preds,m,'A2',diagnostic=True)
        self.assertFalse(r['metrics']['pooled'])
        self.assertEqual(len(r['metrics']['strata']),2)
        preds[0]['prediction']['unit']='z_score'
        self.assertEqual(evaluate(gold,preds,m,'A2',diagnostic=True)['n_valid'],1)

    def test_opaque_target_and_ic50_are_not_kd_regression_records(self):
        record={'sample_id':'x','panel_id':'A1','peptide_sequence':'ACDEFG','target_id':'T001',
                'measurement_type':'IC50','relation':'=','label_exact':True,'kd_molar':1e-6,'pkd':6,
                'eligibility':{'broad_reference_target':True}}
        reasons=record_reasons(record,'A1')
        self.assertIn('real_target_sequence_required_opaque_id_insufficient',reasons)
        self.assertIn('endpoint_is_not_KD',reasons)

    def test_missing_model_or_checkpoint_identity_is_rejected(self):
        for field in ('checkpoint_sha256','model_id','seed','role'):
            manifest=self.manifest();manifest.pop(field)
            with self.assertRaises(ContractError):
                validate_run_manifest(manifest,'A1',diagnostic=True)

    def test_empty_gold_cannot_make_a_complete_score(self):
        with self.assertRaises(ContractError):
            evaluate([],[],self.manifest(),'A1',diagnostic=True)

    def test_declared_run_checkpoint_must_equal_multitask_source(self):
        manifest=self.manifest();manifest['checkpoint_sha256']='b'*64
        with self.assertRaises(ContractError):
            validate_run_manifest(manifest,'A1',diagnostic=True)

    def test_formal_gold_and_candidate_catalog_are_independently_checked(self):
        # A local temporary fixture is not an actual benchmark run or result.
        with tempfile.TemporaryDirectory(prefix='v02_contract_fixture_') as directory:
            tmp=Path(directory)
            checkpoint=tmp/'fixture.weights';checkpoint.write_bytes(b'unit fixture, not trained weights')
            m=self.manifest();digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            m['checkpoint_sha256']=digest;m['checkpoint_path']=str(checkpoint)
            m['source_checkpoints_by_panel']={p:digest for p in ('A1','A2','B1','B2')}
            lock={'status':'frozen','external_experiments_excluded':True,'homology_split_verified':True,
                  'class_holdout_verified':True,'ready_panels':['A1','B1']}
            lock_path=tmp/'fixture_lock.json'
            protocol_path=tmp/'fixture_protocol.json'
            protocol_path.write_text('{"version":"0.2","scope":"unit fixture"}',encoding='utf-8')
            gold_path=tmp/'fixture_gold.jsonl'
            def freeze():
                lock_path.write_text(json.dumps(lock),encoding='utf-8')
                m['benchmark_lock_sha256']=hashlib.sha256(lock_path.read_bytes()).hexdigest()
            def score(records,predictions,panel,**kwargs):
                gold_path.write_text('\n'.join(json.dumps(r) for r in records)+'\n',encoding='utf-8')
                lock['gold_sha256']=hashlib.sha256(gold_path.read_bytes()).hexdigest()
                lock['protocol_sha256']=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
                freeze()
                return evaluate(records,predictions,m,panel,benchmark_lock_path=lock_path,
                                protocol_path=protocol_path,gold_path=gold_path,**kwargs)
            freeze()
            invalid=[{'sample_id':'x','panel_id':'A1','peptide_sequence':'ACDEFG','target_id':'opaque',
                      'measurement_type':'IC50','pkd':6.0,'relation':'=','label_exact':True,'split':'test'}]
            with self.assertRaises(ContractError):
                score(invalid,[],'A1')
            gold=[{'sample_id':'x','panel_id':'B1','segment_sequence':'ACDEFG','class_id':'unknown_class',
                   'class_description_no_rule':'function','molecule_scope':'local_functional_region_in_parent_protein',
                   'eligibility':{'sequence_verified':True,'no_rule_prompt':True},'split':'test'}]
            mapping_path=tmp/'fixture_catalog.jsonl'
            catalog={'candidate_id':'candidate_0','class_id':'real_class','eligible':True,'split':'test'}
            mapping_path.write_text(json.dumps(catalog)+'\n',encoding='utf-8')
            lock.update({'candidate_map_sha256':hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
                         'candidate_ids':['candidate_0']});freeze()
            with self.assertRaises(ContractError):
                score(gold,[],'B1',candidate_to_class={'invented':'unknown_class'},candidate_map_path=mapping_path)
            gold[0]['class_id']='real_class'
            prediction={'sample_id':'x','panel_id':'B1','channel':'generation','checkpoint_sha256':digest,
                        'raw_text':'{"ranked_candidate_ids":["candidate_0"]}'}
            result=score(gold,[prediction],'B1',candidate_to_class={'candidate_0':'real_class'},candidate_map_path=mapping_path)
            self.assertEqual(result['metrics']['class_macro_mrr'],1)
            gold[0]['split']='train'
            with self.assertRaises(ContractError):
                score(gold,[prediction],'B1',candidate_to_class={'candidate_0':'real_class'},candidate_map_path=mapping_path)

if __name__=='__main__':
    unittest.main()

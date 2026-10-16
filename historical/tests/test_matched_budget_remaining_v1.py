"""CPU-only continuation routing and source-admission negative fixtures."""
import ast
import copy
import importlib.util
import json
from datetime import datetime,timedelta,timezone
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from matched_budget_20261016 import execute_remaining_v1 as remaining
from matched_budget_20261016 import summarize_remaining_v1 as report


class SourceTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('CPU_fixture_source_setup',ROOT/'tests/test_matched_budget_batched_supervision.py')
        self.fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.fixtures)
        self.fixture=self.fixtures.ReuseAdmissionTests();self.fixture.setUp()
        self.suite=self.fixture.root;self.protocol=self.fixture.protocol;self.protocol_sha=self.fixture.protocol_sha;self.manifest=self.fixture.manifest
        self.cases=remaining.trial_matrix(self.protocol)[:2]
        self.second=self.suite/remaining.case_id(self.cases[1])
        shutil.copytree(self.fixture.source,self.second)
        remaining.write(self.second/'case.json',self.cases[1])
        actual=remaining.load(self.second/'artifacts/result.json')
        actual['registration']['configuration']['message_input']='neutral'
        remaining.write(self.second/'artifacts/result.json',actual)
        self.audit_patch=patch.object(remaining,'source_auditor',return_value=self.fixtures.supervisor);self.audit_patch.start()

    def tearDown(self): self.audit_patch.stop();self.fixture.tearDown()

    def test_exact_first_two_are_sources_not_new_training(self):
        plan=remaining.admit_original_sources(self.suite,self.protocol,self.protocol_sha,self.manifest)
        self.assertEqual(list(plan),[remaining.case_id(case) for case in self.cases])
        self.assertTrue(all(item['source_training_optimizer_steps']==1548 and item['new_optimizer_steps']==0 for item in plan.values()))
        remaining.verify_reused_evidence(plan,{})

    def test_partial_second_source_rejected(self):
        path=self.second/'artifacts/result.json';actual=remaining.load(path);actual['optimizer_steps']=1547;remaining.write(path,actual)
        with self.assertRaisesRegex(ValueError,'Partial V1 training'):
            remaining.admit_original_sources(self.suite,self.protocol,self.protocol_sha,self.manifest)

    def test_changed_second_checkpoint_SHA_rejected(self):
        (self.second/'artifacts/best_reader.pt').write_text('CPU deliberately changed checkpoint')
        with self.assertRaisesRegex(ValueError,'checkpoint identity'):
            remaining.admit_original_sources(self.suite,self.protocol,self.protocol_sha,self.manifest)


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.protocol=remaining.load(ROOT/'configs/matched_budget_20261016.frozen.json');self.protocol_sha='a'*64
        self.manifest={'datasets':{},'projection':{'seed':314159},'initializations':{}}
        self.manifest_path=self.root/'manifest.json';remaining.write(self.manifest_path,self.manifest)
        self.environment=self.root/'current_environment_freeze.txt';self.environment.write_text('CPU environment fixture')
        self.cases=remaining.canary_matrix(self.protocol)
        for case in self.cases:
            job=self.root/remaining.case_id(case);(job/'artifacts').mkdir(parents=True)
            contract={'matched_budget_registration':{'protocol_sha256':self.protocol_sha,'cache_manifest_path':str(self.manifest_path),
                'cache_manifest_sha256':remaining.sha(self.manifest_path)},'execution_scope':{'intended_level':'smoke'},
                'environment':{'lockfile_sha256':remaining.sha(self.environment),'environment_path':sys.prefix},
                'assets':{'code':[{'path':str(ROOT/'scripts/matched_budget_20261016/run_trial.py'),
                                  'sha256':remaining.sha(ROOT/'scripts/matched_budget_20261016/run_trial.py')}]}}
            for name,value in [('task-contract.json',contract),('case.json',case),
                ('result-record.json',{'reproduction_level':'smoke','lifecycle_status':'succeeded','CPU_fixture_only':True}),
                ('execution-receipt.json',{'exit_code':0,'process_group_absent':True,'termination_reason':None,'CPU_fixture_only':True}),
                ('artifacts/result.json',{'status':'passed','optimizer_steps':1,'scientific_trial_count':0,'test_examples_evaluated':0,
                    'sample_split':'train','train_generation_metrics':{'n_total':4},'CPU_fixture_only':True}),
                ('resource_probe.json',{'CPU_fixture_only':True}),('numerical_policy_actual.json',{
                    **{key:self.protocol['numerical_policy'][key] for key in ('float32_matmul_precision','cuda_matmul_allow_tf32',
                        'cudnn_allow_tf32','cudnn_benchmark','cudnn_deterministic','deterministic_algorithms','deterministic_warn_only')},
                    'applied_before_model_instantiation':True,'process_environment':{key:self.protocol['numerical_policy'][key] for key in
                        ('CUDA_LAUNCH_BLOCKING','TORCH_SHOW_CPP_STACKTRACES','CUBLAS_WORKSPACE_CONFIG')},'CPU_fixture_only':True})]:
                remaining.write(job/name,value)
        self.patch=patch.object(remaining,'validator',return_value=SimpleNamespace(validate=Mock(return_value={'valid_for_acceptance':True,'CPU_fixture_only':True})))
        self.patch.start()

    def tearDown(self): self.patch.stop();self.temp.cleanup()

    def test_original_six_are_reused_without_new_pid_or_canary_execution(self):
        values=remaining.admit_original_qualifications(self.root,self.protocol,self.protocol_sha,self.manifest,self.environment)
        self.assertEqual(len(values),6)
        self.assertTrue(all(value['new_execution_performed'] is False and value['new_pid'] is None for value in values.values()))
        remaining.verify_reused_evidence({},values)

    def test_changed_worker_or_cache_rejected(self):
        job=self.root/remaining.case_id(self.cases[0]);path=job/'task-contract.json';contract=remaining.load(path)
        contract['assets']['code'][0]['sha256']='0'*64;remaining.write(path,contract)
        with self.assertRaisesRegex(ValueError,'worker differs'):
            remaining.admit_original_qualifications(self.root,self.protocol,self.protocol_sha,self.manifest,self.environment)

    def test_failed_or_heldout_qualification_rejected(self):
        path=self.root/remaining.case_id(self.cases[0])/'artifacts/result.json';actual=remaining.load(path)
        actual['test_examples_evaluated']=1;remaining.write(path,actual)
        with self.assertRaisesRegex(ValueError,'TRAIN-only V1 canary'):
            remaining.admit_original_qualifications(self.root,self.protocol,self.protocol_sha,self.manifest,self.environment)

    def test_changed_current_environment_or_historical_numerical_flags_rejected(self):
        self.environment.write_text('changed current environment')
        with self.assertRaisesRegex(ValueError,'pip freeze/environment'):
            remaining.admit_original_qualifications(self.root,self.protocol,self.protocol_sha,self.manifest,self.environment)
        self.environment.write_text('CPU environment fixture')
        path=self.root/remaining.case_id(self.cases[0])/'numerical_policy_actual.json';value=remaining.load(path)
        value['cuda_matmul_allow_tf32']=True;remaining.write(path,value)
        with self.assertRaisesRegex(ValueError,'numerical flag'):
            remaining.admit_original_qualifications(self.root,self.protocol,self.protocol_sha,self.manifest,self.environment)


class RoutingTests(unittest.TestCase):
    def test_real_standard_contract_and_terminal_CPU_failure_archive(self):
        spec=importlib.util.spec_from_file_location('CPU_remaining_physical_fixture',ROOT/'tests/test_matched_budget_supervision.py')
        fixture_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture_module)
        fixture=fixture_module.ContractFixtureTests();fixture.setUp()
        try:
            remaining.write(fixture.suite/'original_source_receipts.json',{'CPU_fixture_only':True})
            remaining.write(fixture.suite/'reused_qualification_receipts.json',{'CPU_fixture_only':True})
            command=[sys.executable,'-c','pass']
            with patch.object(remaining,'ROOT',fixture.root):
                contract=remaining.build_contract(fixture.case,fixture.job,fixture.suite,fixture.protocol_path,fixture.protocol,
                    remaining.sha(fixture.protocol_path),fixture.manifest,remaining.sha(fixture.manifest),fixture.probe,30,command,'CPU fixture only')
                self.assertEqual(contract['assets']['datasets'][0]['version'],'a1_locked_v2')
                self.assertEqual(contract['run_id'],'v1_continuation__'+fixture.job.name)
                remaining.write(fixture.job/'task-contract.json',contract)
                checked=remaining.validator().validate(contract,contract_bytes=(fixture.job/'task-contract.json').read_bytes(),check_files=True,contract_directory=fixture.job)
                self.assertTrue(checked['valid_for_execution'],checked)
                remaining.write(fixture.job/'contract-validation.json',checked)
                receipt=remaining.supervise(command,fixture.job,30,1024**3,datetime.now(timezone.utc)+timedelta(minutes=1))
                self.assertEqual(receipt['exit_code'],0);self.assertTrue(receipt['process_group_absent'])
                self.assertFalse(remaining.archive(fixture.job,contract,receipt))
                result=remaining.load(fixture.job/'result-record.json')
                self.assertEqual(result['run_id'],contract['run_id'])
                self.assertTrue(remaining.load(fixture.job/'result-validation.json')['valid_for_acceptance'])
        finally: fixture.tearDown()

    def test_worker_is_byte_identical_original_V1_and_remaining_order_is_fixed(self):
        original=ROOT/'scripts/matched_budget_20261016/execute_suite.py';new=Path(remaining.__file__)
        def function(path,name):
            text=path.read_text(encoding='utf8');node=next(node for node in ast.parse(text).body if isinstance(node,ast.FunctionDef) and node.name==name)
            return ast.get_source_segment(text,node)
        self.assertEqual(function(original,'worker'),function(new,'worker'))
        protocol=remaining.load(ROOT/'configs/matched_budget_20261016.frozen.json')
        cases=remaining.trial_matrix(protocol)
        self.assertEqual(len(cases[2:]),34)
        self.assertEqual(cases[2]['placement'],'mid');self.assertEqual(cases[2]['message_input'],'real')

    def test_summary_rewrites_temporary_references_to_original_physical_source(self):
        self.assertEqual(report.rewrite_paths({'path':'/tmp/read/case/artifacts/run.jsonl'},
            {'/tmp/read/case':'/real/source/case'}),{'path':'/real/source/case/artifacts/run.jsonl'})

    def test_complete_source_raw_population_is_audited_and_temp_paths_removed(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);source=root/'original';source.mkdir();suite=root/'remaining';suite.mkdir()
            protocol=remaining.load(ROOT/'configs/matched_budget_20261016.frozen.json')
            rows=[dict(sample_id=str(i),split='test',group_id='g',pkd=float(i+1),groups={'target_sequence_sha256':'t'},experimental_context={}) for i in range(2)]
            for scenario in protocol['scenarios']:
                base=root/protocol['data_root']/scenario;base.mkdir(parents=True)
                (base/'samples.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf8')
                protocol['scenarios'][scenario].update(test=2,samples_sha256=remaining.sha(base/'samples.jsonl'))
            protocol_path=root/'protocol.json';remaining.write(protocol_path,protocol)
            plan={};first_predictions=None;first_path=None
            legacy=report.original.frozen_scoring()
            for index,case in enumerate(remaining.trial_matrix(protocol)[:2]):
                job=source/remaining.case_id(case);(job/'artifacts').mkdir(parents=True)
                texts=['{"pKD":1.5}','invalid' if index==0 else '{"pKD":2.5}']
                predictions=[dict(sample_id=r['sample_id'],split='test',gold_pkd=r['pkd'],raw_text=text,
                                  parse=legacy.parse_answer(text)) for r,text in zip(rows,texts)]
                pred_path=job/'artifacts/test_predictions_correct.jsonl';pred_path.write_text('\n'.join(json.dumps(p) for p in predictions),encoding='utf8')
                actual={'status':'complete','actual_epochs':3,'optimizer_steps':1548,'best_epoch':1,
                        'test_generation':{'correct':legacy.generation_metrics(rows,predictions)}}
                remaining.write(job/'artifacts/result.json',actual)
                remaining.write(job/'execution-receipt.json',{'exit_code':0,'process_group_absent':True,'CPU_fixture_only':True})
                remaining.write(job/'result-validation.json',{'valid_for_acceptance':True,'CPU_fixture_only':True})
                plan[remaining.case_id(case)]={'source_job':str(job),'evidence_pins':{
                    str(job/'artifacts/result.json'):remaining.sha(job/'artifacts/result.json')}}
                if index==0: first_predictions,first_path=predictions,pred_path
            remaining.write(suite/'original_source_receipts.json',plan)
            with patch.object(report.original,'ROOT',root):
                result=report.summarize(suite,source,protocol_path)
                self.assertEqual(result['n_original_completed_source_cases'],2)
                self.assertEqual(result['n_new_completed_training_cases'],0)
                self.assertFalse(result['cases'][0]['new_execution_performed'])
                self.assertNotIn('original_V1_36_read_view_',json.dumps(result))
                first_path.write_text(json.dumps(first_predictions[0]),encoding='utf8')
                result=report.summarize(suite,source,protocol_path)
                self.assertEqual(result['n_original_completed_source_cases'],1)
                self.assertEqual(result['cases'][0]['status'],'invalid_or_incomplete')


if __name__=='__main__': unittest.main()

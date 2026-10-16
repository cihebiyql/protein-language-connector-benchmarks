"""CPU fixtures only: no real GPU readiness or scientific scores are asserted."""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from matched_budget_20261016 import execute_suite as supervisor
from matched_budget_20261016 import summarize as summary


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.protocol=supervisor.load(ROOT/'configs/matched_budget_20261016.protocol.json')

    def test_exact_predeclared_seed_first_order_and_36_unique_cases(self):
        cases=supervisor.trial_matrix(self.protocol)
        self.assertEqual(len(cases),36)
        self.assertEqual(len({supervisor.case_id(x) for x in cases}),36)
        self.assertEqual([x['seed'] for x in cases[:8]],[17]*8)
        self.assertEqual([x['scenario'] for x in cases[:8]],['new_peptide_known_target']*4+['new_peptide_new_target']*4)
        self.assertEqual([x['message_input'] for x in cases[:4]],['real','neutral','real','neutral'])
        self.assertTrue(all(x['gate_init']=='near_unit' for x in cases[24:30]))
        self.assertTrue(all(x['sender_mode']=='global16' for x in cases[30:]))

    def test_train_only_canary_covers_each_scenario_location_and_gate(self):
        cases=supervisor.canary_matrix(self.protocol)
        self.assertEqual(len(cases),6)
        self.assertTrue(all(x['phase']=='canary' and x['seed']==17 for x in cases))
        self.assertEqual({(x['placement'],x['gate_init']) for x in cases},{('early','zero'),('mid','zero'),('mid','near_unit')})

    def test_seed_aggregate_does_not_average_success_only(self):
        result=summary.seed_aggregate({17:1.,42:2.,2026:None})
        self.assertEqual(result['coverage'],'2/3')
        self.assertIsNone(result['mean']);self.assertIsNone(result['sample_SD'])
        self.assertEqual(summary.seed_aggregate({17:1.,42:2.,2026:3.})['sample_SD'],1.)

    def test_strict_numerical_policy_is_applied_without_CUDA_and_no_fallback(self):
        fake=SimpleNamespace(set_num_threads=Mock(),set_float32_matmul_precision=Mock(),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                cudnn=SimpleNamespace(allow_tf32=True,benchmark=True,deterministic=False)),
            use_deterministic_algorithms=Mock(),get_float32_matmul_precision=lambda:'highest',
            are_deterministic_algorithms_enabled=lambda:True,is_deterministic_algorithms_warn_only_enabled=lambda:False)
        env=supervisor.runtime_environment(self.protocol,123)
        self.assertEqual(env['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
        self.assertEqual(env['TORCH_SHOW_CPP_STACKTRACES'],'1')
        with patch.dict(os.environ,env):
            receipt=supervisor.apply_numerical_policy(fake,self.protocol['numerical_policy'])
        fake.use_deterministic_algorithms.assert_called_once_with(True,warn_only=False)
        self.assertFalse(fake.backends.cuda.matmul.allow_tf32)
        self.assertFalse(fake.backends.cudnn.allow_tf32)
        self.assertFalse(fake.backends.cudnn.benchmark)
        self.assertTrue(fake.backends.cudnn.deterministic)
        self.assertTrue(receipt['applied_before_model_instantiation'])
        fake.use_deterministic_algorithms.side_effect=RuntimeError('fixture unsupported deterministic kernel')
        with self.assertRaisesRegex(RuntimeError,'unsupported deterministic kernel'):
            supervisor.apply_numerical_policy(fake,self.protocol['numerical_policy'])

    def test_generation_failure_has_no_full_MAE(self):
        metrics=dict(n_total=2,n_valid=1,n_failed=1,parse_valid_rate=.5,
                     full_sample_mae=1.,regression_valid_only={'mae':1.})
        with self.assertRaisesRegex(ValueError,'Full MAE'):
            summary.metrics_summary(metrics,2)
        metrics['full_sample_mae']=None
        self.assertIsNone(summary.metrics_summary(metrics,2)['full_sample_mae'])
        metrics['n_total']=True
        with self.assertRaises(ValueError): summary.metrics_summary(metrics,2)

    def test_component_bootstrap_keeps_correlated_measurements_together(self):
        rows=[{'group_id':'same'} for _ in range(3)]
        result=summary.group_bootstrap(rows,[-1,0,4],draws=20)
        self.assertEqual((result['estimate'],result['lower'],result['upper']),(1,1,1))
        self.assertEqual(result['n_source_family_components'],1)
        with self.assertRaises(ValueError): summary.group_bootstrap(rows,[1])

    def test_contrasts_exactly_match_frozen_names_and_zero_sum_weights(self):
        contrasts=summary.fixed_contrasts()
        self.assertEqual(set(contrasts),{x['name'] for x in self.protocol['paired_contrasts_per_scenario']})
        self.assertTrue(all(sum(w for _,w in terms)==0 for terms in contrasts.values()))
        self.assertEqual(contrasts['depth_content_difference_in_differences'],[
            (('early','real','balanced','zero'),1),(('mid','real','balanced','zero'),-1),
            (('early','neutral','balanced','zero'),-1),(('mid','neutral','balanced','zero'),1)])

    def test_summary_keeps_partial_seed_and_checks_raw_predictions(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);suite=root/'suite';suite.mkdir()
            protocol=copy.deepcopy(self.protocol)
            for scenario in protocol['scenarios']:
                path=root/protocol['data_root']/scenario;path.mkdir(parents=True)
                rows=[dict(sample_id='x',split='test',group_id='g',pkd=1.,groups={'target_sequence_sha256':'t'},experimental_context={}),
                      dict(sample_id='y',split='test',group_id='g',pkd=2.,groups={'target_sequence_sha256':'t'},experimental_context={})]
                (path/'samples.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
                protocol['scenarios'][scenario]['test']=2
                protocol['scenarios'][scenario]['samples_sha256']=supervisor.sha(path/'samples.jsonl')
            protocol_path=root/'protocol.json';supervisor.write(protocol_path,protocol)
            case=supervisor.trial_matrix(protocol)[0];job=suite/supervisor.case_id(case);(job/'artifacts').mkdir(parents=True)
            legacy=summary.frozen_scoring()
            predictions=[dict(sample_id='x',split='test',raw_text='{"pKD":1.5}',parse=legacy.parse_answer('{"pKD":1.5}'),gold_pkd=1.),
                         dict(sample_id='y',split='test',raw_text='{"pKD":3}',parse=legacy.parse_answer('{"pKD":3}'),gold_pkd=2.)]
            (job/'artifacts/test_predictions_correct.jsonl').write_text('\n'.join(json.dumps(p) for p in predictions),encoding='utf-8')
            actual=dict(status='complete',actual_epochs=3,optimizer_steps=1548,best_epoch=2,
                test_generation={'correct':legacy.generation_metrics(rows,predictions)},
                C1={'n_test':2,'n_matched':0,'G_all_test':.25,'P_H_status':'undefined_zero_independent_donors',
                    'means_all_test':{'correct_original_gold_mean_token_nll':1.,'off_original_gold_mean_token_nll':1.25},
                    'token_breakdown':{'correct':{'numeric':{'loss_sum':2.,'count':2,'nll_nats_per_token':1.,'contribution_nats_per_full_token':.2}}}})
            supervisor.write(job/'artifacts/result.json',actual)
            supervisor.write(job/'execution-receipt.json',dict(exit_code=0,process_group_absent=True))
            supervisor.write(job/'result-validation.json',dict(valid_for_acceptance=True,CPU_fixture_only=True))
            with patch.object(summary,'ROOT',root):
                result=summary.summarize(suite,protocol_path)
                self.assertEqual(result['n_complete'],1)
                self.assertEqual(result['seed_summaries'][0]['coverage'],'1/3')
                self.assertIsNone(result['seed_summaries'][0]['mean'])
                self.assertTrue(all(c['mean'] is None for c in result['paired_contrasts']))
                mechanism=next(x for x in result['mechanism_seed_summaries'] if x['diagnostic']=='C1.G_all_test')
                self.assertEqual(mechanism['coverage'],'1/3')
                self.assertIsNone(mechanism['mean'])
                self.assertEqual(mechanism['per_seed']['17'],.25)
                self.assertEqual(result['cases'][0]['metrics']['full_sample_mae'],.75)
                actual['test_generation']['correct']['full_sample_mae']=.5
                supervisor.write(job/'artifacts/result.json',actual)
                self.assertEqual(summary.summarize(suite,protocol_path)['n_complete'],0)


class RawPredictionAuditTests(unittest.TestCase):
    """Literal raw-text replay rejects artifacts with manipulated denominators."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.job=Path(self.temp.name)
        (self.job/'artifacts').mkdir()
        self.legacy=summary.frozen_scoring()
        self.rows=[dict(sample_id=str(i),split='test',group_id='same',pkd=float(i+1),
                   groups={'target_sequence_sha256':'t'},experimental_context={}) for i in range(3)]
        texts=['{"pKD":1.5}','{"pKD":3}','invalid output']
        self.predictions=[dict(sample_id=r['sample_id'],split='test',raw_text=text,
            parse=self.legacy.parse_answer(text),gold_pkd=r['pkd']) for r,text in zip(self.rows,texts)]
        self.metrics=self.legacy.generation_metrics(self.rows,self.predictions)
        self.save(self.predictions)

    def save(self,predictions):
        (self.job/'artifacts/test_predictions_correct.jsonl').write_text(
            '\n'.join(json.dumps(p) for p in predictions)+'\n',encoding='utf-8')

    def tearDown(self): self.temp.cleanup()

    def test_failure_retains_complete_raw_denominator_and_null_full_MAE(self):
        metrics,errors=summary.audit_raw_predictions(self.job,self.rows,self.metrics)
        self.assertEqual((metrics['n_total'],metrics['n_valid'],metrics['n_failed']),(3,2,1))
        self.assertIsNone(metrics['full_sample_mae']);self.assertIsNone(errors)

    def test_mechanism_keeps_category_denominators_distinct_and_never_creates_MAE(self):
        actual={'C1':{'G_all_test':.3,'token_breakdown':{'correct':{
            'numeric':{'nll_nats_per_token':2.,'contribution_nats_per_full_token':.4},
            'mixed':{'nll_nats_per_token':None,'contribution_nats_per_full_token':0.}}}}}
        values,units=summary.mechanism_scalars(actual,[])
        self.assertEqual(values['C1.G_all_test'],.3)
        self.assertEqual(values['C1.token_breakdown.correct.numeric.nll_nats_per_token'],2.)
        self.assertEqual(values['C1.token_breakdown.correct.numeric.contribution_nats_per_full_token'],.4)
        self.assertIsNone(values['C1.token_breakdown.correct.mixed.nll_nats_per_token'])
        self.assertNotEqual(units['C1.token_breakdown.correct.numeric.nll_nats_per_token'],
                            units['C1.token_breakdown.correct.numeric.contribution_nats_per_full_token'])
        self.assertFalse(any('mae' in key.lower() for key in values))

    def test_deleted_failed_raw_row_is_rejected_even_when_reported_full_MAE_null(self):
        self.save(self.predictions[:2])
        with self.assertRaisesRegex(ValueError,'complete frozen test denominator'):
            summary.audit_raw_predictions(self.job,self.rows,self.metrics)

    def test_fabricated_saved_parse_does_not_override_frozen_raw_parser(self):
        self.predictions[-1]['parse']={'valid':True,'value':3.,'reason':None}
        self.save(self.predictions)
        with self.assertRaisesRegex(ValueError,'raw_text parser'):
            summary.audit_raw_predictions(self.job,self.rows,self.metrics)

    def test_fabricated_counts_rejected_by_raw_reconstruction(self):
        self.metrics.update(n_valid=3,n_failed=0,parse_valid_rate=1.,full_sample_mae=.75)
        with self.assertRaisesRegex(ValueError,'raw reconstruction'):
            summary.audit_raw_predictions(self.job,self.rows,self.metrics)

    def test_fabricated_conditional_or_target_macro_MAE_is_rejected(self):
        for mutate in ('mae','group_macro_mae'):
            metrics=copy.deepcopy(self.metrics)
            metrics['regression_valid_only'][mutate]=123.
            with self.subTest(metric=mutate),self.assertRaisesRegex(ValueError,'raw reconstruction'):
                summary.audit_raw_predictions(self.job,self.rows,metrics)

    def test_duplicate_id_rejected_before_any_scoring(self):
        self.predictions[-1]['sample_id']=self.predictions[0]['sample_id']
        self.save(self.predictions)
        with self.assertRaisesRegex(ValueError,'duplicate sample IDs'):
            summary.audit_raw_predictions(self.job,self.rows,self.metrics)

    def test_permuted_population_rejected_with_failures_present(self):
        self.save(list(reversed(self.predictions)))
        with self.assertRaisesRegex(ValueError,'identical frozen order'):
            summary.audit_raw_predictions(self.job,self.rows,self.metrics)


class ContractFixtureTests(unittest.TestCase):
    """Real standard validator + physical tiny fixtures, never real model-ready."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.suite=self.root/'suite';self.suite.mkdir()
        self.job=self.suite/'cpu_fixture_only';self.job.mkdir()
        self.protocol=supervisor.load(ROOT/'configs/matched_budget_20261016.protocol.json')
        self.protocol['status']='frozen'
        stamp=datetime.now(timezone.utc)
        self.protocol['timing'].update(first_substantive_started_at=(stamp-timedelta(minutes=1)).isoformat(),
            earliest_associated_preparation_at=(stamp-timedelta(minutes=1)).isoformat(),effective_deadline_at=(stamp+timedelta(hours=1)).isoformat())
        self.protocol['resources']['host']=socket.gethostname()
        self.case=supervisor.trial_matrix(self.protocol)[0]
        data=self.root/'data';data.mkdir()
        self.samples=data/'samples.jsonl';self.samples.write_text('fixture, not actual samples')
        self.lock=data/'benchmark_lock.json';self.lock.write_text('fixture, not actual lock')
        self.protocol['scenarios'][self.case['scenario']]['samples_sha256']=supervisor.sha(self.samples)
        self.manifest=self.root/'manifest.json'
        supervisor.write(self.manifest,{'purpose':'matched_budget_fixed_message_cache','datasets':{
            self.case['scenario']:{'samples_path':str(self.samples),'lock_path':str(self.lock),'layouts':{}}},
            'source_pins':{},'initializations':{},'projection':{'path':str(self.lock),'sha256':supervisor.sha(self.lock)}})
        self.protocol_path=self.root/'protocol.json';supervisor.write(self.protocol_path,self.protocol)
        for raw in ('papers/bench_v02/ppikb/article.pdf','models/Qwen--Qwen2.5-1.5B-Instruct/model.safetensors',
                    'models/Qwen--Qwen2.5-1.5B-Instruct/config.json'):
            path=self.root/raw;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('CPU fixture only')
        (self.suite/'environment_freeze.txt').write_text('no install; CPU fixture')
        for name in ('stdout.log','stderr.log'): (self.job/name).touch()
        self.probe=dict(utc=supervisor.now(),readiness='ready_for_exact_bounded_contract',blocking_reasons=[],
                        CPU_fixture_only=True,scientific_readiness=False)
        supervisor.write(self.job/'resource_probe.json',self.probe)
        supervisor.write(self.job/'case.json',self.case)
        self.patch=patch.object(supervisor,'ROOT',self.root);self.patch.start()
        command=supervisor.build_command(self.case,self.job,self.protocol_path,supervisor.sha(self.protocol_path),
            self.manifest,supervisor.sha(self.manifest),30)
        self.contract=supervisor.build_contract(self.case,self.job,self.suite,self.protocol_path,self.protocol,
            supervisor.sha(self.protocol_path),self.manifest,supervisor.sha(self.manifest),self.probe,30,command,'CPU fixture only; not GPU release')
        supervisor.write(self.job/'task-contract.json',self.contract)
        self.validation=supervisor.validator().validate(self.contract,contract_bytes=(self.job/'task-contract.json').read_bytes(),
            check_files=True,contract_directory=self.job)
        supervisor.write(self.job/'contract-validation.json',self.validation)

    def tearDown(self):
        self.patch.stop();self.temp.cleanup()

    def test_non_template_frozen_standard_contract_checks_physical_fixture_manifest(self):
        self.assertTrue(self.validation['valid_for_execution'],self.validation)
        dataset=self.contract['assets']['datasets'][0]
        self.assertEqual(dataset['manifest_path'],str(self.lock))
        self.assertEqual(dataset['manifest_sha256'],supervisor.sha(self.lock))
        checked=[x for x in self.validation['files_checked'] if x['path']==str(self.lock)]
        self.assertTrue(any(x['hash_verified'] for x in checked))

    def test_missing_or_wrong_manifest_digest_rejected(self):
        bad=copy.deepcopy(self.contract);bad['assets']['datasets'][0]['manifest_sha256']='0'*64
        checked=supervisor.validator().validate(bad,check_files=True,contract_directory=self.job)
        self.assertFalse(checked['valid_for_execution'])
        self.assertTrue(any('mismatch' in x['message'] for x in checked['errors']))

    def test_draft_protocol_and_changed_manifest_fail_closed(self):
        protocol=copy.deepcopy(self.protocol);protocol['status']='draft'
        supervisor.write(self.protocol_path,protocol)
        with self.assertRaisesRegex(ValueError,'not frozen'):
            supervisor.verify_inputs(self.protocol_path,supervisor.sha(self.protocol_path),self.manifest,supervisor.sha(self.manifest))
        with self.assertRaisesRegex(ValueError,'differs'):
            supervisor.verify_inputs(self.protocol_path,'0'*64,self.manifest,supervisor.sha(self.manifest))

    def test_not_spawned_failure_archives_without_invented_pid(self):
        receipt=supervisor.not_started_receipt(self.job,'CPU admission negative fixture')
        self.assertFalse(supervisor.archive(self.job,self.contract,receipt))
        result=supervisor.load(self.job/'result-record.json')
        self.assertFalse(result['execution']['performed'])
        self.assertIsNone(result['execution']['pid'])
        self.assertEqual(result['reproduction_level'],'registered')
        self.assertTrue(supervisor.load(self.job/'result-validation.json')['valid_for_acceptance'])

    @unittest.skipUnless(sys.platform.startswith('linux'),'whole process groups exercised in WSL Linux')
    def test_timeout_kills_owned_child_tree_and_archives_real_failure(self):
        import ctypes
        self.assertEqual(ctypes.CDLL(None).prctl(36,1,0,0,0),0)
        code="import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);time.sleep(30)"
        receipt=supervisor.supervise([sys.executable,'-c',code],self.job,.25,1024**3,
            datetime.now(timezone.utc)+timedelta(seconds=10),poll_seconds=.02)
        self.assertTrue(receipt['performed']);self.assertEqual(receipt['termination_reason'],'timeout')
        self.assertTrue(receipt['process_group_absent'])
        self.assertIsNotNone(receipt['exit_code'])
        self.assertFalse(supervisor.archive(self.job,self.contract,receipt))
        self.assertTrue(supervisor.load(self.job/'result-validation.json')['valid_for_acceptance'])

    @unittest.skipUnless(sys.platform.startswith('linux'),'interrupt process cleanup exercised in WSL Linux')
    def test_keyboard_interrupt_retains_terminal_receipt(self):
        original_sleep=supervisor.time.sleep
        injected=[False]
        def interrupt_once(seconds):
            if not injected[0]:
                injected[0]=True
                raise KeyboardInterrupt('CPU injected interruption')
            original_sleep(seconds)
        with patch.object(supervisor.time,'sleep',side_effect=interrupt_once):
            receipt=supervisor.supervise([sys.executable,'-c','import time;time.sleep(30)'],self.job,30,1024**3,
                datetime.now(timezone.utc)+timedelta(minutes=1))
        self.assertTrue(receipt['process_group_absent'])
        self.assertEqual(receipt['termination_reason'],'controller_exception')
        self.assertEqual(receipt['exception']['type'],'KeyboardInterrupt')
        self.assertFalse(supervisor.archive(self.job,self.contract,receipt))

    @unittest.skipUnless(sys.platform.startswith('linux'),'real SIGTERM receipt/cleanup exercised in WSL Linux')
    def test_sigterm_requests_controlled_stop_and_preserves_archive(self):
        previous=signal.signal(signal.SIGTERM,supervisor.request_stop)
        timer=threading.Timer(.1,lambda:os.kill(os.getpid(),signal.SIGTERM))
        try:
            timer.start()
            receipt=supervisor.supervise([sys.executable,'-c','import time;time.sleep(30)'],self.job,30,1024**3,
                datetime.now(timezone.utc)+timedelta(minutes=1),poll_seconds=.01)
            self.assertEqual(receipt['termination_reason'],'controller_signal_'+str(int(signal.SIGTERM)))
            self.assertTrue(receipt['process_group_absent'])
            self.assertFalse(supervisor.archive(self.job,self.contract,receipt))
        finally:
            timer.join();signal.signal(signal.SIGTERM,previous);supervisor.PENDING_SIGNAL=None

    def test_spawn_error_is_terminal_and_archived(self):
        receipt=supervisor.supervise(['a_definitely_nonexistent_cpu_fixture_executable'],self.job,30,1024**3,
            datetime.now(timezone.utc)+timedelta(minutes=1))
        self.assertFalse(receipt['performed']);self.assertTrue(receipt['process_group_absent'])
        self.assertEqual(receipt['exception']['type'],'FileNotFoundError')
        self.assertFalse(supervisor.archive(self.job,self.contract,receipt))


if __name__=='__main__': unittest.main()

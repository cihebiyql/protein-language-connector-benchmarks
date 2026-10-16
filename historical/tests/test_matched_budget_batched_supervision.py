"""CPU-only V2 supervision/reuse fixtures; never real training evidence."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from matched_budget_20261016 import execute_suite_batched as supervisor
from matched_budget_20261016 import summarize_batched as summary


class ReuseAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.protocol=supervisor.load(ROOT/'configs/matched_budget_20261016.frozen.json')
        self.protocol_sha=supervisor.sha(ROOT/'configs/matched_budget_20261016.frozen.json')
        self.case=supervisor.trial_matrix(self.protocol)[0]
        self.source=self.root/supervisor.case_id(self.case);(self.source/'artifacts').mkdir(parents=True)
        self.manifest={'projection':{'tensor_sha256':'input-independent-projection'},
            'datasets':{self.case['scenario']:{'layouts':{'balanced':{'path':'frozen_message.pt','sha256':'same'}}}},
            'initializations':{'17:zero':{'path':'shared_untrained.pt','sha256':'same','untrained':True}}}
        self.old_manifest=self.root/'old_manifest.json';supervisor.write(self.old_manifest,self.manifest)
        self.contract=dict(is_template=False,contract_status='frozen',execution_scope={'intended_level':'training'},
            matched_budget_registration={'protocol_sha256':self.protocol_sha,'cache_manifest_path':str(self.old_manifest),
                'cache_manifest_sha256':supervisor.sha(self.old_manifest)})
        supervisor.write(self.source/'task-contract.json',self.contract)
        supervisor.write(self.source/'case.json',self.case)
        supervisor.write(self.source/'execution-receipt.json',dict(exit_code=0,process_group_absent=True,termination_reason=None,CPU_fixture_only=True))
        supervisor.write(self.source/'result-record.json',dict(lifecycle_status='succeeded',reproduction_level='training',
            contract_sha256=supervisor.sha(self.source/'task-contract.json'),CPU_fixture_only=True))
        supervisor.write(self.source/'result-validation.json',dict(valid_for_acceptance=True,CPU_fixture_only=True))
        artifact=self.source/'artifacts';(artifact/'best_reader.pt').write_text('tiny fixture, not a model')
        checkpoint_sha=supervisor.sha(artifact/'best_reader.pt')
        supervisor.write(artifact/'history.json',[dict(epoch=1,validation_answer_token_nll=3.),
            dict(epoch=2,validation_answer_token_nll=2.),dict(epoch=3,validation_answer_token_nll=2.)])
        supervisor.write(artifact/'selection_before_test.json',dict(best_epoch=2,all_training_epochs_completed=3,
            optimizer_steps=1548,test_labels_used_for_selection=False,checkpoint_sha256=checkpoint_sha))
        self.actual=dict(status='complete',actual_epochs=3,optimizer_steps=1548,best_epoch=2,checkpoint_sha256=checkpoint_sha,
            registration={'configuration':{k:self.case[k] for k in ('scenario','seed','placement','message_input','sender_mode','gate_init')}},
            artifact_sha256={p.name:supervisor.sha(p) for p in artifact.iterdir()})
        supervisor.write(artifact/'result.json',self.actual)
        self.validator=Mock(return_value=SimpleNamespace(validate=Mock(return_value={'valid_for_acceptance':True,'CPU_fixture_only':True})))
        self.patch=patch.object(supervisor,'validator',self.validator);self.patch.start()

    def tearDown(self): self.patch.stop();self.temp.cleanup()

    def admit(self):
        return supervisor.validate_reuse_job(self.source,self.case,self.protocol_sha,self.manifest)

    def test_complete_training_reused_as_zero_new_optimizer_steps(self):
        value=self.admit()
        self.assertEqual(value['source_training_optimizer_steps'],1548)
        self.assertEqual(value['new_optimizer_steps'],0)
        self.assertFalse(value['new_training_performed'])
        self.assertEqual(value['source_best_epoch'],2)
        self.validator.return_value.validate.assert_called_once()
        supervisor.check_reuse_pins(value)

    def test_partial_training_rejected(self):
        self.actual['optimizer_steps']=1547
        supervisor.write(self.source/'artifacts/result.json',self.actual)
        with self.assertRaisesRegex(ValueError,'Partial V1 training'): self.admit()

    def test_changed_checkpoint_rejected(self):
        (self.source/'artifacts/best_reader.pt').write_text('changed fixture')
        with self.assertRaisesRegex(ValueError,'checkpoint identity'): self.admit()

    def test_changed_evidence_after_admission_rejected(self):
        value=self.admit()
        (self.source/'artifacts/history.json').write_text('changed history')
        with self.assertRaisesRegex(ValueError,'evidence changed'): supervisor.check_reuse_pins(value)

    def test_earliest_validation_tie_not_reselected(self):
        self.actual['best_epoch']=3
        supervisor.write(self.source/'artifacts/result.json',self.actual)
        with self.assertRaisesRegex(ValueError,'checkpoint identity'): self.admit()

    def test_nonaccepted_or_unverified_source_archive_rejected(self):
        self.validator.return_value.validate.return_value={'valid_for_acceptance':False}
        with self.assertRaisesRegex(ValueError,'physical file validation'): self.admit()

    def test_failed_source_process_rejected(self):
        supervisor.write(self.source/'execution-receipt.json',dict(exit_code=1,process_group_absent=True))
        with self.assertRaisesRegex(ValueError,'terminal accepted'): self.admit()

    def test_case_or_fixed_message_change_rejected(self):
        bad=copy.deepcopy(self.manifest)
        bad['datasets'][self.case['scenario']]['layouts']['balanced']['sha256']='different'
        with self.assertRaisesRegex(ValueError,'message projection/payload'):
            supervisor.validate_reuse_job(self.source,self.case,self.protocol_sha,bad)
        other=dict(self.case,message_input='neutral')
        with self.assertRaisesRegex(ValueError,'corresponding fixed-order'):
            supervisor.validate_reuse_job(self.source,other,self.protocol_sha,self.manifest)

    def test_only_first_one_or_two_cases_can_enter_reuse_plan(self):
        plan=supervisor.reuse_plan(self.root,1,self.protocol,self.protocol_sha,self.manifest)
        self.assertEqual(list(plan),[supervisor.case_id(self.case)])
        with self.assertRaises(ValueError): supervisor.reuse_plan(self.root,3,self.protocol,self.protocol_sha,self.manifest)
        with self.assertRaises(ValueError): supervisor.reuse_plan(None,1,self.protocol,self.protocol_sha,self.manifest)
        self.assertEqual(supervisor.reuse_plan(None,0,self.protocol,self.protocol_sha,self.manifest),{})

    def test_real_standard_validator_physical_V2_reuse_contract_and_terminal_archive(self):
        """Administrative CPU synthetic files only; never GPU/training readiness."""
        reuse=self.admit()
        self.patch.stop()  # The new contract and terminal record use the REAL validator.
        case={**self.case,'reuse_training':reuse}
        suite=self.root/'V2_cpu_fixture';suite.mkdir();job=suite/supervisor.case_id(case);job.mkdir()
        (suite/'environment_freeze.txt').write_text('CPU synthetic administrative fixture; no environment mutation')
        for name in ('stdout.log','stderr.log'): (job/name).touch()
        supervisor.write(job/'case.json',case)
        data=self.root/'dataset';data.mkdir()
        (data/'samples.jsonl').write_text('physical tiny CPU fixture, not actual dataset')
        (data/'benchmark_lock.json').write_text('physical tiny CPU lock fixture')
        manifest=copy.deepcopy(self.manifest)
        manifest['datasets'][case['scenario']].update(samples_path=str(data/'samples.jsonl'),lock_path=str(data/'benchmark_lock.json'))
        manifest_path=self.root/'V2_manifest.json';supervisor.write(manifest_path,manifest)
        protocol=copy.deepcopy(self.protocol)
        protocol['resources']['host']=socket.gethostname()
        protocol['scenarios'][case['scenario']]['samples_sha256']=supervisor.sha(data/'samples.jsonl')
        protocol_path=self.root/'V2_protocol_fixture.json';supervisor.write(protocol_path,protocol)
        for raw in ('papers/bench_v02/ppikb/article.pdf','models/Qwen--Qwen2.5-1.5B-Instruct/model.safetensors',
                    'models/Qwen--Qwen2.5-1.5B-Instruct/config.json'):
            path=self.root/raw;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('CPU fixture only')
        probe=dict(utc=supervisor.now(),readiness='CPU_fixture_not_GPU_readiness',blocking_reasons=[])
        supervisor.write(job/'resource_probe.json',probe)
        command=[sys.executable,'-c','pass']
        with patch.object(supervisor,'ROOT',self.root):
            contract=supervisor.build_contract(case,job,suite,protocol_path,protocol,supervisor.sha(protocol_path),
                manifest_path,supervisor.sha(manifest_path),probe,30,command,'CPU fixture only, not GPU release')
            supervisor.write(job/'task-contract.json',contract)
            checked=supervisor.validator().validate(contract,contract_bytes=(job/'task-contract.json').read_bytes(),check_files=True,contract_directory=job)
            supervisor.write(job/'contract-validation.json',checked)
            self.assertTrue(checked['valid_for_execution'],checked)
            self.assertEqual(contract['execution_scope']['intended_level'],'independent_inference')
            self.assertEqual(contract['targets'][0]['repetitions']['epochs'],0)
            self.assertEqual(contract['engineering_revision']['new_optimizer_steps'],0)
            receipt=supervisor.supervise(command,job,30,1024**3,datetime.now(timezone.utc)+timedelta(minutes=1))
            self.assertTrue(receipt['performed']);self.assertEqual(receipt['exit_code'],0)
            self.assertTrue(receipt['process_group_absent'])
            # Failed scientific output is expected: only a real CPU `pass` ran.
            self.assertFalse(supervisor.archive(job,contract,receipt))
            record=supervisor.load(job/'result-record.json')
            self.assertEqual(record['reproduction_level'],'independent_inference')
            self.assertEqual(record['contract_sha256'],supervisor.sha(job/'task-contract.json'))
            self.assertEqual(record['training_provenance']['new_training_count'],0)
            self.assertEqual(record['training_provenance']['source_training_optimizer_steps'],1548)
            self.assertTrue(supervisor.load(job/'result-validation.json')['valid_for_acceptance'])


class V2BoundaryTests(unittest.TestCase):
    def test_same_36_scientific_order_and_six_train_qualification_cases(self):
        from matched_budget_20261016 import execute_suite as original
        protocol=supervisor.load(ROOT/'configs/matched_budget_20261016.frozen.json')
        self.assertEqual(supervisor.trial_matrix(protocol),original.trial_matrix(protocol))
        self.assertEqual(supervisor.canary_matrix(protocol),original.canary_matrix(protocol))

    def test_worker_command_names_only_new_controller(self):
        command=supervisor.build_command({},Path('/tmp/fixture'),Path('/tmp/protocol'),'p',Path('/tmp/cache'),'c',1800)
        self.assertEqual(Path(command[1]).name,'execute_suite_batched.py')
        self.assertIn('--worker',command)

    def test_inference_reuse_is_not_counted_as_new_training(self):
        state={'jobs':[{'complete':True,'case':{'phase':'canary'}},
                       {'complete':True,'case':{'phase':'trial','reuse_training':{'new_optimizer_steps':0}}},
                       {'complete':True,'case':{'phase':'trial'}},
                       {'complete':False,'case':{'phase':'trial'}}]}
        supervisor.update_completion_counts(state)
        self.assertEqual(state['TRAIN_canaries_completed'],1)
        self.assertEqual(state['reused_training_inference_cases_completed'],1)
        self.assertEqual(state['new_training_trials_completed'],1)
        self.assertEqual(state['total_uniform_V2_evaluation_cases_completed'],2)

    @unittest.skipUnless(sys.platform.startswith('linux'),'V2 process-group timeout fixture requires WSL Linux')
    def test_v2_timeout_records_terminal_whole_group_absence(self):
        import ctypes
        self.assertEqual(ctypes.CDLL(None).prctl(36,1,0,0,0),0)
        with tempfile.TemporaryDirectory() as raw:
            job=Path(raw)
            for name in ('stdout.log','stderr.log'): (job/name).touch()
            code="import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);time.sleep(30)"
            receipt=supervisor.supervise([sys.executable,'-c',code],job,.15,1024**3,
                datetime.now(timezone.utc)+timedelta(seconds=10),poll_seconds=.01)
            self.assertEqual(receipt['termination_reason'],'timeout')
            self.assertTrue(receipt['process_group_absent'])
            self.assertIsNotNone(receipt['exit_code'])
            self.assertTrue((job/'execution-receipt.json').is_file())


class V2SummaryBudgetTests(unittest.TestCase):
    def setUp(self):
        self.actual=dict(actual_epochs=3,optimizer_steps=1548,generation_batch_size=4,
            execution_revision={'id':'fixed_order_generation_batch4_v2'},
            training_budget_original={'epochs':3,'optimizer_steps':1548},new_optimizer_steps=0,
            training_performed_in_this_run=False,training_reused_from_v1=True)
        self.case={'reuse_training':{'source_job':'CPU_fixture','source_training_epochs':3,'source_training_optimizer_steps':1548}}
        self.contract={'execution_scope':{'intended_level':'independent_inference'}}
        self.record={'reproduction_level':'independent_inference','lifecycle_status':'succeeded'}

    def test_reuse_source_budget_not_reported_as_new_training(self):
        view=summary.budget_view(self.actual,self.case,self.contract,self.record)
        self.assertEqual(view['new_optimizer_steps'],0)
        self.assertEqual(view['new_training_epochs'],0)
        self.assertEqual(view['new_training_count'],0)
        self.assertEqual(view['cumulative_or_source_training_budget'],{'epochs':3,'optimizer_steps':1548})

    def test_reuse_claiming_new_updates_or_training_level_rejected(self):
        self.actual['new_optimizer_steps']=1548
        with self.assertRaisesRegex(ValueError,'zero new optimizer'): summary.budget_view(self.actual,self.case,self.contract,self.record)
        self.actual['new_optimizer_steps']=0;self.record['reproduction_level']='training'
        with self.assertRaisesRegex(ValueError,'execution level'): summary.budget_view(self.actual,self.case,self.contract,self.record)

    def test_fresh_training_and_uniform_batch_must_both_be_proven(self):
        self.actual.update(new_optimizer_steps=1548,training_performed_in_this_run=True,training_reused_from_v1=False)
        self.contract['execution_scope']['intended_level']='training';self.record['reproduction_level']='training'
        view=summary.budget_view(self.actual,{},self.contract,self.record)
        self.assertEqual(view['new_training_count'],1)
        self.actual['generation_batch_size']=1
        with self.assertRaisesRegex(ValueError,'batch4'): summary.budget_view(self.actual,{},self.contract,self.record)

    def test_partial_source_budget_rejected_and_three_seed_null_rule_retained(self):
        self.actual['optimizer_steps']=1547
        with self.assertRaisesRegex(ValueError,'training budget'): summary.budget_view(self.actual,self.case,self.contract,self.record)
        self.assertIsNone(summary.original.seed_aggregate({17:1.,42:2.,2026:None})['mean'])

    def test_V2_adapter_reuses_complete_raw_audit_and_reports_one_reused_case(self):
        """Physical analytic fixtures only, not real model execution evidence."""
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);suite=root/'V2';suite.mkdir()
            protocol=supervisor.load(ROOT/'configs/matched_budget_20261016.frozen.json')
            rows=[dict(sample_id=str(i),split='test',group_id='g',pkd=float(i+1),
                       groups={'target_sequence_sha256':'t'},experimental_context={}) for i in range(2)]
            for scenario in protocol['scenarios']:
                base=root/protocol['data_root']/scenario;base.mkdir(parents=True)
                (base/'samples.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
                protocol['scenarios'][scenario].update(test=2,samples_sha256=supervisor.sha(base/'samples.jsonl'))
            protocol_path=root/'protocol.json';supervisor.write(protocol_path,protocol)
            base_case=supervisor.trial_matrix(protocol)[0]
            source=root/'source_artifacts';source.mkdir()
            supervisor.write(source/'result.json',{'actual_epochs':3,'optimizer_steps':1548,'checkpoint_sha256':'fixture'})
            reuse={**self.case['reuse_training'],'source_artifacts':str(source),
                   'evidence_pins':{str(source/'result.json'):supervisor.sha(source/'result.json')}}
            case={**base_case,'reuse_training':reuse};job=suite/supervisor.case_id(base_case);(job/'artifacts').mkdir(parents=True)
            supervisor.write(suite/'reuse_plan.json',{supervisor.case_id(base_case):reuse})
            supervisor.write(job/'case.json',case);supervisor.write(job/'task-contract.json',self.contract)
            record={**self.record,'contract_sha256':supervisor.sha(job/'task-contract.json'),'CPU_fixture_only':True}
            supervisor.write(job/'result-record.json',record)
            supervisor.write(job/'result-validation.json',{'valid_for_acceptance':True,'CPU_fixture_only':True})
            supervisor.write(job/'execution-receipt.json',{'exit_code':0,'process_group_absent':True,'CPU_fixture_only':True})
            legacy=summary.original.frozen_scoring()
            predictions=[dict(sample_id=r['sample_id'],split='test',gold_pkd=r['pkd'],raw_text=json.dumps({'pKD':r['pkd']+.5}),
                parse=legacy.parse_answer(json.dumps({'pKD':r['pkd']+.5}))) for r in rows]
            pred_path=job/'artifacts/test_predictions_correct.jsonl'
            pred_path.write_text('\n'.join(json.dumps(p) for p in predictions),encoding='utf-8')
            actual={**self.actual,'status':'complete','best_epoch':2,'checkpoint_sha256':'fixture',
                    'test_generation':{'correct':legacy.generation_metrics(rows,predictions)}}
            supervisor.write(job/'artifacts/result.json',actual)
            with patch.object(summary.original,'ROOT',root):
                result=summary.summarize(suite,protocol_path)
                self.assertEqual(result['n_complete'],1)
                self.assertEqual(result['n_completed_new_training_trials'],0)
                self.assertEqual(result['n_completed_reused_training_inference_cases'],1)
                self.assertEqual(result['cases'][0]['new_optimizer_steps'],0)
                self.assertEqual(result['cases'][0]['training_budget_optimizer_steps'],1548)
                self.assertNotIn('actual_epochs',result['cases'][0])
                self.assertIsNone(result['seed_summaries'][0]['mean'])
                pred_path.write_text(json.dumps(predictions[0])+'\n',encoding='utf-8')
                self.assertEqual(summary.summarize(suite,protocol_path)['n_complete'],0)


if __name__=='__main__': unittest.main()

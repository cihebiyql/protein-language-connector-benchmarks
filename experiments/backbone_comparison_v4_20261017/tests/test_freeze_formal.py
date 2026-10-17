"""Synthetic CPU-only freezer fixtures; no actual experiment is frozen or launched."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCOPE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCOPE))
from src import mm_evaluate as ev
spec=importlib.util.spec_from_file_location('v4_freezer_fixture',SCOPE/'resources/freeze_formal.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8');return path


class FreezerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.scope=self.root/'new/experiments/backbone_comparison_v4_20261017'
        (self.scope/'resources').mkdir(parents=True)
        (self.scope/'src').mkdir();(self.scope/'configs').mkdir();(self.scope/'docs').mkdir()
        for p in (SCOPE/'src').glob('*.py'):(self.scope/'src'/p.name).write_bytes(p.read_bytes())
        self.now=datetime.fromisoformat(f.DEADLINE)-timedelta(hours=2)
        self.config=json.loads((SCOPE/'configs/development_protocol.json').read_text(encoding='utf8'))
        self.config['training']['generation_batch_size']=8
        for b,s in self.config['backbones'].items():
            s['revision']='989aa7980e4cf806f80c7fef2b1adb7bc71aa306' if b==f.BACKBONES[0] else 'cdbee75f17c01a7cc42f958dc650907174af0554'
            s['model_manifest_path']='experiments/backbone_comparison_v4_20261017/resources/'+b+'.json'
        self.config_path=put(self.scope/'configs/canary.json',self.config)
        self.manifest_path=self.scope/'runs/canary/artifacts/run_manifest.json'
        self.archive=self.scope/'runs/canary/archive'
        artifacts=self.manifest_path.parent
        anchor=put(artifacts/'anchor.json',{'passed':True,'split':'train'})
        length=put(artifacts/'length.json',{'source_split':'train','all_train_rows_inspected':2062,'validation_or_test_opened':False,'first_effective_batch_indices':[0,1,2,3],'first_effective_batch_is_optimizer_update':1})
        longest=put(artifacts/'longest.json',{'split':'train','rows':8,'max_new_tokens':24,'development_only':True})
        metric=put(artifacts/'metrics.json',{'split':'train','rows':8,'development_only':True,'numeric':{'n_finite':8,'n_failed':0},'c1_seconds':1.})
        ref=lambda p:{'path':str(p.relative_to(artifacts)),'sha256':f.sha(p)}
        self.manifest={'scope_id':f.SCOPE_ID,'backbone':f.BACKBONES[0],'execution_phase':'train-canary','phase':'development','status':'complete',
            'test_labels_used_for_fit':False,'protocol_sha256':f.sha(self.config_path),'backbone_config_sha256':f.digest(self.config['backbones'][f.BACKBONES[0]]),
            'scenario':'new_peptide_known_target','arm':'M3','seed':17,'producer_source_sha256':f.sha(self.scope/'src/mm_train.py'),
            'training':{'optimizer_steps':4,'examples_seen':16,'epochs_completed':0},
            'runtime':{'training_performance':{'elapsed_seconds':4.,'validation_used':False,'test_labels_used_for_fit':False,'peak_cuda_memory_bytes':1024},
                'canary_generation_seconds':1.,'model_loading_seconds':1.,'asset_verification_seconds':1.},
            'model_policy':{'trainable_parameters':10,'frozen_base_eval':True,'trainable_dtype':'torch.float32','base_dtype':'torch.bfloat16'},
            'provenance':{'code_sha256':{p.name:f.sha(p) for p in (self.scope/'src').glob('*.py')},
                'view_manifest_sha256':'a'*64,'cache_before_run':{'manifest_sha256':'b'*64},'model_assets':{'manifest_sha256':'c'*64},
                'audit_artifacts':{'train_anchor_qualification':ref(anchor),'train_length_qualification':ref(length),'max_length_generation_qualification':ref(longest),'train_canary_metrics':ref(metric)}}}
        self.save_canary()

    def tearDown(self):self.tmp.cleanup()

    def save_canary(self):
        put(self.manifest_path,self.manifest)
        job=self.scope/'runs/canary'
        cap=put(job/'gpu_limit.json',{'visible_devices':'4','allocator_budget_bytes':20*f.GIB,'module':'src.mm_train','peak_allocated_bytes':1024,'peak_reserved_bytes':2048})
        command=[sys.executable,'-m','src.mm_worker_limit','--gpu-memory-budget-bytes',str(20*f.GIB),'--receipt',str(cap),'--module','src.mm_train','--','--output',str(self.manifest_path.parent),'--phase','train-canary','--max-updates','4']
        contract=put(job/'task-contract.json',{'task_id':f.SCOPE_ID,'execution':{'command':command},'resources':{'gpu_devices':[4]}})
        put(job/'supervision/receipt.json',{'status':'succeeded','exit_code':0,'remaining_group_members':[],'contract_sha256':f.sha(contract)})
        record=put(self.archive/'result-record.json',{'contract_path':str(contract),'contract_sha256':f.sha(contract),'task_id':f.SCOPE_ID,
            'review':{'status':'accepted'},'lifecycle_status':'succeeded','artifacts':[{'path':str(self.manifest_path),'sha256':f.sha(self.manifest_path)}]})
        terminal=put(self.archive/'terminal_validation.json',{'valid':True,'valid_for_acceptance':True,'files_skipped':[]})
        put(self.archive/'archive_receipt.json',{'status':'accepted','record_sha256':f.sha(record),'validation_sha256':f.sha(terminal),'contract_sha256':f.sha(contract)})

    def accept(self):return f.accepted_canary(f.BACKBONES[0],self.manifest_path,self.archive,self.config,self.config_path,self.scope)

    def test_uses_actual_runtime_training_performance_not_top_training_elapsed(self):
        self.manifest['training']['elapsed_seconds']=999999.;self.save_canary()
        result=self.accept();self.assertEqual(result['seconds_per_optimizer_step'],1.)
        self.assertEqual(result['training_seconds'],4.)

    def test_terminal_self_claim_without_canonical_acceptance_rejected(self):
        terminal=self.archive/'terminal_validation.json';put(terminal,{'valid':True,'valid_for_acceptance':False,'files_skipped':[]})
        receipt=json.loads((self.archive/'archive_receipt.json').read_text(encoding='utf8'));receipt['validation_sha256']=f.sha(terminal);put(self.archive/'archive_receipt.json',receipt)
        with self.assertRaises(ValueError):self.accept()

    def test_runtime_source_drift_after_canary_rejected(self):
        (self.scope/'src/mm_data_evaluation.py').write_text('# changed after canary')
        with self.assertRaisesRegex(ValueError,'source changed'):self.accept()

    def test_validation_use_and_missing_real_training_time_rejected(self):
        self.manifest['runtime']['training_performance']['validation_used']=True;self.save_canary()
        with self.assertRaises(ValueError):self.accept()
        self.manifest['runtime']['training_performance']['validation_used']=False
        self.manifest['runtime']['training_performance'].pop('elapsed_seconds');self.save_canary()
        with self.assertRaises(ValueError):self.accept()

    def test_wrong_backbone_and_archive_manifest_byte_drift_rejected(self):
        with self.assertRaises(ValueError):f.accepted_canary(f.BACKBONES[1],self.manifest_path,self.archive,self.config,self.config_path,self.scope)
        self.manifest_path.write_text('{}')
        with self.assertRaises(ValueError):self.accept()

    def test_cap_and_cleanup_are_required_even_if_archive_claims_acceptance(self):
        cap=self.scope/'runs/canary/gpu_limit.json';value=json.loads(cap.read_text(encoding='utf8'))
        value['peak_reserved_bytes']=21*f.GIB;put(cap,value)
        with self.assertRaisesRegex(ValueError,'actual GPU'):self.accept()
        self.save_canary()
        receipt=self.scope/'runs/canary/supervision/receipt.json'
        value=json.loads(receipt.read_text(encoding='utf8'));value['remaining_group_members']=[123];put(receipt,value)
        with self.assertRaisesRegex(ValueError,'supervision'):self.accept()

    def test_archive_existing_is_model_free_and_checks_exact_TRAIN_completion(self):
        archive=self.scope/'runs/canary/new_archive'
        calls=[]
        def fake_archive(contract,supervision,roots,standard,output,*,completion_spec):
            calls.append((contract,supervision,roots,standard,output,completion_spec))
            self.assertEqual(completion_spec['target_counts']['registered_endpoint']['pointer'],'/rows')
            self.assertIn({'file':'run_manifest.json','pointer':'/training/optimizer_steps','equals':4},completion_spec['assertions'])
            return {'status':'accepted','fixture_only':True}
        result=f.archive_existing_canary(f.BACKBONES[0],self.manifest_path,archive,self.root/'standard_fixture',archive_fn=fake_archive)
        self.assertEqual(result['status'],'accepted');self.assertEqual(len(calls),1)
        self.assertEqual(f.sha(archive.with_name('new_archive_gpu_evidence')/'gpu_limit.json'),f.sha(self.scope/'runs/canary/gpu_limit.json'))

    def test_ready_label_cannot_override_cpu_gpu_or_staleness(self):
        observation={'utc':self.now.isoformat(),'hostname':'node1','gpu':4,'readiness':'ready','reasons':[],
            'cpu_idle_percent':10.,'cpu_iowait_percent':0.,'memory_available_bytes':100*f.GIB,'storage_available_bytes':100*f.GIB,
            'gpu_frame':{'used_mib':15,'utilization':0},'gpu_compute_processes':[]}
        path=put(self.scope/'resources/ready4.json',observation)
        self.assertEqual(f.ready_assignment(4,path,'node1',self.scope,self.now)['gpu'],4)
        for update in ({'cpu_idle_percent':1.},{'gpu_compute_processes':['fixture_external']},{'utc':(self.now-timedelta(minutes=4)).isoformat()}):
            put(path,{**observation,**update})
            with self.subTest(update=update),self.assertRaises(ValueError):f.ready_assignment(4,path,'node1',self.scope,self.now)

    def test_full_build_keeps24_registry_and_all_source_pins_in_temporary_fixture(self):
        # Everything below is an isolated synthetic fixture, never remote or scientific input.
        old=self.root/'old';oldscope=old/'experiments/molecular_memory_v3_20261017'
        views=oldscope/'data_views';cache=oldscope/'cache/paired_esm15_30_v1';project=self.root/'new'
        datas={};viewdoc={'scenarios':{}}
        for scenario,files in ev.DATA_SHA256.items():
            datas[scenario]={}
            for name in files:
                path=put(old/'data/processed/bench_v02/a1_locked_v2'/scenario/name,{'fixture':name})
                datas[scenario][name]=f.sha(path)
            v=put(views/scenario/'train.jsonl',{'fixture':'train'})
            viewdoc['scenarios'][scenario]={'source_shas':datas[scenario],'pins':{},'views':{'train':{'path':str(v.relative_to(views)),'sha256':f.sha(v)}}}
        put(views/'manifest.json',viewdoc)
        payload=put(cache/'fixture.bin',{'fixture':'cached tensor substitute'})
        put(cache/'manifest.json',{'cache_status':'complete','completed_sequences':2506,'raw_residues_covered':408342,
            'sequences':{'fixture':{'path':'fixture.bin','sha256':f.sha(payload)}}})
        evidence={}
        for b,s in self.config['backbones'].items():
            rows=[]
            for repo,revision,modelroot in [(s['repo'],s['revision'],project/s['backbone_path']),
                ('facebook/esm2_t30_150M_UR50D','a695f6045e2e32885fa60af20c13cb35398ce30c',project/self.config['model']['esm_backbone_path'])]:
                p=put(modelroot/'model.safetensors',{'fixture':repo})
                rows.append({'repo':repo,'revision':revision,'file':p.name,'sha256':f.sha(p),'bytes':p.stat().st_size})
            modelmanifest=put(project/s['model_manifest_path'],rows)
            evidence[b]={'view_manifest_sha256':f.sha(views/'manifest.json'),'cache_manifest_sha256':f.sha(cache/'manifest.json'),
                'model_manifest_sha256':f.sha(modelmanifest),'seconds_per_optimizer_step':1.,'generation_seconds_per_row':.1,
                'c1_seconds_per_row':.1,'model_loading_seconds':1.,'asset_verification_seconds':1.,'trainable_parameters':10}
        standard=self.root/'standard'
        for name in ['validate_reproduction.py','task-contract.template.json','result-record.template.json']:put(standard/name,{'fixture':True})
        lock=put(self.scope/'resources/env.json',{'fixture':True})
        ready=put(self.scope/'resources/ready.json',{'utc':self.now.isoformat(),'hostname':'node1','gpu':4,'readiness':'ready','reasons':[],
            'cpu_idle_percent':10.,'cpu_iowait_percent':0.,'memory_available_bytes':100*f.GIB,'storage_available_bytes':100*f.GIB,
            'gpu_frame':{'used_mib':15,'utilization':0},'gpu_compute_processes':[]})
        template_value=json.loads((SCOPE/'docs/evaluation_protocol.json').read_text(encoding='utf8'));template_value['data_sha256']=datas
        template=put(self.scope/'docs/evaluation_template.json',template_value)
        args=SimpleNamespace(project_root=str(project),input_project_root=str(old),scope_root=str(self.scope),standard_root=str(standard),
            python=sys.executable,host='node1',environment_lock=str(lock),canary_config=str(self.config_path),evaluation_template=str(template),tag='fixture_only',
            canary=[(b,'not_real_manifest','not_real_archive') for b in f.BACKBONES],ready=[('4',str(ready))],epochs=10,budget_change_reason=None,case_timeout_seconds=None)
        with patch.object(f,'accepted_canary',side_effect=lambda b,*_a:evidence[b]),patch.object(ev,'DATA_SHA256',datas):
            receipt=f.build(args,now_fn=lambda:self.now)
        queue=json.loads(Path(receipt['outputs']['queue']['path']).read_text(encoding='utf8'))
        registry=json.loads(Path(receipt['outputs']['registry']['path']).read_text(encoding='utf8'))
        self.assertEqual(len(registry['cases']),24);self.assertFalse(registry['old_results_counted'])
        self.assertEqual(queue['candidate_gpu_devices'],list(range(8)))
        self.assertIn(str(self.scope/'src/mm_data_evaluation.py'),queue['source_pins'])
        self.assertEqual(queue['paths']['cache_root'],str(cache))
        self.assertEqual(queue['paths']['data_root'],str(old/'data/processed/bench_v02/a1_locked_v2'))
        self.assertEqual(receipt['common_budget']['epochs'],10)
        self.assertFalse(receipt['formal_models_executed_by_freezer'])
        with patch.object(f,'accepted_canary',side_effect=lambda b,*_a:evidence[b]),patch.object(ev,'DATA_SHA256',datas):
            with self.assertRaisesRegex(ValueError,'already exists'):f.build(args,now_fn=lambda:self.now)


if __name__=='__main__':unittest.main()

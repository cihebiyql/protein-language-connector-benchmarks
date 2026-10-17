"""Prospectively freeze the v4 matrix after two canonically accepted TRAIN canaries.

Run on the execution host. All roots are explicit; no SSH, model imports, training,
background process, existing-file modification, or score-dependent budget selection.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

SCOPE_ID = 'backbone_comparison_v4_20261017'
DEADLINE = '2026-10-18T03:26:57+08:00'
FIRST = '2026-10-17T17:26:57+08:00'
BACKBONES = ('qwen25_1p5b', 'qwen3_4b_2507')
GIB = 1024**3


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition: raise ValueError(message)


def finite_positive(value):
    return type(value) in (int,float) and math.isfinite(value) and value > 0


def write_new(path, value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def artifact(root, ref):
    require(isinstance(ref,dict) and set(ref)=={'path','sha256'},'invalid artifact reference')
    path=(root/ref['path']).resolve(strict=True)
    require(path.is_relative_to(root.resolve()) and sha(path)==ref['sha256'],'artifact escape or SHA mismatch')
    return path


def gpu_and_supervision(manifest_path, contract_path):
    job=Path(manifest_path).resolve().parent.parent
    receipt=read(job/'supervision/receipt.json');cap_path=job/'gpu_limit.json';cap=read(cap_path)
    contract=read(contract_path);command=contract['execution']['command']
    value=lambda flag:command[command.index(flag)+1]
    require(receipt.get('status')=='succeeded' and receipt.get('exit_code')==0
        and receipt.get('remaining_group_members')==[] and receipt.get('contract_sha256')==sha(contract_path),
        'canary supervision did not exit and clean its exact owned group')
    require(Path(value('--receipt')).resolve()==cap_path.resolve()
        and Path(value('--output')).resolve()==Path(manifest_path).resolve().parent
        and value('--phase')=='train-canary' and value('--max-updates')=='4'
        and value('--module')=='src.mm_train' and value('--gpu-memory-budget-bytes')==str(20*GIB),
        'canary GPU worker command differs from exact qualification')
    devices=contract['resources']['gpu_devices']
    require(len(devices)==1 and cap.get('visible_devices')==str(devices[0])
        and cap.get('allocator_budget_bytes')==20*GIB and cap.get('module')=='src.mm_train'
        and all(type(cap.get(k)) is int and 0<cap[k]<=20*GIB for k in ('peak_allocated_bytes','peak_reserved_bytes')),
        'actual GPU cap/peaks/visible assignment not verified')
    return cap_path


def archive_existing_canary(backbone,manifest_path,archive_root,standard_root, *, archive_fn=None):
    """Model-free canonical archive of an already exited TRAIN run; no retry or fit."""
    manifest_path=Path(manifest_path).resolve(strict=True);archive_root=Path(archive_root).resolve()
    require(not archive_root.exists(),'new canonical canary archive must not overwrite existing output')
    artifacts=manifest_path.parent;job=artifacts.parent
    contract=job/'task-contract.json';manifest=read(manifest_path)
    require(manifest.get('scope_id')==SCOPE_ID and manifest.get('status')=='complete'
        and manifest.get('backbone')==backbone and manifest.get('phase')=='development'
        and manifest.get('execution_phase')=='train-canary' and manifest.get('test_labels_used_for_fit') is False,
        'cannot archive an incomplete, cross-scope or formal canary')
    cap_path=gpu_and_supervision(manifest_path,contract)
    evidence_root=archive_root.with_name(archive_root.name+'_gpu_evidence')
    evidence_root.mkdir(parents=True,exist_ok=False)
    copied=evidence_root/'gpu_limit.json'
    with copied.open('xb') as stream:stream.write(cap_path.read_bytes())
    require(sha(copied)==sha(cap_path),'GPU receipt changed during archive evidence copy')
    pairs=[('run_manifest.json','/scope_id',SCOPE_ID),('run_manifest.json','/status','complete'),
        ('run_manifest.json','/backbone',backbone),('run_manifest.json','/phase','development'),
        ('run_manifest.json','/execution_phase','train-canary'),('run_manifest.json','/test_labels_used_for_fit',False),
        ('run_manifest.json','/training/optimizer_steps',4),('run_manifest.json','/training/examples_seen',16),
        ('run_manifest.json','/runtime/training_performance/validation_used',False),
        ('train_canary_metrics.json','/rows',8),('train_canary_metrics.json','/split','train'),
        ('train_canary_metrics.json','/numeric/n_finite',8),('train_canary_metrics.json','/numeric/n_failed',0),
        ('train_anchor_qualification.json','/passed',True),
        ('max_length_generation_qualification.json','/split','train'),('max_length_generation_qualification.json','/rows',8),
        ('train_length_qualification.json','/source_split','train'),('train_length_qualification.json','/validation_or_test_opened',False)]
    completion={'artifact_root':str(artifacts),'assertions':[{'file':file,'pointer':pointer,'equals':expected} for file,pointer,expected in pairs],
        'target_counts':{'registered_endpoint':{'file':'train_canary_metrics.json','pointer':'/rows'}}}
    if archive_fn is None:
        from src.mm_archive import archive
        archive_fn=archive
    result=archive_fn(contract,job/'supervision',[artifacts,evidence_root],standard_root,archive_root,completion_spec=completion)
    require(result.get('status')=='accepted','canonical TRAIN canary terminal archive rejected')
    return result


def accepted_canary(backbone, manifest_path, archive_root, config, config_path, scope):
    manifest_path, archive_root=Path(manifest_path).resolve(strict=True),Path(archive_root).resolve(strict=True)
    require(manifest_path.is_relative_to(scope) and archive_root.is_relative_to(scope),'canary evidence must belong to the new isolated scope')
    manifest=read(manifest_path); receipt=read(archive_root/'archive_receipt.json')
    terminal=read(archive_root/'terminal_validation.json'); record=read(archive_root/'result-record.json')
    contract=Path(record['contract_path']).resolve(strict=True)
    require(receipt.get('status')=='accepted' and terminal.get('valid') is True
        and terminal.get('valid_for_acceptance') is True and not terminal.get('files_skipped')
        and record.get('review',{}).get('status')=='accepted' and record.get('lifecycle_status')=='succeeded'
        and receipt.get('record_sha256')==sha(archive_root/'result-record.json')
        and receipt.get('validation_sha256')==sha(archive_root/'terminal_validation.json')
        and receipt.get('contract_sha256')==record.get('contract_sha256')==sha(contract), 'canary terminal acceptance or SHA binding missing')
    contract_data=read(contract)
    gpu_path=gpu_and_supervision(manifest_path,contract)
    require(contract_data.get('task_id')==SCOPE_ID and record.get('task_id')==SCOPE_ID,'canary belongs to another scope')
    inventory=record.get('artifacts',[])
    require(any(Path(x.get('path','')).resolve()==manifest_path and x.get('sha256')==sha(manifest_path) for x in inventory), 'accepted archive does not bind exact canary manifest')
    require(manifest.get('scope_id')==SCOPE_ID and manifest.get('backbone')==backbone
        and manifest.get('execution_phase')=='train-canary' and manifest.get('phase')=='development'
        and manifest.get('status')=='complete' and manifest.get('test_labels_used_for_fit') is False
        and manifest.get('protocol_sha256')==sha(config_path)
        and manifest.get('backbone_config_sha256')==digest(config['backbones'][backbone]), 'canary identity/config differs')
    require(manifest.get('scenario')=='new_peptide_known_target' and manifest.get('arm')=='M3'
        and manifest.get('seed')==17,'expected TRAIN canary identity missing')
    source={x.name:sha(x) for x in sorted((scope/'src').glob('*.py'))}
    require(manifest.get('producer_source_sha256')==source['mm_train.py']
        and manifest.get('provenance',{}).get('code_sha256')==source,'source changed after qualifying canary')
    perf=manifest.get('runtime',{}).get('training_performance',{})
    training=manifest.get('training',{})
    require(training.get('optimizer_steps')==4 and training.get('examples_seen')==16
        and perf.get('validation_used') is False and perf.get('test_labels_used_for_fit') is False
        and finite_positive(perf.get('elapsed_seconds')), 'actual TRAIN performance/updates missing')
    policy=manifest.get('model_policy',{})
    require(type(policy.get('trainable_parameters')) is int and policy['trainable_parameters']>0
        and policy.get('frozen_base_eval') is True and policy.get('trainable_dtype')=='torch.float32'
        and policy.get('base_dtype')=='torch.bfloat16','exact CUDA frozen-base policy not qualified')
    peak=perf.get('peak_cuda_memory_bytes')
    require(type(peak) is int and 0<peak<=20*GIB,'canary peak memory exceeds common GPU budget')
    require(all(finite_positive(manifest['runtime'].get(k)) for k in ('canary_generation_seconds','model_loading_seconds','asset_verification_seconds')),'finite actual stage timings required')
    audit=manifest['provenance']['audit_artifacts']; root=manifest_path.parent
    anchor=read(artifact(root,audit['train_anchor_qualification']))
    require(anchor.get('passed') is True and anchor.get('split')=='train','answer-prefix numeric causality qualification failed')
    length=read(artifact(root,audit['train_length_qualification']))
    require(length.get('source_split')=='train' and length.get('all_train_rows_inspected')==2062
        and length.get('validation_or_test_opened') is False and len(length.get('first_effective_batch_indices',[]))==4
        and length.get('first_effective_batch_is_optimizer_update')==1,'longest TRAIN effective-batch qualification missing')
    longest=read(artifact(root,audit['max_length_generation_qualification']))
    require(longest.get('split')=='train' and longest.get('rows')==config['training']['generation_batch_size']
        and longest.get('max_new_tokens')==24 and longest.get('development_only') is True,'longest TRAIN generation qualification differs')
    metrics=read(artifact(root,audit['train_canary_metrics']))
    require(metrics.get('split')=='train' and metrics.get('rows')==8 and metrics.get('development_only') is True
        and metrics.get('numeric',{}).get('n_finite')==8 and metrics.get('numeric',{}).get('n_failed')==0,
        'TRAIN numeric serialization population incomplete')
    require(finite_positive(metrics.get('c1_seconds')),'finite actual C1 timing required')
    return {'backbone':backbone,'manifest_path':str(manifest_path),'manifest_sha256':sha(manifest_path),
        'gpu_limit_receipt_sha256':sha(gpu_path),'archive_root':str(archive_root),'archive_receipt_sha256':sha(archive_root/'archive_receipt.json'),
        'training_seconds':perf['elapsed_seconds'],'optimizer_steps':training['optimizer_steps'],
        'view_manifest_sha256':manifest['provenance']['view_manifest_sha256'],
        'cache_manifest_sha256':manifest['provenance']['cache_before_run']['manifest_sha256'],
        'model_manifest_sha256':manifest['provenance']['model_assets']['manifest_sha256'],
        'seconds_per_optimizer_step':perf['elapsed_seconds']/training['optimizer_steps'],
        'generation_seconds_per_row':manifest['runtime']['canary_generation_seconds']/8,
        'c1_seconds_per_row':metrics['c1_seconds']/8,'model_loading_seconds':manifest['runtime']['model_loading_seconds'],
        'asset_verification_seconds':manifest['runtime']['asset_verification_seconds'],
        'peak_cuda_memory_bytes':peak,'trainable_parameters':policy['trainable_parameters']}


def ready_assignment(gpu,path,host,scope,now):
    require(type(gpu) is int and 0<=gpu<=7,'GPU outside authorized candidate set')
    path=Path(path).resolve(strict=True); obs=read(path)
    age=(now-datetime.fromisoformat(obs['utc'].replace('Z','+00:00'))).total_seconds()
    frame=obs.get('gpu_frame',{})
    require(path.is_relative_to(scope) and obs.get('readiness')=='ready' and obs.get('hostname')==host
        and obs.get('gpu')==gpu and obs.get('reasons')==[] and 0<=age<=180
        and obs.get('cpu_idle_percent',-1)>=8 and 0<=obs.get('cpu_iowait_percent',100)<=10
        and obs.get('memory_available_bytes',0)>=36*GIB and obs.get('storage_available_bytes',0)>=10*GIB
        and not obs.get('gpu_compute_processes') and frame.get('used_mib',1000)<=256
        and frame.get('utilization',100)<=5,'initial ready evidence stale, incomplete, or busy')
    return {'gpu':gpu,'host':host,'readiness':'ready','evidence':{'path':str(path),'sha256':sha(path)}}


def build(args, *, now_fn=None):
    require(all(Path(getattr(args,k)).is_absolute() for k in ('project_root','input_project_root','scope_root','standard_root','python','environment_lock','canary_config','evaluation_template')),'all execution/input roots must be explicitly absolute')
    project=Path(args.project_root).resolve(strict=True); old=Path(args.input_project_root).resolve(strict=True)
    scope=Path(args.scope_root).resolve(strict=True); standard=Path(args.standard_root).resolve(strict=True)
    require(scope.is_relative_to(project) and scope!=project,'new scope must belong to new project')
    require(project!=old and not project.is_relative_to(old) and not old.is_relative_to(project),'new and canonical old projects must be isolated')
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,32}',args.tag) is not None,'invalid freeze tag')
    require(args.host=='node1','this authorized round prefers node1; fallback needs independent resource qualification')
    config_path=Path(args.canary_config).resolve(strict=True); config=read(config_path)
    require(config.get('schema_version')=='mm-v4.protocol.v1' and config.get('deadline')==DEADLINE
        and config.get('first_preparation')==FIRST,'wrong scope/deadline config')
    require(config['training']['micro_batch_size']==2 and config['training']['grad_accumulation']==2
        and config['training']['effective_batch_size']==4 and config['training']['generation_batch_size']==8,
        'these canaries must qualify common micro2/accum2/effective4/gen8')
    require(type(args.epochs) is int and 1<=args.epochs<=10,'epochs must fit prospective common budget')
    require(args.epochs==10 or bool(args.budget_change_reason),'common resource-driven epoch change needs explicit reason before test')
    require(len(args.canary)==2 and {x[0] for x in args.canary}==set(BACKBONES),'exactly two distinct accepted canaries required')
    if getattr(args,'archive_existing_canaries',False):
        sys.path.insert(0,str(scope))
        for b,m,a in args.canary:
            if not Path(a).exists():archive_existing_canary(b,m,a,standard)
    evidence={b:accepted_canary(b,m,a,config,config_path,scope) for b,m,a in args.canary}
    sys.path.insert(0,str(scope))
    from src.mm_evaluate import DATA_SHA256,COUNTS,validate_backbones,validate_protocol
    from src.mm_queue import fixed_cases,validate_plan
    validate_backbones(config['backbones'],formal=True)
    data=old/'data/processed/bench_v02/a1_locked_v2'
    oldscope=old/'experiments/molecular_memory_v3_20261017'
    views,cache=oldscope/'data_views',oldscope/'cache/paired_esm15_30_v1'
    assets={};datasets=[];models=[]
    def pin(path,expected=None):
        path=Path(path).resolve(strict=True); actual=sha(path)
        require(expected is None or expected==actual,'actual frozen input SHA mismatch: '+str(path))
        assets[str(path)]=actual
        return path
    for scenario,names in DATA_SHA256.items():
        for name,expected in names.items():
            path=pin(data/scenario/name,expected);datasets.append({'name':scenario+'/'+name,'path':str(path),'sha256':expected})
    view_path=pin(views/'manifest.json');cache_path=pin(cache/'manifest.json')
    require(all(c['view_manifest_sha256']==sha(view_path) and c['cache_manifest_sha256']==sha(cache_path) for c in evidence.values()),'canonical cache/views differ from accepted canaries')
    view_manifest=read(view_path);cache_manifest=read(cache_path)
    require(cache_manifest.get('cache_status')=='complete' and cache_manifest.get('completed_sequences')==2506
        and cache_manifest.get('raw_residues_covered')==408342,'canonical ESM cache incomplete')
    for item in cache_manifest['sequences'].values(): pin(cache/item['path'],item['sha256'])
    for scenario,item in view_manifest['scenarios'].items():
        require(item['source_shas']==DATA_SHA256[scenario],'canonical view source mismatch')
        for relative,ref in item['pins'].items():pin(old/relative,ref['sha256'])
        for split,ref in item['views'].items():pin(views/ref['path'],ref['sha256'])
    for backbone,spec in config['backbones'].items():
        manifest_path=pin(project/spec['model_manifest_path'])
        require(sha(manifest_path)==evidence[backbone]['model_manifest_sha256'],'model manifest differs from qualifying canary')
        records=read(manifest_path)
        for row in records:
            require(row['repo'] in (spec['repo'],'facebook/esm2_t30_150M_UR50D'),'unexpected model repository')
            root=project/(spec['backbone_path'] if row['repo']==spec['repo'] else config['model']['esm_backbone_path'])
            candidate=(root/row['file']).resolve(strict=True)
            require(candidate.is_relative_to(root.resolve()),'model manifest path escapes root')
            expected_revision=spec['revision'] if row['repo']==spec['repo'] else 'a695f6045e2e32885fa60af20c13cb35398ce30c'
            require(row['revision']==expected_revision,'model revision changed after canary')
            path=pin(candidate,row['sha256'])
            require(path.stat().st_size==row['bytes'],'model size mismatch')
            if row['file'].endswith('.safetensors'):
                models.append({'name':row['repo']+'/'+row['file'],'path':str(path),'sha256':row['sha256'],'coverage':'exact frozen pretrained shard'})
    env=pin(args.environment_lock)
    for name in ('validate_reproduction.py','task-contract.template.json','result-record.template.json'):pin(standard/name)
    # Preserve lexical paths under the new scope, even when pretrained inputs resolve to canonical old files.
    sources={str(path):sha(path) for path in sorted((scope/'src').glob('*.py'))}
    require(str(scope/'src/mm_data_evaluation.py') in sources,'evaluation-only data boundary source not pinned')
    sources[str(Path(__file__).resolve())]=sha(__file__)
    now=(now_fn or (lambda:datetime.now(timezone.utc)))()
    remaining=(datetime.fromisoformat(DEADLINE)-now).total_seconds()
    require(remaining>240,'insufficient remaining authorized window to freeze')
    ready=[ready_assignment(int(g),p,args.host,scope,now) for g,p in args.ready]
    require(ready and len({x['gpu'] for x in ready})==len(ready),'at least one unique actual ready GPU required')
    for assignment in ready:pin(assignment['evidence']['path'],assignment['evidence']['sha256'])
    tag=args.tag
    outputs={'config':scope/f'configs/formal_{tag}.json','evaluation_protocol':scope/f'docs/evaluation_{tag}.json',
        'queue':scope/f'configs/queue_{tag}.json','registry':scope/f'docs/registry_{tag}.json',
        'receipt':scope/f'docs/freeze_{tag}.json'}
    require(all(not p.exists() for p in outputs.values()),'freeze output already exists; use a new explicit version')
    frozen=deepcopy(config);frozen.update(status='frozen',frozen_at=now.isoformat())
    frozen['training']['epochs']=args.epochs; frozen['budget_signature']=digest(frozen['training'])
    frozen['backbone_signatures']={b:digest(v) for b,v in frozen['backbones'].items()}
    frozen['qualified_interface_parameter_counts']={b:c['trainable_parameters'] for b,c in evidence.items()}
    frozen['formal_budget_evidence']={'canaries':evidence,'decision_basis':'actual TRAIN time/memory; no validation/test score selection',
        'common_change_reason':args.budget_change_reason,'new_scope_does_not_reopen_old_round':True}
    frozen.pop('development_canary',None)
    protocol=read(args.evaluation_template);protocol.update(status='FROZEN',backbones=frozen['backbones'],
        evaluator_source_sha256=sources[str(scope/'src/mm_evaluate.py')],producer_source_sha256=sources[str(scope/'src/mm_train.py')])
    protocol['generation']['batch_size']=8
    expected={scenario:{'epochs_completed':args.epochs,'optimizer_steps':math.ceil(count['train']/4)*args.epochs,'examples_seen':count['train']*args.epochs} for scenario,count in COUNTS.items()}
    # Write only after all expensive actual-byte and accepted-canary checks pass.
    write_new(outputs['config'],frozen);pin(outputs['config'])
    protocol['training_budget'].update(status='FROZEN',training=frozen['training'],training_budget_sha256=frozen['budget_signature'],
        config_path=str(outputs['config'].relative_to(scope)),config_sha256=sha(outputs['config']),expected_training=expected)
    validate_protocol(protocol,formal=True);write_new(outputs['evaluation_protocol'],protocol);pin(outputs['evaluation_protocol'])
    estimates={}
    for b,c in evidence.items():
        estimates[b]={sc:expected[sc]['optimizer_steps']*c['seconds_per_optimizer_step']*1.5
            + COUNTS[sc]['test']*(c['generation_seconds_per_row']+c['c1_seconds_per_row'])
            + c['model_loading_seconds']+c['asset_verification_seconds']+120 for sc in COUNTS}
    timeout=args.case_timeout_seconds or math.ceil(max(v for group in estimates.values() for v in group.values())*1.5+300)
    timeout=min(timeout,int(remaining-120));require(timeout>0,'no positive case timeout')
    paths={'python':str(Path(args.python).absolute()),'project_root':str(project),'scope_root':str(scope),'config':str(outputs['config']),
        'cache_root':str(cache),'view_root':str(views),'evaluation_protocol':str(outputs['evaluation_protocol']),
        'data_root':str(data),'standard_root':str(standard)}
    request={'project_root':str(project),'scope_root':str(scope),'environment_path':str(Path(args.python).absolute().parent.parent),
        'environment_lock':str(env),'code_paths':list(sources),'datasets':datasets,'models':list({m['path']:m for m in models}.values()),
        'objective':'New fixed split Qwen2.5-1.5B versus Qwen3-4B-Instruct-2507; within-backbone M0/M3,24 fresh fits, original strict scoring',
        'level':'component','checkpoint_selection':'full validation answer-token NLL minimum; earliest exact tie',
        'metric':'strict_LM_and_independent_numeric_test_vector','unit':'frozen measurement',
        'metric_definition':'Full locked population strict JSON coverage/MAE plus separate numeric MAE/RMSE/Spearman/target macro and same-theta C1; no score repair',
        'resources':{'host':args.host,'cpu_threads':2,'memory_limit_bytes':32*GIB,'gpu_memory_budget_bytes':20*GIB},
        'numerical_policy':{'parameter_dtype':'Frozen LM bfloat16; trainable interface/head float32','input_dtype':'ESM cache float32; IDs int64',
            'autocast':False,'float32_matmul_precision':'highest','cuda_matmul_allow_tf32':False,'cudnn_allow_tf32':False,
            'determinism':'True; eager attention; CUBLAS_WORKSPACE_CONFIG=:4096:8','applied_before_model_instantiation':True}}
    queue={'schema_version':'mm-v4.queue.v1','status':'FROZEN','deadline':DEADLINE,'paths':paths,'source_pins':sources,'asset_pins':assets,
        'ready_assignments':ready,'candidate_gpu_devices':list(range(8)),'candidate_probe_interval_seconds':20,
        'occupied_gpu_devices':[g for g in range(8) if g not in {x['gpu'] for x in ready}],
        'occupied_gpu_devices_semantics':'initial not-ready snapshot; no permanent candidate blacklist',
        'gpu_memory_budget_bytes':20*GIB,'timeout_seconds_per_case':timeout,'resource_retry':{'max_attempts':3,'interval_seconds':20},
        'contract_request':request}
    validate_plan(queue);write_new(outputs['queue'],queue)
    registry={'scope_id':SCOPE_ID,'n_expected':24,'cases':[{**x,'status':'not_started'} for x in fixed_cases()],
        'failed_and_missing_retained':True,'old_results_counted':False,'order':'seed_scenario_backbone_variant'}
    write_new(outputs['registry'],registry)
    receipt={'scope_id':SCOPE_ID,'frozen_at':now.isoformat(),'deadline':DEADLINE,'canary_evidence':evidence,
        'estimates_seconds':estimates,'estimate_limit':'TRAIN canaries approximate validation/evaluation cost; not a completion guarantee',
        'queue_max_case_timeout_seconds':timeout,'all24_estimated_gpu_seconds':sum(sum(x.values()) for x in estimates.values())*6,
        'common_budget':frozen['training'],'source_pins':sources,'outputs':{k:{'path':str(p),'sha256':sha(p)} for k,p in outputs.items() if k!='receipt'},
        'launch_command':[str(Path(args.python).absolute()),'-m','src.mm_queue','--plan',str(outputs['queue']),'--output',str(scope/'runs'/f'formal_{tag}')],
        'freezer_source_sha256':sha(__file__),'formal_models_executed_by_freezer':False}
    write_new(outputs['receipt'],receipt)
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project-root','input-project-root','scope-root','standard-root','python','host','environment-lock','canary-config','evaluation-template','tag'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--canary',nargs=3,action='append',required=True,metavar=('BACKBONE','MANIFEST','ACCEPTED_ARCHIVE'))
    p.add_argument('--ready',nargs=2,action='append',required=True,metavar=('GPU','FRESH_PROBE'))
    p.add_argument('--archive-existing-canaries',action='store_true',help='Create new canonical archives of existing complete/exited TRAIN canaries only; no model execution')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--budget-change-reason')
    p.add_argument('--case-timeout-seconds',type=int)
    args=p.parse_args(argv)
    require(args.case_timeout_seconds is None or args.case_timeout_seconds>0,'positive timeout required')
    receipt=build(args)
    print(json.dumps({'status':'frozen','outputs':receipt['outputs'],'launch_command':receipt['launch_command']},ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())

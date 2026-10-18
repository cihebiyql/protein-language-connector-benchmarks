"""Finish technical qualification and freeze once, before formal evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)

def freeze(scope):
    scope=Path(scope).resolve()
    if (scope/'frozen_manifest.json').exists():raise FileExistsError('A frozen experiment already exists')
    if list((scope/'runs').glob('*/formal')):raise RuntimeError('Formal experiment already exists')
    config=read(scope/'config.json')
    prepared=read(scope/'private/preparation_v1/complete.json')
    if prepared['status']!='technical_preparation_complete':raise ValueError('Preparation incomplete')
    cpu_env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    log=scope/('private/qualification_cpu_tests_'+datetime.now(timezone.utc).strftime('%H%M%S')+'.log')
    with log.open('x') as stream:
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=scope,
            env=cpu_env,stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,check=True,timeout=300)
    smokes={};counts=set();memory_peak=0
    for arm in config['arms']:
        run=scope/'runs'/f'{arm}__seed17'/'smoke';r=read(run/'completion.json')
        if r['status']!='complete' or r['optimizer_updates']!=4 or r['training_examples_seen']!=32:
            raise ValueError('Technical smoke incomplete')
        if r['frozen']['trainer_sha256']!=sha(scope/'src/qa_train.py'):
            raise ValueError('Trainer differs from qualified source')
        for name,item in r['artifacts'].items():
            if sha(run/name)!=item['sha256']:raise ValueError('Smoke artifact changed')
        history=read(run/'training_history.json')
        if any(not math.isfinite(e['mean_grad_norm']) or e['mean_grad_norm']<=0 for e in history):
            raise ValueError('No finite nonzero adapter gradient')
        counts.add(r['capacity']['trainable_parameters'])
        memory_peak=max(memory_peak,r['max_gpu_reserved_bytes'])
        smokes[arm]={'completion_sha256':sha(run/'completion.json'),'optimizer_updates':4,
                    'TRAIN_examples_seen':32,'max_gpu_reserved_bytes':r['max_gpu_reserved_bytes'],
                    'adapter_parameters':r['capacity']['trainable_parameters'],'no_validation_or_test_selection':True}
    if len(counts)!=1:raise ValueError('Adapter capacities differ')
    if config['min_free_memory_mib']*1024**2<memory_peak+1024**3:raise ValueError('Resource minimum below measured qualification peak plus headroom')
    # Verify old frozen data remain byte-identical without reusing any old labels.
    runtime=read(scope/'private/runtime.json')
    old=Path(runtime['previous_model_manifest']).parents[1]
    data_manifest=read(scope/'data/manifest.json')
    for name,expected in data_manifest['old_v5_frozen_file_pins'].items():
        if sha(old/Path(name.replace('\\','/')))!=expected:raise ValueError('Old frozen data changed')
    write(scope/'resources/qualification_receipt.json',{
        'schema':'same-peptide-technical-qualification.v1','status':'passed','at_utc':datetime.now(timezone.utc).isoformat(),
        'cpu_tests_exit_code':0,'cpu_test_log_sha256':sha(log),'smokes':smokes,
        'common_adapter_parameters':next(iter(counts)),'maximum_smoke_reserved_bytes':memory_peak,
        'old_v5_frozen_data_unchanged':True,'formal_results_observed':False,
        'data_manifest_sha256':sha(scope/'data/manifest.json'),
        'features_sha256':sha(scope/'cache/features.npz'),'attributes_sha256':sha(scope/'cache/attributes.npz')})
    write(scope/'private/config_before_freeze.json',config)
    config['status']='frozen'
    write(scope/'config.json',config)
    write(scope/'ACTIVE_EXPERIMENT.json',{'schema':'same-peptide-active.v1','status':'qualified_frozen_not_formal_started',
        'expected_cases':9,'accepted_cases':0,'seeds':config['seeds'],'arms':config['arms'],
        'scope_id':scope.name,'node1_independent':True,'scheduled_monitor':False,
        'stop_rule':'config.json decision','old_v5_preserved':True})
    files=[]
    for directory in ('src','tests','resources'):
        files += list((scope/directory).glob('*.py'))
    files += list((scope/'data').glob('*.json'))+list((scope/'data').glob('*.jsonl'))
    files += [scope/name for name in ('config.json','PROTOCOL.md','README.md','cache/features.npz',
        'cache/attributes.npz','cache/features_receipt.json','cache/attributes_receipt.json',
        'resources/model_assets.json','resources/qualification_receipt.json')]
    manifest={'schema':'same-peptide-frozen-closure.v1','status':'frozen_before_formal_fits',
        'frozen_utc':datetime.now(timezone.utc).isoformat(),'config_sha256':sha(scope/'config.json'),
        'files':[{'path':p.relative_to(scope).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(set(files))]}
    write(scope/'frozen_manifest.json',manifest)
    print(json.dumps({'status':'qualified_and_frozen','files':len(manifest['files']),
        'manifest_sha256':sha(scope/'frozen_manifest.json'),'cpu_test_log_tail':log.read_text()[-160:]}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scope',required=True);a=p.parse_args()
    try:freeze(a.scope)
    except BaseException as error:
        import traceback
        write(Path(a.scope)/'private/freeze_failure.json',{'error':str(error),'traceback':traceback.format_exc()})
        print(json.dumps({'status':'freeze_failed','error':str(error)}),flush=True)
        sys.exit(1)

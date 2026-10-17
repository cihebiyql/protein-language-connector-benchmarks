"""Freeze45 fresh fits from actual TRAIN timing; no validation/test score selection."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

SCOPE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCOPE))
from src.mt_queue import validate_plan


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',required=True,type=int)
    p.add_argument('--tag',required=True)
    p.add_argument('--reason-file',required=True)
    args=p.parse_args()
    args.reason=Path(args.reason_file).read_text(encoding='utf-8').strip()
    if not args.reason:raise ValueError('Explicit resource budget decision required')
    if not 1<=args.epochs<=5 or not args.tag.replace('_','').isalnum():raise ValueError('Invalid prospective budget/tag')
    evidence=read(SCOPE/'runs/qualification_v2/complete.json')
    if evidence['status']!='qualified' or evidence['case_count']!=6:raise ValueError('Actual representative TRAIN qualification incomplete')
    expected={(t,m) for t in ('elm','rt','meltome') for m in ('ESM','M3')}
    if {(r['case']['task'],r['case']['method']) for r in evidence['cases']}!=expected:raise ValueError('Wrong qualification identities')
    scientific={str(p):sha(p) for p in sorted((SCOPE/'src').glob('*.py'))}
    for result in evidence['cases']:
        if result['status']!='qualified' or not result['no_validation_or_test_used'] or result['accuracy_used_for_admission']:
            raise ValueError('Qualification used a forbidden admission endpoint')
        path=Path(result['manifest_path'])
        if sha(path)!=result['manifest_sha256']:raise ValueError('Qualification manifest changed')
        manifest=read(path)
        if manifest['code_sha256']!={Path(k).name:v for k,v in scientific.items()}:raise ValueError('Scientific or execution sources changed after qualification')
        if manifest['training']['optimizer_steps']!=4 or manifest['training']['examples_seen']!=16:raise ValueError('Invalid actual TRAIN work')
    config=deepcopy(read(SCOPE/'configs/development_v2.json'))
    config.update(status='frozen',frozen_utc=datetime.now(timezone.utc).isoformat())
    config['training']['epochs']=args.epochs
    config['training_signature']=digest(config['training'])
    manifest=read(SCOPE/'data/manifest.json')
    config['expected_training']={task:{'epochs_completed':args.epochs,
        'optimizer_steps':math.ceil(item['counts']['train']/4)*args.epochs,
        'examples_seen':item['counts']['train']*args.epochs} for task,item in manifest['tasks'].items()}
    config['budget_evidence']={'qualification_receipt_sha256':sha(SCOPE/'runs/qualification_v2/complete.json'),
        'basis':'actual TRAIN time/memory only; no validation/test scores read', 'reason':args.reason,
        'each_formal_case_has_separate_TRAIN_qualification_before_fresh_fit':True,
        'qwen3_qualification':'performed when actual18GiB+headroom becomes available; common epochs already fixed, never changed by score'}
    config_path=SCOPE/'configs'/('formal_'+args.tag+'.json')
    plan_path=SCOPE/'configs'/('queue_'+args.tag+'.json')
    receipt_path=SCOPE/'docs'/('freeze_'+args.tag+'.json')
    if any(path.exists() for path in (config_path,plan_path,receipt_path)):raise FileExistsError('Frozen version already exists')
    save(config_path,config)
    assets=[config_path,SCOPE/'data/manifest.json',SCOPE/'cache/manifest.json',SCOPE/'cache/cache_spec.json',
        SCOPE/'cache/description_vectors.npz',SCOPE/'cache/description_vectors.json',
        SCOPE/'resources/model_manifest_qwen25.json',SCOPE/'resources/model_manifest_qwen3.json',
        SCOPE/'runs/qualification_v2/complete.json']
    assets += [SCOPE/'data'/name for name in manifest['files']]
    # Actual pretrained files are independently pinned, not only a model name.
    for spec in config['backbones'].values():
        for item in read(spec['model_manifest_path']):
            if item['repo']==spec['repo']:
                path=Path(spec['backbone_path'])/item['file']
                if sha(path)!=item['sha256']:raise ValueError('Actual official model asset mismatch')
                assets.append(path)
    cases=[{'case_id':f'{task}__{method}__{backbone}__seed{seed}', 'task':task,'method':method,
        'backbone':backbone,'seed':seed} for seed in (17,42,2026) for task in ('elm','rt','meltome')
        for method,backbone in (('ESM','none'),('M0','qwen25_1p5b'),('M3','qwen25_1p5b'),('M0','qwen3_4b_2507'),('M3','qwen3_4b_2507'))]
    plan={'scope_id':'multitask_connector_v5_20261017','status':'frozen','phase':'formal',
        'paths':{'python':sys.executable,'scope_root':str(SCOPE),'data_root':str(SCOPE/'data'),
            'cache_root':str(SCOPE/'cache'),'config':str(config_path)},
        'source_pins':scientific,'asset_pins':{str(path.resolve()):sha(path) for path in assets},
        'cases':cases,'gpu_devices':[3,4,5],'per_case_timeout_seconds':28800,
        'resource_wait_timeout_seconds':86400,'require_train_canary':True,
        'resource_policy':{'expected_host':'node1','cpu_threads':2,'rss_limit_bytes':64*1024**3,
            'cpu_rss_limit_bytes':16*1024**3,'storage_reserve_bytes':10*1024**3,
            'gpu_memory_budget_bytes':{'qwen25_1p5b':8*1024**3,'qwen3_4b_2507':18*1024**3}},
        'analysis_command':[sys.executable,'-m','src.mt_analyze','--plan',str(plan_path),
            '--queue-root',str(SCOPE/'runs'/args.tag),'--data-root',str(SCOPE/'data'),
            '--output',str(SCOPE/'analysis'/args.tag)],
        'scheduling_order':'seed_then_elm_rt_meltome_then_method; lower-memory ready work may advance while4B waits',
        'no_scheduled_monitor':True,'prior_experiments_preserved':True}
    validate_plan(plan)
    save(plan_path,plan)
    save(receipt_path,{'status':'frozen','frozen_utc':config['frozen_utc'],'case_count':45,
        'config_path':str(config_path),'config_sha256':sha(config_path),
        'plan_path':str(plan_path),'plan_sha256':sha(plan_path),'expected_training':config['expected_training'],
        'qualification_receipt_sha256':sha(SCOPE/'runs/qualification_v2/complete.json'),
        'training_signature':config['training_signature'],'source_pins':scientific,
        'parent_budget_decision':args.reason,'formal_models_executed_by_freezer':False})
    print(json.dumps({'status':'frozen','plan_sha256':sha(plan_path),'expected_training':config['expected_training']}),flush=True)


if __name__=='__main__':main()

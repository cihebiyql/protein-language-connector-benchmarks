"""Actual TRAIN-only resource qualification; never reads validation or test scores."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SCOPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCOPE))
from src.mt_queue import build_spec
from src.mt_supervise import supervise


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, obj):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj,stream,indent=2,allow_nan=False)
        stream.write('\n')


def qualify(plan, plan_path, case, gpu, root):
    attempt = root / case['case_id']
    attempt.mkdir(exist_ok=False)
    spec = build_spec(plan_path, plan, case, attempt, gpu)
    save(attempt/'spec.json', spec)
    code = supervise(attempt/'spec.json', attempt/'supervision')
    if code:
        raise RuntimeError('Actual TRAIN canary failed: '+case['case_id'])
    manifest = read(attempt/'artifacts/run_manifest.json')
    train, qualification = manifest['training'], manifest['canary_qualification']
    if (manifest['status'] != 'complete' or manifest['phase'] != 'train-canary'
        or train['optimizer_steps'] != 4 or train['examples_seen'] != 16
        or qualification['train_rows_scored'] != 8
        or any(qualification[key] for key in ('validation_opened','calibration_opened','test_opened'))):
        raise ValueError('Actual qualification does not match the prospective TRAIN-only contract')
    finished = read(attempt/'supervision/finished.json')
    if finished['exit_code'] != 0 or finished['remaining_group_members']:
        raise ValueError('Qualification group did not exit cleanly')
    files = list((attempt/'artifacts').glob('*'))
    result = {'case':case,'gpu':gpu,'status':'qualified',
        'manifest_path':str(attempt/'artifacts/run_manifest.json'),
        'manifest_sha256':sha(attempt/'artifacts/run_manifest.json'),
        'training':train,'runtime':manifest['runtime'], 'elapsed_seconds':manifest['elapsed_seconds'],
        'supervision':finished,'artifacts':{p.name:sha(p) for p in files if p.is_file()},
        'no_validation_or_test_used':True,'accuracy_used_for_admission':False}
    save(attempt/'qualification.json',result)
    print(json.dumps({'case_id':case['case_id'],'status':'qualified','training':train}),flush=True)
    return result


def main():
    features = read(SCOPE/'resources/feature_stage_complete_v1.json')
    if features['status'] != 'complete':raise RuntimeError('Feature stage incomplete')
    root = SCOPE/'runs/qualification_v1'
    root.mkdir(parents=True,exist_ok=False)
    config = SCOPE/'configs/development_v2.json'
    assets = [config,SCOPE/'data/manifest.json',SCOPE/'cache/manifest.json',SCOPE/'cache/cache_spec.json',
        SCOPE/'cache/description_vectors.npz',SCOPE/'cache/description_vectors.json',
        SCOPE/'resources/model_manifest_qwen25.json',SCOPE/'resources/model_manifest_qwen3.json']
    assets += [SCOPE/'data'/p for p in read(SCOPE/'data/manifest.json')['files']]
    cases = [{'case_id':task+'__'+method+'__'+backbone+'__seed17','task':task,'method':method,
        'backbone':backbone,'seed':17} for method,backbone in [('ESM','none'),('M3','qwen25_1p5b')]
        for task in ('meltome','elm','rt')]
    plan = {'scope_id':'multitask_connector_v5_20261017','status':'prospective_qualification',
        'phase':'train-canary','require_train_canary':False,
        'paths':{'python':sys.executable,'scope_root':str(SCOPE),'data_root':str(SCOPE/'data'),
            'cache_root':str(SCOPE/'cache'),'config':str(config)},
        'source_pins':{str(p):sha(p) for p in sorted((SCOPE/'src').glob('*.py'))},
        'asset_pins':{str(p):sha(p) for p in assets}, 'cases':cases,'gpu_devices':[3,4,5],
        'per_case_timeout_seconds':1800,'resource_wait_timeout_seconds':86400,
        'resource_policy':{'expected_host':'node1','cpu_threads':2,'rss_limit_bytes':64*1024**3,
            'cpu_rss_limit_bytes':16*1024**3,'storage_reserve_bytes':10*1024**3,
            'gpu_memory_budget_bytes':{'qwen25_1p5b':8*1024**3,'qwen3_4b_2507':18*1024**3}}}
    plan_path = root/'plan.json'
    save(plan_path,plan)
    results=[]
    def cpu():
        return [qualify(plan,plan_path,c,None,root) for c in cases if c['method']=='ESM']
    with ThreadPoolExecutor(max_workers=4) as executor:
        cpu_future=executor.submit(cpu)
        futures=[executor.submit(qualify,plan,plan_path,c,gpu,root)
            for c,gpu in zip([c for c in cases if c['method']=='M3'],(3,4,5))]
        for future in futures:results.append(future.result())
        results.extend(cpu_future.result())
    save(root/'complete.json',{'status':'qualified','utc':datetime.now(timezone.utc).isoformat(),
        'plan_sha256':sha(plan_path),'case_count':len(results),'cases':results,
        'budget_selection_uses_only_TRAIN_time_memory':True,'no_test_scores_inspected':True})
    print(json.dumps({'status':'qualified','cases':len(results)}),flush=True)


if __name__=='__main__':main()

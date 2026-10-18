"""One finite node1 job: qualify, extract, freeze names, test, report, publish."""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def identity(pid):
    v=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'session':int(v[3]),'start_ticks':int(v[19])}
@contextmanager
def lock(p):
    import fcntl
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield
def frozen(scope):
    m=read(scope/'frozen_manifest.json')
    for name,expected in m['source_pins'].items():
        p=(scope/name).resolve()
        if not p.is_relative_to(scope.resolve()) or sha(p)!=expected:raise RuntimeError('Frozen input integrity failure')
    return sha(scope/'frozen_manifest.json')
def gpu_check(gpu,minimum):
    if gpu!=4:raise RuntimeError('This pilot may use only physical GPU4')
    output=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    return int(output.strip())>=minimum
def protected_v6(scope):
    runtime=read(scope/'private/runtime.json');prior=Path(runtime['protected_v6_scope'])
    manifest=prior/'frozen_manifest.json'
    if sha(manifest)!=runtime['protected_v6_manifest_sha256']:raise RuntimeError('Protected v6 manifest changed')
    for item in read(manifest)['files']:
        if sha(prior/item['path'])!=item['sha256']:raise RuntimeError('Protected v6 frozen file changed')
    return True
def public_metadata(scope):
    data=read(scope/'data/manifest.json')
    write(scope/'data/PUBLIC_MANIFEST.json',{'schema':'motifae-panel-public-summary.v1','panel_sha256':sha(scope/'data/panel.jsonl'),
        'catalog_sha256':sha(scope/'data/catalog.json'),'manifest_sha256':sha(scope/'data/manifest.json'),
        'counts':data['summary']['splits'],'grouping':data['grouping'],'split_rule':data['split_rule'],
        'original_train_rows':data['summary']['original_train_rows'],'protected_old_validation_test_rows':data['summary']['original_protected_rows'],
        'old_validation_test_untouched':True,'all_raw_rows_private':True})
    assets=read(scope/'assets/DOWNLOAD_MANIFEST.json')
    write(scope/'reports/ASSET_PROVENANCE.json',{'author_commit':assets['official_repository_commit'],
        'assets':[{'name':v['file'],'sha256':v['sha256'],'bytes':v['bytes']} for v in assets['files']],
        'source_url':'https://zenodo.org/records/20547228','weights_kept_private':True})
def run(scope):
    scope=Path(scope).resolve();os.chdir(scope)
    if socket.gethostname().split('.')[0]!='node1':raise RuntimeError('Wrong execution host')
    with lock(scope/'private/pipeline.lock'):
        if (scope/'private/pipeline_finished.json').exists():raise RuntimeError('Completed pilot must not restart')
        config=read(scope/'config.json');gpu=config['gpu'];freeze_sha=frozen(scope);protected_v6(scope)
        status={'status':'running','started_at':now(),'host':'node1','pid':os.getpid(),'GPU':gpu,
            'frozen_manifest_sha256':freeze_sha,'stages':{},'local_callback_required':False,'scheduled_monitor':False}
        def save():status['updated_at']=now();write(scope/'STATUS.json',status)
        save();public_metadata(scope)
        env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2',
            'TOKENIZERS_PARALLELISM':'false','CUBLAS_WORKSPACE_CONFIG':':4096:8','HF_HUB_OFFLINE':'1',
            'PYTHONUNBUFFERED':'1'}
        def stage(name,args,requires_gpu=False,minimum=8000,timeout=7200):
            frozen(scope)
            if requires_gpu:
                until=time.monotonic()+6*3600
                while not gpu_check(gpu,minimum):
                    if time.monotonic()>until:raise RuntimeError('Resource wait exhausted')
                    status['status']='resource_wait';status['waiting_stage']=name;save();time.sleep(30)
                status['status']='running';status.pop('waiting_stage',None)
            status['stages'][name]={'status':'running','started_at':now()};save()
            logs=scope/'private/logs';logs.mkdir(parents=True,exist_ok=True)
            with (logs/(name+'.log')).open('x') as stream:
                child=subprocess.Popen([sys.executable,*args],cwd=scope,
                    env={**env,'CUDA_VISIBLE_DEVICES':str(gpu) if requires_gpu else ''},
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                status['stages'][name]['process']=identity(child.pid);save()
                try:code=child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    import signal
                    # This exact child was just created in its own session.
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30);raise
            status['stages'][name].update(status='complete' if code==0 else 'failed',exit_code=code,finished_at=now());save()
            if code:raise RuntimeError('Stage failed: '+name)
        failed_stage=None
        status['protected_v6_verified_before']=True
        try:
            stage('cpu_tests',['-m','unittest','discover','-s','tests','-v'],timeout=600)
            write(scope/'resources/CPU_TEST_RECEIPT.json',{'status':'passed','exit_code':0,'log_sha256':sha(scope/'private/logs/cpu_tests.log'),'frozen_manifest_sha256':freeze_sha})
            stage('qwen_synthetic_smoke',['src/qa_pilot.py','--scope',str(scope),'--phase','smoke'],True,12000)
            feature_args=['src/motif_features.py','--scope',str(scope),'--esm-checkpoint',config['esm_checkpoint'],
                '--motif-checkpoint',str(scope/'assets/MotifAE_k3_step_80000.pt')]
            stage('feature_extract',feature_args+['--stage','extract'],True,8000,timeout=10800)
            smoke=read(scope/'cache/reference_smoke.json')
            # Numeric smoke contains source fingerprints and aggregate errors;
            # publish only scalar diagnostics, not reference data or file paths.
            allowed={k:v for k,v in smoke.items() if isinstance(v,(int,float,bool)) or v is None}
            write(scope/'resources/GPU_SMOKE_PUBLIC.json',{'status':'passed','numeric_diagnostics':allowed,
                'private_smoke_sha256':sha(scope/'cache/reference_smoke.json'),'all_raw_reference_data_private':True})
            stage('dictionary_fit',['src/motif_dictionary.py','--scope',str(scope),'--phase','fit'],timeout=10800)
            stage('feature_test',['src/motif_dictionary.py','--scope',str(scope),'--phase','evaluate'])
            stage('qwen_test',['src/qa_pilot.py','--scope',str(scope),'--phase','formal'],True,12000,timeout=14400)
            stage('aggregate',['src/pilot_report.py','--scope',str(scope)])
        except BaseException as error:
            failed_stage=next((name for name,v in status['stages'].items() if v['status']!='complete'),'pipeline')
            import traceback
            write(scope/'private/failure.json',{'stage':failed_stage,'error':str(error),'traceback':traceback.format_exc()})
            # Technical failure remains distinct from a completed negative pilot.
            try:stage('technical_report',['src/pilot_report.py','--scope',str(scope),'--failure-stage',failed_stage])
            except Exception:pass
        try:
            stage('publication',['resources/publish_pilot.py','--scope',str(scope),'--stage','final'],timeout=1800)
        except Exception as error:
            write(scope/'private/publication_failure.json',{'type':type(error).__name__,'message':str(error)})
        status['protected_v6_verified_after']=protected_v6(scope)
        status['status']='complete' if not failed_stage and status['stages'].get('publication',{}).get('status')=='complete' else 'technical_attention_required'
        status['finished_at']=now();save();write(scope/'private/pipeline_finished.json',status)
        return status
def launch(scope):
    scope=Path(scope).resolve();frozen(scope)
    with lock(scope/'private/launcher.lock'):
        receipt=scope/'private/launch.json'
        if receipt.exists():raise RuntimeError('Prior launch preserved; no duplicate launch')
        with (scope/'private/parent.log').open('x') as stream:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--scope',str(scope),'--stage','run'],
                cwd=scope,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        result={'status':'launched','process':identity(child.pid),'at_utc':now(),'host':'node1',
            'frozen_manifest_sha256':sha(scope/'frozen_manifest.json'),'node1_independent':True,
            'local_callback_required':False,'scheduled_monitor':False}
        write(receipt,result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scope',required=True);p.add_argument('--stage',choices=['run','launch'],required=True)
    a=p.parse_args();print(json.dumps(launch(a.scope) if a.stage=='launch' else run(a.scope)),flush=True)

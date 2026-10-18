"""Finite technical preparation on node1, no scheduled monitor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)

def run(scope,gpu):
    scope=Path(scope).resolve();os.chdir(scope)
    if socket.gethostname().split('.')[0]!='node1':raise RuntimeError('Wrong host')
    config=json.loads((scope/'config.json').read_text())
    output=scope/'private/preparation_v1';output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists():raise FileExistsError('Preparation already completed')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
         'MKL_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8',
         'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1'}
    def stage(name,args,cuda=True):
        write(output/'status.json',{'stage':name,'pid':os.getpid(),'unix':time.time(),'gpu':gpu if cuda else None})
        if cuda:
            free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<(3000 if name=='extract' else 16000):raise RuntimeError('Insufficient free VRAM before '+name)
        with (output/(name+'.stdout.log')).open('xb') as out,(output/(name+'.stderr.log')).open('xb') as err:
            subprocess.run([sys.executable,*args],cwd=scope,env={**env,'CUDA_VISIBLE_DEVICES':str(gpu) if cuda else ''},
                stdin=subprocess.DEVNULL,stdout=out,stderr=err,check=True,timeout=7200)
    try:
        # Check both external assets ONCE; formal cases subsequently check file
        # identity metadata against this exact full-hash technical receipt.
        old_manifest=Path(config['model_path']).parents[2]/'experiments/multitask_connector_v5_20261017/resources/model_manifest_qwen3.json'
        # v5 lives in a different deployment root; use an explicit private reference.
        runtime=json.loads((scope/'private/runtime.json').read_text())
        pins=json.loads(Path(runtime['previous_model_manifest']).read_text())
        assets={}
        for key,repo in [('model_path',config['model_repo']),('esm_model_path','facebook/esm2_t30_150M_UR50D')]:
            root=Path(config[key]);entries=[]
            for item in pins:
                if item['repo']!=repo:continue
                p=root/item['file'];stat=p.stat()
                if stat.st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('Frozen model asset mismatch '+p.name)
                entries.append({'file':p.name,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':item['sha256']})
            if not entries or not any(x['file'].endswith('.safetensors') for x in entries):raise ValueError('Missing weight pins')
            assets[key]={'repo':repo,'revision':config.get('model_revision') if key=='model_path' else 'a695f6045e2e32885fa60af20c13cb35398ce30c','files':entries}
        write(scope/'resources/model_assets.json',assets)
        stage('extract',['src/qa_features.py','--scope',str(scope),'--stage','extract','--model-path',config['esm_model_path']])
        stage('specialists',['src/qa_features.py','--scope',str(scope),'--stage','specialists'],cuda=False)
        for arm in config['arms']:
            stage('smoke_'+arm,['src/qa_train.py','--scope',str(scope),'--arm',arm,'--seed','17','--phase','smoke'])
        write(output/'complete.json',{'status':'technical_preparation_complete','unix':time.time(),
            'arms':config['arms'],'each_smoke_updates':4,'formal_runs_started':False,'GPU':gpu,
            'model_assets_sha256':sha(scope/'resources/model_assets.json'),
            'features_sha256':sha(scope/'cache/features.npz'),'attributes_sha256':sha(scope/'cache/attributes.npz')})
        write(output/'status.json',{'stage':'complete','unix':time.time()})
    except BaseException as e:
        import traceback
        write(output/'failure.json',{'error':str(e),'traceback':traceback.format_exc(),'unix':time.time()});raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',required=True);p.add_argument('--gpu',type=int,default=6);p.add_argument('--launch',action='store_true');a=p.parse_args()
    scope=Path(a.scope).resolve()
    if a.launch:
        directory=scope/'private/preparation_v1';directory.mkdir(parents=True,exist_ok=True)
        if (directory/'launch.json').exists():raise FileExistsError('Previous launch preserved')
        with (directory/'parent.stdout.log').open('xb') as out,(directory/'parent.stderr.log').open('xb') as err:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--scope',str(scope),'--gpu',str(a.gpu)],
                cwd=scope,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,close_fds=True)
        fields=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()
        r={'pid':child.pid,'session':int(fields[3]),'start_ticks':int(fields[19]),'host':'node1','scheduled_monitor':False}
        write(directory/'launch.json',r);print(json.dumps(r),flush=True)
    else:run(scope,a.gpu)

if __name__=='__main__':main()

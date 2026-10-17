"""Launch one finite detached queue; never create a scheduled monitoring task."""
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True)
    p.add_argument('--queue-root',required=True)
    args=p.parse_args()
    scope=Path(__file__).resolve().parents[1]
    plan=Path(args.plan).resolve(strict=True)
    root=Path(args.queue_root).resolve()
    if not root.is_relative_to(scope/'runs') or root.exists():raise ValueError('New owned run directory required')
    binding=scope/'resources'/('launch_'+root.name+'.json')
    if binding.exists():raise FileExistsError('Existing launch identity retained; no duplicate queue')
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(scope))
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='2'
    command=[sys.executable,'-m','src.mt_queue','--plan',str(plan),'--output',str(root)]
    log=scope/'resources'/('controller_'+root.name+'.log')
    with log.open('xb') as stream:
        child=subprocess.Popen(command,cwd=scope,env=env,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    value={'status':'launched','utc':datetime.now(timezone.utc).isoformat(),'pid':child.pid,
        'pgid':os.getpgid(child.pid),'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),
        'queue_root':str(root),'gpu_devices':[3,4,5],'scheduled_monitor':False,'command':command}
    with binding.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(value),flush=True)

if __name__=='__main__':main()

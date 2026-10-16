"""Sequential continuation for this bounded local experiment, with visible status."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/real_run_20261015'

def status(phase,**extra):
    value={'updated_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
           'phase':phase,'controller_pid':os.getpid(),**extra}
    path=OUT/'suite_status.json'
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(path)
    print(json.dumps(value),flush=True)

def t1_complete(config):
    try:
        for seed in (17,42,2026):
            value=json.loads((OUT/f'{config}_seed{seed}/result.json').read_text(encoding='utf-8'))
            if value['optimizer_steps']!=900 or value['config']['seed']!=seed:
                return False
        return True
    except (FileNotFoundError,json.JSONDecodeError):
        return False

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-pid',type=int,required=True)
    args=parser.parse_args()
    status('waiting_for_L30_J0_stage_A',wait_pid=args.wait_pid)
    while not t1_complete('L30-J0'):
        try:
            os.kill(args.wait_pid,0)
        except ProcessLookupError:
            status('failed',reason='Original L30-J0 process exited before all three runs completed')
            raise SystemExit(1)
        time.sleep(5)
    tasks=[('L15_J14_stage_A',['scripts/train_real_bridge.py','--config','L15-J14','--microbatch','8'],'L15-J14.log'),
           ('L30_J0_stage_B',['scripts/train_t3_bridge.py','--config','L30-J0','--microbatch','8','--eval-batch','4'],'L30-J0_T3.log'),
           ('L15_J14_stage_B',['scripts/train_t3_bridge.py','--config','L15-J14','--microbatch','8','--eval-batch','4'],'L15-J14_T3.log'),
           ('final_comparison',['scripts/summarize_real_results.py','--require-complete','--report','reports/real_training_results.md'],'summary.log')]
    for name,arguments,log_name in tasks:
        status(name,arguments=arguments,log=str(OUT/log_name))
        with (OUT/log_name).open('a',encoding='utf-8') as log:
            completed=subprocess.run([sys.executable,*arguments],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            status('failed',failed_phase=name,exit_code=completed.returncode,log=str(OUT/log_name))
            raise SystemExit(completed.returncode)
    status('complete')

if __name__=='__main__':
    main()

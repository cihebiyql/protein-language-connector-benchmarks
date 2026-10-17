"""Finite feature preparation on the three already-owned physical GPUs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

SCOPE = Path(__file__).resolve().parents[1]
GPUS = (3, 4, 5)


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def resources():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    free = {int(line.split(',')[0]): int(line.split(',')[1]) for line in raw.splitlines()}
    mem = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    if mem['MemAvailable'] < 32*1024**3:
        raise RuntimeError('Actual host memory headroom below32GiB')
    return free


def run(args):
    if socket.gethostname() != 'node1':
        raise RuntimeError('Expected node1')
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name] = '2'
    processes = []
    try:
        free = resources()
        for index, gpu in enumerate(GPUS):
            if free[gpu] < 2560:
                raise RuntimeError('Insufficient actual extraction memory on selected GPU')
            command = ['timeout','--signal=TERM','--kill-after=15s','4h',sys.executable,
                '-m','src.mt_extract','extract','--data-root',str(SCOPE/'data'),
                '--cache-root',str(SCOPE/'cache'),'--esm-path',str(Path(args.models_root)/'facebook--esm2_t30_150M_UR50D'),
                '--gpu',str(gpu),'--shard-index',str(index),'--shards','3','--allocator-gib','2']
            log = SCOPE/'resources'/f'feature_shard{index}_v1.log'
            with log.open('xb') as stream:
                process = subprocess.Popen(command, cwd=SCOPE, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
        write_new(SCOPE/'resources/feature_workers_v1.json', {'utc':datetime.now(timezone.utc).isoformat(),
            'pids':[p.pid for p in processes], 'physical_gpus':list(GPUS), 'allocator_GiB_per_worker':2,
            'other_jobs_preserved':True,'source_sha256':hashlib.sha256((SCOPE/'src/mt_extract.py').read_bytes()).hexdigest()})
        while any(p.poll() is None for p in processes):
            if any(p.returncode not in (None,0) for p in processes):
                raise RuntimeError('Feature shard failed; preserving logs and partial payloads')
            time.sleep(2)
        if any(p.returncode for p in processes):
            raise RuntimeError('Nonzero feature extraction exit')
        subprocess.run([sys.executable,'-m','src.mt_extract','finalize','--data-root',str(SCOPE/'data'),
            '--cache-root',str(SCOPE/'cache')], cwd=SCOPE, env=env, check=True)
        started = time.monotonic()
        while True:
            free = resources()
            gpu = max(GPUS, key=lambda key:free[key])
            if free[gpu] >= 8704:
                break
            if time.monotonic()-started > 14400:
                raise RuntimeError('Descriptor extraction memory wait elapsed')
            time.sleep(20)
        subprocess.run(['timeout','--signal=TERM','--kill-after=15s','30m',sys.executable,
            '-m','src.mt_extract','descriptions','--data-root',str(SCOPE/'data'),
            '--cache-root',str(SCOPE/'cache'),'--qwen-path',str(Path(args.models_root)/'Qwen--Qwen2.5-1.5B-Instruct'),
            '--gpu',str(gpu),'--allocator-gib','8'], cwd=SCOPE, env=env, check=True)
        write_new(SCOPE/'resources/feature_stage_complete_v1.json', {
            'status':'complete','utc':datetime.now(timezone.utc).isoformat(),
            'shard_exit_codes':[p.returncode for p in processes], 'description_gpu':gpu,
            'data_manifest_sha256':hashlib.sha256((SCOPE/'data/manifest.json').read_bytes()).hexdigest(),
            'cache_manifest_sha256':hashlib.sha256((SCOPE/'cache/manifest.json').read_bytes()).hexdigest(),
            'description_vectors_sha256':hashlib.sha256((SCOPE/'cache/description_vectors.npz').read_bytes()).hexdigest()})
    except Exception as exc:
        write_new(SCOPE/'resources/feature_stage_failed_v1.json', {'status':'failed',
            'utc':datetime.now(timezone.utc).isoformat(),'error_type':type(exc).__name__,'error':str(exc)})
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
        for process in processes:
            if process.poll() is None:
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['start','work'])
    parser.add_argument('--models-root',required=True)
    args = parser.parse_args()
    if args.mode == 'work':
        run(args)
    else:
        binding = SCOPE/'resources/feature_launch_v1.json'
        if binding.exists():
            raise RuntimeError('Existing feature stage retained; refusing duplicate')
        with (SCOPE/'resources/feature_driver_v1.log').open('xb') as stream:
            p = subprocess.Popen([sys.executable,__file__,'work','--models-root',args.models_root],
                cwd=SCOPE, stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value={'status':'launched','pid':p.pid,'pgid':os.getpgid(p.pid),'utc':datetime.now(timezone.utc).isoformat()}
        write_new(binding,value)
        print(json.dumps(value))


if __name__ == '__main__':main()

"""Bound CPU-only cache preparation; retain logs, process-group and exit receipts."""
import argparse
import ctypes
from datetime import datetime
import os
from pathlib import Path
import runpy
import signal
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import execute_suite as supervision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        parent = int(os.environ['MATCHED_PREP_PARENT_PID'])
        if os.getppid() != parent:
            raise RuntimeError('CPU preparation controller absent')
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'CPU worker PDEATHSIG failed')
        if os.getppid() != parent:
            raise RuntimeError('CPU preparation controller exited during setup')
        sys.argv = [str(HERE / 'prepare_cache.py'), '--output', str(args.output / 'cache'), '--threads', '2']
        runpy.run_path(str(HERE / 'prepare_cache.py'), run_name='__main__')
        return
    if not sys.platform.startswith('linux'):
        raise RuntimeError('CPU preparation uses Linux process-group supervision')
    args.output.mkdir(parents=True, exist_ok=False)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'CPU controller subreaper failed')
    signal.signal(signal.SIGINT, supervision.request_stop)
    signal.signal(signal.SIGTERM, supervision.request_stop)
    deadline = datetime.fromisoformat('2026-10-16T23:20:00+08:00')
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--output', str(args.output)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', MATCHED_PREP_PARENT_PID=str(os.getpid()))
    registration = dict(purpose='CPU-only fixed-message preparation; no backbone execution or model training',
                        command=command, timeout_seconds=600, host_RSS_limit_bytes=12 * 1024**3,
                        source_sha256={str(p): supervision.sha(p) for p in sorted(HERE.glob('*.py'))},
                        effective_deadline_at=deadline.isoformat(), environment_overrides={
                            k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})
    supervision.write(args.output / 'preparation-registration.json', registration)
    receipt = supervision.supervise(command, args.output, 600, 12 * 1024**3, deadline, env)
    if receipt['exit_code'] != 0 or not receipt['process_group_absent'] or receipt['termination_reason'] is not None:
        raise RuntimeError('CPU preparation did not complete; retained terminal receipt')
    manifest = args.output / 'cache/manifest.json'
    supervision.write(args.output / 'preparation-complete.json', dict(
        manifest=str(manifest), manifest_sha256=supervision.sha(manifest), receipt=receipt))
    print(str(manifest), flush=True)


if __name__ == '__main__':
    main()

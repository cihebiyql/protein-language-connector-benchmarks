"""Apply the contracted PyTorch allocator cap before importing a GPU worker."""
from contextlib import contextmanager
import argparse
import math
import subprocess
import select
import threading
import time
import json
import os
from pathlib import Path
import runpy
import sys
from datetime import datetime, timezone


# This separate stdlib-only process does not need the CUDA worker's Python GIL.
# It can signal only its exact live parent after checking Linux process start time.
_GUARD_CODE = r"""
import json, os, select, signal, sys, time
from datetime import datetime, timezone
from pathlib import Path
pid, start_tick = int(sys.argv[1]), sys.argv[2]
wall_deadline, mono_deadline = float(sys.argv[3]), float(sys.argv[4])
receipt = Path(sys.argv[5])
def parent_matches():
    try:
        fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return os.getppid() == pid and fields[19] == start_tick
    except (OSError, ValueError, IndexError):
        return False
if not parent_matches():
    raise SystemExit(61)
print(json.dumps({'status':'ready','parent_pid':pid}),flush=True)
while parent_matches():
    remaining = min(wall_deadline - time.time(), mono_deadline - time.monotonic())
    if remaining <= 0:
        value = dict(status='deadline_expired', target_pid=pid, target_start_tick=start_tick,
                     ended_utc=datetime.now(timezone.utc).isoformat(),
                     mechanism='independent_stdlib_child_SIGKILL_exact_parent_only')
        try:
            with receipt.open('x', encoding='utf-8') as stream:
                json.dump(value, stream, indent=2)
        finally:
            if parent_matches():
                os.kill(pid, signal.SIGKILL)
        break
    ready, _, _ = select.select([sys.stdin], [], [], min(.25, remaining))
    if ready and not os.read(sys.stdin.fileno(), 1):
        break
"""


def guard_budget(environ, *, wall_now=None, monotonic_now=None):
    deadline = datetime.fromisoformat(environ['MM_WORKER_ABSOLUTE_DEADLINE'])
    mono = float(environ['MM_WORKER_MONOTONIC_DEADLINE'])
    if deadline.tzinfo is None or not math.isfinite(mono):
        raise ValueError('worker guard requires aware finite frozen deadlines')
    wall = time.time() if wall_now is None else wall_now
    now_mono = time.monotonic() if monotonic_now is None else monotonic_now
    if min(deadline.timestamp() - wall, mono - now_mono) <= 0:
        raise ValueError('worker guard budget already expired')
    return deadline.timestamp(), mono


@contextmanager
def worker_deadline_guard(receipt_path):
    if os.name != 'posix' or not Path('/proc/self/stat').exists():
        raise RuntimeError('GPU worker independent deadline guard requires Linux /proc')
    wall, mono = guard_budget(os.environ)
    pid = os.getpid()
    start_tick = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
    guard = subprocess.Popen([sys.executable, '-B', '-c', _GUARD_CODE, str(pid), start_tick,
                              str(wall), str(mono), str(receipt_path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    stopping = threading.Event()
    try:
        ready, _, _ = select.select([guard.stdout], [], [], 3)
        ack = json.loads(guard.stdout.readline()) if ready else {}
        if ack != {'status': 'ready', 'parent_pid': pid} or guard.poll() is not None:
            raise RuntimeError('independent GPU worker deadline guard failed startup handshake')
        def guard_liveness():
            guard.wait()
            if not stopping.is_set():
                os._exit(126)
        watcher = threading.Thread(target=guard_liveness, daemon=True)
        watcher.start()
        yield {'guard_pid': guard.pid, 'target_pid': pid, 'target_start_tick': start_tick,
               'absolute_deadline_epoch': wall, 'monotonic_deadline': mono,
               'mechanism': 'independent_stdlib_child_checks_exact_parent_start_tick',
               'signal_scope': 'exact_gpu_worker_parent_pid_only'}
    finally:
        stopping.set()
        guard.stdin.close()
        try:
            guard.wait(timeout=2)
        except subprocess.TimeoutExpired:
            guard.terminate()
            guard.wait(timeout=2)
        guard.stdout.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-memory-budget-bytes', required=True, type=int)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--module', choices=('src.mm_extract', 'src.mm_train'), required=True)
    args, rest = parser.parse_known_args(argv)
    if rest[:1] == ['--']:
        rest = rest[1:]
    if args.gpu_memory_budget_bytes <= 0 or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Explicit GPU assignment and positive allocator budget required')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    receipt_path = Path(args.receipt)
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    with worker_deadline_guard(receipt_path.with_name(receipt_path.name + '.deadline_guard.json')) as guard_binding:
        return execute_worker(args, rest, receipt_path, guard_binding)


def execute_worker(args, rest, receipt_path, guard_binding):
    import torch
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    total = torch.cuda.get_device_properties(0).total_memory
    if args.gpu_memory_budget_bytes > total:
        raise ValueError('Requested GPU allocator budget exceeds physical GPU memory')
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_budget_bytes / total, device=0)
    receipt = dict(started_at=datetime.now(timezone.utc).isoformat(),
                   visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                   allocator_budget_bytes=args.gpu_memory_budget_bytes,
                   physical_total_bytes=total, module=args.module,
                   determinism=True, tf32=False, cpu_threads=2,
                   independent_deadline_guard=guard_binding,
                   scope='PyTorch allocator; CUDA context/library overhead reported separately by supervisor resource probes')
    old_argv = sys.argv
    try:
        sys.argv = [args.module] + rest
        runpy.run_module(args.module, run_name='__main__', alter_sys=True)
    finally:
        sys.argv = old_argv
        receipt.update(ended_at=datetime.now(timezone.utc).isoformat(),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        with receipt_path.open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')


if __name__ == '__main__':
    main()

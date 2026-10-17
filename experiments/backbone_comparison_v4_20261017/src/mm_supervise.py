"""Bounded Linux process supervisor. No model imports and no shared configuration edits."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utcnow():
    return datetime.now(timezone.utc)


def write_new(path, obj):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def cpu_frame():
    parts = Path('/proc/stat').read_text().splitlines()[0].split()[1:]
    values = [int(x) for x in parts]
    # guest counters are already included in user/nice.
    return sum(values[:8]), values[3], values[4]


def probe(expected_host, gpu, rss_limit, root, sample_seconds=1.0):
    result = {'utc': utcnow().isoformat(), 'hostname': socket.gethostname(), 'gpu': gpu}
    reasons = []
    if result['hostname'] != expected_host:
        reasons.append('hostname_mismatch')
    a = cpu_frame()
    time.sleep(sample_seconds)
    b = cpu_frame()
    delta = max(b[0] - a[0], 1)
    result['cpu_idle_percent'] = 100 * (b[1] - a[1]) / delta
    result['cpu_iowait_percent'] = 100 * (b[2] - a[2]) / delta
    result['load_1m'], _, _ = os.getloadavg()
    result['cpu_count'] = os.cpu_count()
    mem = {line.split(':')[0]: int(line.split()[1]) * 1024
           for line in Path('/proc/meminfo').read_text().splitlines()}
    result['memory_available_bytes'] = mem['MemAvailable']
    if result['cpu_idle_percent'] < 8:
        reasons.append('cpu_saturated')
    if result['cpu_iowait_percent'] > 10:
        reasons.append('io_pressure')
    if mem['MemAvailable'] < rss_limit + 4 * 1024**3:
        reasons.append('insufficient_host_memory')
    disk = os.statvfs(root)
    result['storage_available_bytes'] = disk.f_bavail * disk.f_frsize
    if result['storage_available_bytes'] < 10 * 1024**3:
        reasons.append('insufficient_storage')
    if gpu is not None:
        cmd = ['nvidia-smi', f'--id={gpu}',
               '--query-gpu=uuid,memory.used,memory.free,utilization.gpu',
               '--format=csv,noheader,nounits']
        raw = subprocess.check_output(cmd, text=True, timeout=15).strip()
        uuid, used, free, util = [x.strip() for x in raw.split(',')]
        result['gpu_frame'] = dict(uuid=uuid, used_mib=int(used), free_mib=int(free), utilization=int(util))
        processes = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
             '--format=csv,noheader,nounits'], text=True, timeout=15)
        result['gpu_compute_processes'] = [x for x in processes.splitlines() if uuid in x]
        if result['gpu_compute_processes'] or int(util) > 5 or int(used) > 256:
            reasons.append('selected_gpu_busy')
    result.update(readiness='ready' if not reasons else 'busy', reasons=reasons)
    return result


def group_members(pgid):
    members = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        path = entry / 'stat'
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != 'Z':
                members.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            pass
    return members


def group_rss(pgid):
    total = 0
    for pid in group_members(pgid):
        try:
            for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    total += int(line.split()[1]) * 1024
        except OSError:
            pass
    return total


def terminate_group(pgid):
    # Caller supplies only its own Popen(start_new_session=True) PGID. A broken
    # procfs must not disable cleanup of that already owned process group.
    def maybe_alive():
        try:
            return bool(group_members(pgid))
        except OSError:
            return True  # unknown is never interpreted as proof of absence
    for sig, grace in ((signal.SIGTERM, 3), (signal.SIGKILL, 2)):
        if not maybe_alive():
            break
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        end = time.monotonic() + grace
        while maybe_alive() and time.monotonic() < end:
            time.sleep(0.1)


def standard_module(path):
    spec = importlib.util.spec_from_file_location('canonical_reproduction_validator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(contract_path, validator_path, output, gpu=None):
    if os.name != 'posix' or not Path('/proc').is_dir():
        raise RuntimeError('This supervisor requires Linux /proc and POSIX process groups')
    contract_path = Path(contract_path).resolve()
    raw = contract_path.read_bytes()
    contract = json.loads(raw)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    resources = contract['resources']
    deadline = datetime.fromisoformat(contract['timing']['effective_deadline_at'])
    timeout = contract['execution']['timeout_seconds']
    rss_limit = resources['memory_limit_bytes']
    validator = standard_module(validator_path)
    validation = validator.validate(contract, contract_bytes=raw, check_files=True,
                                    contract_directory=contract_path.parent)
    write_new(root / 'standard_start_validation.json', validation)
    if not validation.get('valid_for_execution'):
        raise RuntimeError('Canonical standard rejected contract; retained validation receipt')
    current_probe = probe(resources['expected_hostname'], gpu, rss_limit, contract['execution']['working_directory'])
    write_new(root / 'startup_resource_probe.json', current_probe)
    if current_probe['readiness'] != 'ready':
        write_new(root / 'receipt.json', {'status': 'not_started', 'reason': 'fresh_resource_gate', 'probe': current_probe})
        return 3
    # Validate again AFTER the resource probe; no hash/probe latency is hidden.
    validation = validator.validate(contract, contract_bytes=raw, check_files=False)
    write_new(root / 'boundary_validation.json', validation)
    if not validation.get('valid_for_execution') or timeout + 5 > (deadline - utcnow()).total_seconds():
        write_new(root / 'receipt.json', {'status': 'not_started', 'reason': 'remaining_window_at_start'})
        return 3
    command = contract['execution']['command']
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise ValueError('Command must be an argv string list, never a shell command')
    env = os.environ.copy()
    threads = str(resources['cpu_threads'])
    env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
               OPENBLAS_NUM_THREADS=threads, NUMEXPR_NUM_THREADS=threads,
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
               CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))
    started = utcnow()
    begin = time.monotonic()
    process = None
    reason = None
    monitor_exception = None
    peak_rss = 0
    stdout = Path(contract['execution']['stdout_path'])
    stderr = Path(contract['execution']['stderr_path'])
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stderr.parent.mkdir(parents=True, exist_ok=True)
    if not stdout.is_file() or not stderr.is_file() or stdout.stat().st_size or stderr.stat().st_size:
        raise ValueError('Admission must reserve fresh empty stdout/stderr; existing content is immutable')
    with stdout.open('r+') as out, stderr.open('r+') as err:
        try:
            env['MM_WORKER_ABSOLUTE_DEADLINE'] = deadline.isoformat()
            env['MM_WORKER_MONOTONIC_DEADLINE'] = str(begin + timeout)
            process = subprocess.Popen(command, cwd=contract['execution']['working_directory'], env=env,
                                       stdout=out, stderr=err, start_new_session=True)
            actual_started = utcnow()
            write_new(root / 'started.json', {'pid': process.pid, 'pgid': process.pid,
                      'started_at': actual_started.isoformat(), 'contract_sha256': hashlib.sha256(raw).hexdigest(),
                      'remaining_seconds_at_actual_start': (deadline - actual_started).total_seconds()})
            if timeout > (deadline - actual_started).total_seconds():
                reason = 'actual_start_window_mismatch'
            while reason is None and process.poll() is None:
                used = group_rss(process.pid)
                peak_rss = max(peak_rss, used)
                if used > rss_limit:
                    reason = 'host_rss_limit'
                elif time.monotonic() - begin >= timeout:
                    reason = 'timeout'
                elif utcnow() >= deadline:
                    reason = 'absolute_deadline'
                time.sleep(0.5)
        except BaseException as exc:
            reason = 'supervisor_exception'
            monitor_exception = {'type': type(exc).__name__, 'message': str(exc)}
        finally:
            if process is not None:
                terminate_group(process.pid)
                process.wait(timeout=10)
    if process is None:
        write_new(root / 'receipt.json', dict(status='failed', exit_code=None,
                  termination_reason=reason or 'launch_failed', monitor_exception=monitor_exception,
                  pid=None, pgid=None, remaining_group_members=[], ended_at=utcnow().isoformat()))
        return 2
    try:
        remaining = group_members(process.pid)
    except OSError as exc:
        remaining = None
        reason = 'cleanup_verification_unavailable'
        monitor_exception = {'type': type(exc).__name__, 'message': str(exc)}
    receipt = dict(status='succeeded' if process.returncode == 0 and reason is None and remaining == [] else 'failed',
                   exit_code=process.returncode, termination_reason=reason, monitor_exception=monitor_exception, pid=process.pid, pgid=process.pid,
                   started_at=started.isoformat(), ended_at=utcnow().isoformat(),
                   elapsed_seconds=time.monotonic() - begin, peak_group_rss_bytes=peak_rss,
                   remaining_group_members=remaining, handles_absent_verified_at=utcnow().isoformat(),
                   contract_path=str(contract_path), contract_sha256=hashlib.sha256(raw).hexdigest(),
                   stdout_sha256=sha(stdout), stderr_sha256=sha(stderr))
    write_new(root / 'receipt.json', receipt)
    return 0 if receipt['status'] == 'succeeded' else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', required=True)
    p.add_argument('--validator', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--gpu', type=int)
    args = p.parse_args()
    return run(args.contract, args.validator, args.output, args.gpu)


if __name__ == '__main__':
    raise SystemExit(main())

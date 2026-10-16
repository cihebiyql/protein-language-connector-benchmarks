"""Supplement the exact owned controller with UTC wall-time bounds, never scores."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def command(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[:-1]
    except FileNotFoundError:
        return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat('2026-10-16T23:20:00+08:00')
    expected = str(args.suite.resolve()).encode()

    def matches(parts):
        return (any(p.endswith(b'/execute_remaining_v1.py') for p in parts)
                and b'--worker' not in parts and b'--output' in parts
                and parts[parts.index(b'--output') + 1] == expected)

    targets = [int(p.name) for p in Path('/proc').glob('[0-9]*') if matches(command(int(p.name)))]
    if len(targets) != 1:
        raise RuntimeError('Exact owned original V1 controller identity not unique')
    pid = targets[0]
    record = {'started_utc': datetime.now(timezone.utc).isoformat(), 'controller_pid': pid,
              'command': [p.decode() for p in command(pid)], 'effective_deadline_at': deadline.isoformat(),
              'scope': 'UTC upper bounds only; no model, test score, parser, seed or source changes',
              'signal_sent': False, 'termination_reason': None}
    try:
        while matches(command(pid)):
            stamp = datetime.now(timezone.utc)
            reason = 'absolute_original_scope_deadline' if stamp >= deadline else None
            for job in args.suite.iterdir():
                running = job / 'running-process.json'
                if not job.is_dir() or not running.exists() or (job / 'execution-receipt.json').exists():
                    continue
                process = read(running)
                contract = read(job / 'task-contract.json')
                elapsed = (stamp - datetime.fromisoformat(process['started_utc'])).total_seconds()
                timeout = contract['execution']['timeout_seconds']
                if elapsed >= timeout:
                    reason = 'individual_UTC_wall_timeout'
                    record.update(job=str(job), worker_pid=process['pid'], worker_pgid=process['pgid'],
                                  measured_UTC_interval_seconds=elapsed, contracted_timeout_seconds=timeout)
                    break
            if reason:
                if not matches(command(pid)):
                    break
                record.update(signal_sent=True, termination_reason=reason,
                              signal_utc=stamp.isoformat(), signal='SIGTERM_to_exact_owned_controller')
                os.kill(pid, signal.SIGTERM)
                break
            time.sleep(.5)
    finally:
        record['ended_utc'] = datetime.now(timezone.utc).isoformat()
        with args.receipt.open('x', encoding='utf-8') as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

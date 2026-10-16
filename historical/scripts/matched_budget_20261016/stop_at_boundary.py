"""Request graceful handoff of the exact owned V1 controller after full trials."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def command(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[:-1]
    except FileNotFoundError:
        return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--completed-trials', type=int, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    expected = str(args.suite.resolve()).encode()

    def matches(parts):
        return (any(p.endswith(b'/execute_suite.py') for p in parts) and b'--worker' not in parts
                and b'--output' in parts and parts[parts.index(b'--output') + 1] == expected)

    targets = [int(p.name) for p in Path('/proc').glob('[0-9]*') if matches(command(int(p.name)))]
    if len(targets) != 1:
        raise RuntimeError('Exact owned V1 controller identity not unique: ' + str(targets))
    pid = targets[0]
    started = time.monotonic()
    while time.monotonic() - started < 1800:
        if not matches(command(pid)):
            raise RuntimeError('V1 controller exited before requested boundary')
        state_path = args.suite / 'suite_state.json'
        state = load(state_path) if state_path.exists() else {}
        if state.get('status') == 'failed_or_partial':
            raise RuntimeError('V1 failed before requested handoff; no signal sent')
        completed = [j for j in state.get('jobs', []) if j['case']['phase'] == 'trial' and j['complete']]
        if len(completed) >= args.completed_trials:
            record = {'requested_utc': datetime.now(timezone.utc).isoformat(), 'pid': pid,
                      'command': [p.decode() for p in command(pid)], 'signal': 'SIGTERM',
                      'completed_trials': len(completed), 'reason': 'Uniform inference throughput V2 handoff; no score-based selection',
                      'old_results_preserved': True}
            with args.receipt.open('x', encoding='utf-8') as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2)
            os.kill(pid, signal.SIGTERM)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            return
        time.sleep(.5)
    raise TimeoutError('V1 boundary wait exceeded 1800 seconds; no signal sent')


if __name__ == '__main__':
    main()

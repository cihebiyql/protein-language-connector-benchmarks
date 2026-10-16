"""Read-only progress view; no scoring, selection, retries or process changes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    args = parser.parse_args()
    terminal, active = [], []
    for job in sorted(args.suite.iterdir()):
        if not job.is_dir() or not (job / 'case.json').exists():
            continue
        case = load(job / 'case.json')
        if (job / 'execution-receipt.json').exists():
            receipt = load(job / 'execution-receipt.json')
            terminal.append({'job': job.name, 'phase': case['phase'],
                             'exit_code': receipt['exit_code'],
                             'termination_reason': receipt['termination_reason'],
                             'ended_utc': receipt['ended_utc'],
                             'elapsed_seconds': receipt['elapsed_seconds']})
            continue
        item = {'job': job.name, 'phase': case['phase'], 'stage': 'admission'}
        if (job / 'running-process.json').exists():
            running = load(job / 'running-process.json')
            item.update(pid=running['pid'], stage='worker', elapsed_seconds=(
                datetime.now(timezone.utc) - datetime.fromisoformat(running['started_utc'])).total_seconds())
        log = job / 'stdout.log'
        lines = log.read_text(encoding='utf-8').splitlines() if log.exists() else []
        if lines:
            try:
                event = json.loads(lines[-1])
                item['last_event'] = {k: event[k] for k in ('utc', 'event', 'epoch', 'step', 'optimizer_steps') if k in event}
            except ValueError:
                item['last_event'] = 'partial stdout line'
        generated = job / 'artifacts/test_predictions_correct.jsonl'
        if generated.exists():
            with generated.open(encoding='utf-8') as handle:
                item['test_answers_recorded'] = sum(1 for line in handle if line.endswith('\n'))
        active.append(item)
    state_path = args.suite / 'suite_state.json'
    state = load(state_path) if state_path.exists() else {}
    print(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'status': state.get('status', 'starting'),
        'terminal_canaries': sum(x['phase'] == 'canary' for x in terminal),
        'terminal_trials': sum(x['phase'] == 'trial' for x in terminal),
        'last_terminal': max(terminal, key=lambda x: x['ended_utc']) if terminal else None, 'active': active,
        'exception': state.get('exception')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

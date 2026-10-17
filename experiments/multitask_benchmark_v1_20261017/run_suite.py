"""Run the non-pKD CPU foundation in a fresh namespace or verify saved artifacts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

SCOPE = Path(__file__).resolve().parent
sys.path.insert(0, str(SCOPE / 'src'))
import meltome
import replay_existing


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require_hash(path, expected):
    if sha(path) != expected:
        raise ValueError('Artifact changed: ' + str(path))


def verify(project):
    historical = SCOPE / 'results/existing_replay_v1'
    manifest = json.loads((historical / 'manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        require_hash(project / item['path'], item['sha256'])
    run = SCOPE / 'results/meltome_human_v1/cpu_baseline_v1'
    receipt = json.loads((run / 'receipt.json').read_text(encoding='utf-8'))
    if receipt['status'] != 'complete':
        raise ValueError('Meltome baseline is incomplete')
    for name, item in receipt['files'].items():
        require_hash(run / name, item['sha256'])
    config = json.loads((run / 'frozen_config.json').read_text(encoding='utf-8'))
    for name, value in config['files_sha256'].items():
        require_hash(SCOPE / name, value)
    return {'status': 'verified', 'historical_file_pins': len(manifest['files']),
            'meltome_run_file_pins': len(receipt['files']), 'meltome_data_pins': len(config['files_sha256']),
            'models_or_GPU_executed': False}


def run_fresh(project, run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', run_id):
        raise ValueError('Use a new simple run ID of at most64 characters')
    output = SCOPE / 'results/suites' / run_id
    meltome_output = SCOPE / 'results/meltome_human_v1' / run_id
    if output.exists() or meltome_output.exists():
        raise FileExistsError('Run ID already exists; all prior outputs are preserved')
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'started_utc': datetime.now(timezone.utc).isoformat(), 'run_id': run_id,
               'scope': 'CPU infrastructure and historical prediction replay',
               'V4_connector_multitask_training': False, 'GPU_used': False,
               'runner_sha256': sha(__file__)}
    try:
        result = replay_existing.run(project, output / 'historical_replay')
        statuses = {name: panel['status'] for name, panel in result['panels'].items()}
        receipt['historical_panels'] = statuses
        if any(value != 'complete' for value in statuses.values()):
            raise ValueError('Historical panel verification incomplete; inspect retained outputs')
        fresh_meltome = meltome.run(SCOPE / 'configs/meltome_human_v1.json', run_id)
        metrics = json.loads((fresh_meltome / 'metrics.json').read_text(encoding='utf-8'))
        receipt.update(status='complete', meltome_results=str(fresh_meltome.relative_to(SCOPE)),
                       meltome_test=metrics['test'])
    except Exception as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt['ended_utc'] = datetime.now(timezone.utc).isoformat()
        with (output / 'suite_receipt.json').open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=SCOPE.parents[1])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--verify', action='store_true')
    group.add_argument('--run-id')
    args = parser.parse_args()
    result = verify(args.project_root.resolve()) if args.verify else run_fresh(args.project_root.resolve(), args.run_id)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

"""Snapshot and verify historical assets without modifying their contents."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROOTS = ('results/bench_v02', 'results/foundation_20261016',
         'results/real_run_20261015', 'results/real_t3_20261015',
         'data/processed/bench_v02/a1_locked_v2', 'scripts/foundation_20261016')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def inventory():
    files = {}
    for name in ROOTS:
        for path in sorted((ROOT / name).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                files[path.relative_to(ROOT).as_posix()] = {
                    'bytes': path.stat().st_size, 'sha256': digest(path)}
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--verify-output', type=Path)
    args = parser.parse_args()
    current = inventory()
    if args.verify_output:
        previous = json.loads(args.snapshot.read_text(encoding='utf-8'))['files']
        changes = [name for name in sorted(set(previous) | set(current))
                   if previous.get(name) != current.get(name)]
        record = {'verified_utc': datetime.now(timezone.utc).isoformat(),
                  'snapshot_sha256': digest(args.snapshot), 'unchanged': not changes,
                  'changed_missing_or_added_paths': changes, 'files_checked': len(current)}
        with args.verify_output.open('x', encoding='utf-8') as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
        if changes:
            raise SystemExit('Historical asset changes detected')
    else:
        record = {'created_utc': datetime.now(timezone.utc).isoformat(),
                  'purpose': 'Preserve existing scores, frozen A1 inputs and historical scientific implementation',
                  'roots': ROOTS, 'excluded': ['__pycache__'], 'files': current,
                  'file_count': len(current), 'bytes': sum(v['bytes'] for v in current.values())}
        with args.snapshot.open('x', encoding='utf-8') as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in record.items() if k != 'files'}, ensure_ascii=False))


if __name__ == '__main__':
    main()

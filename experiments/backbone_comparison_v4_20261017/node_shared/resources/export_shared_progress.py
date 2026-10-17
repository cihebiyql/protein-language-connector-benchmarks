"""Read-only snapshot of frozen controls and current results; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True)
a = p.parse_args()
if not a.tag.replace('_', '').isalnum():
    raise ValueError('Invalid export tag')
scope = Path(__file__).resolve().parents[1]
analysis = scope / 'analysis' / ('snapshot_' + a.tag)
result = subprocess.run([sys.executable, str(scope / 'resources/analyze_v4_shared.py'),
    '--scope-root', str(scope), '--queue-root', str(scope / 'runs/formal_shared_v1'),
    '--output', str(analysis)], cwd=scope)
if result.returncode:
    raise RuntimeError('Independent snapshot analysis failed')
output = scope / 'resources' / ('export_' + a.tag + '.tar.gz')
roots = [scope/'configs', scope/'docs', analysis]
names = ['formal_launch_shared_v1.json', 'formal_driver_shared_v1.json',
    'shared_runtime_targeted_cpu.json', 'shared_helpers_cpu_verification.json']
files = {path for root in roots for path in root.rglob('*') if path.is_file()}
for name in names:
    path = scope/'resources'/name
    if path.is_file(): files.add(path)
for backbone in ('qwen25_1p5b', 'qwen3_4b_2507'):
    job = scope/'runs'/('canary_'+backbone+'_shared_v1')
    for subdir in ('artifacts', 'archive', 'supervision', 'archive_gpu_evidence'):
        for path in (job/subdir).rglob('*'):
            if path.is_file() and path.suffix not in ('.pt', '.safetensors'):
                files.add(path)
    for name in ('gpu_limit.json', 'task-contract.json'):
        if (job/name).is_file(): files.add(job/name)
queue = scope/'runs/formal_shared_v1'
for name in ('queue_events.jsonl', 'plan.json'):
    if (queue/name).is_file(): files.add(queue/name)
# Include completed outputs for later snapshots, never live tensor files.
for case in (queue/'cases').glob('*'):
    for attempt in case.glob('attempt_*'):
        for path in attempt.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.csv', '.log', '.txt'):
                files.add(path)
with tarfile.open(output, 'x:gz') as archive:
    for path in sorted(files): archive.add(path, arcname=str(path.relative_to(scope)))
print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(), 'path':str(output),
    'files':len(files), 'bytes':output.stat().st_size,
    'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}), flush=True)

"""Start exactly one owned detached queue driver; preserve its log and identity."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

scope = Path(__file__).resolve().parents[1]
binding = scope / 'resources/formal_launch_shared_v1.json'
if binding.exists():
    raise RuntimeError('Existing launch identity preserved; refusing a duplicate queue')
plan = scope / 'configs/queue_shared_v1.json'
receipt = json.loads((scope / 'docs/freeze_shared_v1.json').read_text())
if receipt['outputs']['queue']['sha256'] != hashlib.sha256(plan.read_bytes()).hexdigest():
    raise RuntimeError('Frozen queue digest differs')
deadline = datetime.fromisoformat(receipt['deadline'])
if datetime.now(timezone.utc) >= deadline:
    raise RuntimeError('Absolute deadline reached')
log = scope / 'resources/formal_driver_shared_v1.log'
with log.open('xb') as stream:
    child = subprocess.Popen([sys.executable, str(scope / 'resources/run_formal_and_analyze_shared_v1.py')],
        cwd=scope, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
        start_new_session=True)
record = {'status': 'launched', 'pid': child.pid, 'pgid': os.getpgid(child.pid),
    'utc': datetime.now(timezone.utc).isoformat(), 'queue_sha256': hashlib.sha256(plan.read_bytes()).hexdigest(),
    'selected_gpu_ids': [4,5,3], 'maximum_concurrent_gpu_workers': 3,
    'log': str(log), 'driver_sha256': hashlib.sha256((scope / 'resources/run_formal_and_analyze_shared_v1.py').read_bytes()).hexdigest()}
with binding.open('x') as stream:
    json.dump(record, stream, indent=2)
print(json.dumps(record), flush=True)

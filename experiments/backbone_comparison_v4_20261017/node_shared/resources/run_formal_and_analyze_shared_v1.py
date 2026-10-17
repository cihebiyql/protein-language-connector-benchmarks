"""Own one frozen queue and generate a terminal analysis without an interactive SSH session."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

scope = Path(__file__).resolve().parents[1]
receipt = json.loads((scope / 'docs/freeze_shared_v1.json').read_text())
env = dict(os.environ)
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    env[name] = '2'
env['TOKENIZERS_PARALLELISM'] = 'false'
status_path = scope / 'resources/formal_driver_shared_v1.json'

def save(value):
    value['utc'] = datetime.now(timezone.utc).isoformat()
    temporary = status_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(status_path)

save({'status': 'running', 'pid': os.getpid(), 'launch_command': receipt['launch_command']})
queue = subprocess.run(receipt['launch_command'], cwd=scope, env=env)
analysis = subprocess.run([sys.executable, str(scope / 'resources/analyze_v4_shared.py'),
    '--scope-root', str(scope), '--queue-root', str(scope / 'runs/formal_shared_v1'),
    '--output', str(scope / 'analysis/formal_shared_v1_terminal')], cwd=scope, env=env)
save({'status': 'terminal', 'pid': os.getpid(), 'queue_exit_code': queue.returncode,
    'analysis_exit_code': analysis.returncode})
sys.exit(queue.returncode if queue.returncode else analysis.returncode)

"""Start exactly one detached, scope-owned node-local publication callback."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import node1_publish as publisher
from finish_and_export import identity_alive


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'pgid': int(fields[2]), 'sid': int(fields[3]), 'start_ticks': int(fields[19])}


def launch(config_path):
    config, path = publisher.load_config(config_path)
    scope = Path(config['scope_root']).resolve()
    root = scope / 'private/node1_publication'
    with publisher.exclusive(root / 'launcher.lock'):
        for receipt in root.glob('*/launch.json'):
            previous = publisher.publication.read(receipt)
            if identity_alive(previous['identity']):
                raise RuntimeError('An exact live node-local callback already owns this scope')
        directory = root / config['run_id']; directory.mkdir(parents=True, exist_ok=True)
        if (directory / 'PUBLICATION_RECEIPT.json').exists():
            return {'status': 'already_completed', 'receipt': str(directory / 'PUBLICATION_RECEIPT.json')}
        if (directory / 'launch.json').exists():
            raise FileExistsError('Prior dead launch preserved; run publisher directly to inspect/recover, never duplicate automatically')
        command = [config['python'], str(Path(__file__).with_name('node1_publish.py')), '--config', str(path)]
        env = {**os.environ, **config['environment_overrides'], 'GIT_SSH_COMMAND': config['git_ssh_command'],
               'GIT_TERMINAL_PROMPT': '0', 'CUDA_VISIBLE_DEVICES': '',
               'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
               'NUMEXPR_NUM_THREADS': '2'}
        with (directory / 'stdout.log').open('xb') as stdout, (directory / 'stderr.log').open('xb') as stderr:
            process = subprocess.Popen(command, cwd=scope, env=env, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True)
        child = identity(process.pid)
        if child['pid'] != child['pgid'] or child['pid'] != child['sid']:
            raise RuntimeError('Detached process session identity is not proven')
        result = {'status': 'launched', 'identity': child, 'at_unix': time.time(),
            'run_id': config['run_id'], 'config_sha256': publisher.publication.sha(path),
            'stdin': '/dev/null', 'start_new_session': True, 'node_local_only': True,
            'application_or_system_scheduled_monitor': False}
        publisher.publication.write(directory / 'launch.json', result)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', required=True)
    args = p.parse_args(); print(json.dumps(launch(args.config)), flush=True)


if __name__ == '__main__': main()

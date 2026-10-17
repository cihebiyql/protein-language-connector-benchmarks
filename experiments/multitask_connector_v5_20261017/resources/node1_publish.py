"""Run one completion/export/publication chain wholly on the execution node."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finish_and_export as finish
import sync_results as publication


def load_config(path):
    path = Path(path).resolve(strict=True)
    config = publication.read(path)
    required = ('scope_root', 'queue_root', 'python', 'repo', 'run_id', 'git_ssh_command',
                'git_author_name', 'git_author_email')
    if any(not isinstance(config.get(key), str) or not config[key] for key in required):
        raise ValueError('Explicit scope/queue/python/repo/run/auth/author configuration required')
    scope = Path(config['scope_root']).resolve(strict=True)
    private = scope / 'private'
    if scope.name != publication.SCOPE_ID or not path.is_relative_to(private):
        raise ValueError('Runtime configuration must remain inside this scope private directory')
    for key in ('queue_root', 'python', 'repo'):
        if not Path(config[key]).is_absolute(): raise ValueError('Absolute configured runtime paths required')
    if not Path(config['queue_root']).resolve(strict=True).is_relative_to(scope / 'runs'):
        raise ValueError('Queue escapes the new scope runs directory')
    if not Path(config['repo']).resolve(strict=True).is_relative_to(private):
        raise ValueError('Use a dedicated clone inside this scope private directory')
    publication.safe_id(config['run_id'])
    tokens = shlex.split(config['git_ssh_command'])
    if not tokens or Path(tokens[0]).name != 'ssh' or re.search(r'[\n\r\x00;|&`$]', config['git_ssh_command']):
        raise ValueError('A single explicit deploy-key SSH command is required')
    overrides = config.get('environment_overrides', config.get('environmentOverrides', {}))
    if set(overrides) - {'PATH', 'GIT_EXEC_PATH', 'GIT_TEMPLATE_DIR'}:
        raise ValueError('Only isolated Git runtime environment overrides are supported')
    if any(not isinstance(v, str) or '\x00' in v for v in overrides.values()):
        raise ValueError('Invalid isolated runtime override')
    config['environment_overrides'] = overrides
    return config, path


@contextmanager
def exclusive(path):
    import fcntl
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError('Another publisher owns this scope') from exc
        yield


@contextmanager
def environment(config):
    values = {**config['environment_overrides'], 'GIT_SSH_COMMAND': config['git_ssh_command'],
              'GIT_TERMINAL_PROMPT': '0', 'GIT_AUTHOR_NAME': config['git_author_name'],
              'GIT_AUTHOR_EMAIL': config['git_author_email'], 'GIT_COMMITTER_NAME': config['git_author_name'],
              'GIT_COMMITTER_EMAIL': config['git_author_email'], 'CUDA_VISIBLE_DEVICES': '',
              'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
              'NUMEXPR_NUM_THREADS': '2'}
    before = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in before.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value


def verified_publication(worktree, public, run_id):
    """Recheck an immutable own commit even when main advanced after publication."""
    commit = publication.call(['git', 'rev-parse', 'HEAD'], worktree).stdout.strip()
    remote = publication.call(['git', 'ls-remote', 'origin', 'refs/heads/main'], worktree).stdout.split()[0]
    publication.call(['git', 'fetch', 'origin', 'main'], worktree)
    publication.call(['git', 'merge-base', '--is-ancestor', commit, 'origin/main'], worktree)
    url = publication.call(['git', 'remote', 'get-url', 'origin'], worktree).stdout.strip()
    match = re.fullmatch(r'(?:git@github.com:|https://github.com/)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?', url)
    if not match: raise ValueError('Public GitHub origin without embedded credentials required')
    verified = []
    for name in ('PUBLICATION_MANIFEST.json', 'STATUS.md'):
        file = public / name; relative = file.relative_to(worktree).as_posix()
        request = urllib.request.Request('https://raw.githubusercontent.com/' + match[1] + '/' + commit + '/'
            + urllib.parse.quote(relative), headers={'User-Agent': 'V5-node-completion-verifier'})
        with urllib.request.urlopen(request, timeout=60) as response: data = response.read()
        if hashlib.sha256(data).hexdigest() != publication.sha(file): raise ValueError('Anonymous public bytes differ')
        verified.append({'relative': relative, 'sha256': publication.sha(file)})
    return {'commit': commit, 'remote_head': remote, 'anonymous_verified': verified, 'run_id': run_id}


def publish_with_recovery(worktree, targets, run_id, *, sleep=time.sleep):
    """Retry only publication of this same commit; append remote changes without force."""
    initial = publication.call(['git', 'rev-parse', 'HEAD'], worktree).stdout.strip()
    try:
        return publication.commit_and_verify(worktree, targets, run_id)
    except (subprocess.CalledProcessError, OSError, RuntimeError) as error:
        current = publication.call(['git', 'rev-parse', 'HEAD'], worktree).stdout.strip()
        if current == initial: raise  # No owned publication commit exists; never hide commit errors.
        last = error
    merge_message = ('Preserve concurrent main work while appending the completed V5 evidence\n\n'
        'Merge remote progress only in the dedicated publication worktree; retain the fixed result namespace.\n\n'
        'Constraint: No force push or primary checkout changes\nConfidence: high\nScope-risk: narrow\n'
        'Co-authored-by: OmX <omx@oh-my-codex.dev>')
    for attempt in range(2):
        sleep(10 * (attempt + 1))
        try:
            publication.call(['git', 'fetch', 'origin', 'main'], worktree)
            try:
                publication.call(['git', 'merge-base', '--is-ancestor', 'HEAD', 'origin/main'], worktree)
                return verified_publication(worktree, targets[0], run_id)
            except subprocess.CalledProcessError:
                pass
            publication.call(['git', 'merge', '--no-edit', '-m', merge_message, 'origin/main'], worktree)
            publication.call(['git', 'push', 'origin', 'HEAD:main'], worktree)
            return verified_publication(worktree, targets[0], run_id)
        except (subprocess.CalledProcessError, OSError, RuntimeError) as exc:
            last = exc
            # A conflict remains for human review; never resolve or reset it automatically.
            if (Path(worktree) / '.git').is_file():
                merge_head = publication.call(['git', 'rev-parse', '--git-path', 'MERGE_HEAD'], worktree).stdout.strip()
                if (Path(worktree) / merge_head).exists(): raise
    raise last


def run(config_path):
    config, config_path = load_config(config_path)
    scope, queue = Path(config['scope_root']).resolve(), Path(config['queue_root']).resolve()
    state = scope / 'private/node1_publication' / config['run_id']
    state.mkdir(parents=True, exist_ok=True)
    binding = {'config_sha256': publication.sha(config_path), 'publisher_source_sha256': publication.sha(__file__),
               'run_id': config['run_id']}
    with exclusive(scope / 'private/node1_publication/publisher.lock'), environment(config):
        bound = state / 'binding.json'
        if bound.exists() and publication.read(bound) != binding: raise ValueError('Prior publication binding differs; preserved')
        publication.write(bound, binding)
        final = state / 'PUBLICATION_RECEIPT.json'
        if final.exists(): return {**publication.read(final), 'reused_completed_receipt': True}
        export = finish.finish(scope, queue, scope / 'resources/completion_exports' / config['run_id'], wait=True)
        publication.write(state / 'export_receipt.json', export)
        if publication.sha(export['archive']) != export['sha256']: raise ValueError('Local node export SHA differs')
        extracted = state / 'private_export'
        if not extracted.exists(): publication.safe_extract(export['archive'], extracted)
        manifest = publication.verify_export(extracted, export['manifest'], export['manifest_sha256'])
        repo = Path(config['repo']).resolve()
        # Dedicated clone-local author settings; never git config --global.
        publication.call(['git', 'config', '--local', 'user.name', config['git_author_name']], repo)
        publication.call(['git', 'config', '--local', 'user.email', config['git_author_email']], repo)
        worktree = publication.isolated_worktree(repo, scope / 'private/publication_worktrees', config['run_id'])
        publication.write(state / 'worktree.json', {'path': str(worktree), 'source_export_sha256': export['sha256']})
        base = worktree / 'experiments' / publication.SCOPE_ID
        public = publication.publish_snapshot(extracted, base / 'results' / config['run_id'], manifest)
        analysis = base / 'analysis' / config['run_id']
        original = public / manifest['independent_analysis']
        if original.is_dir(): shutil.copytree(original, analysis)
        meta = publication.read(public / 'PUBLICATION_MANIFEST.json')
        status = base / 'COMPLETION_STATUS.json'
        publication.write(status, {'status': 'terminal' if manifest['completion_trigger'] == 'queue_terminal' else 'partial_controller_dead',
            'run_id': config['run_id'], 'expected_cases': 45, 'analysis_status_counts': meta['analysis_status_counts'],
            'queue_status_counts': meta['queue_status_counts'], 'completion_trigger': manifest['completion_trigger'],
            'results': 'results/' + config['run_id'], 'analysis': 'analysis/' + config['run_id'] if analysis.is_dir() else None,
            'all_positions_preserved': True, 'three_complete_seeds_required_for_mean_SD': True,
            'publication_host': 'node1', 'local_computer_callback_required': False})
        targets = [public] + ([analysis] if analysis.is_dir() else []) + [status]
        result = publish_with_recovery(worktree, targets, config['run_id'])
        result.update(source_export_sha256=export['sha256'], expected_cases=45,
                      completion_trigger=manifest['completion_trigger'], executed_wholly_on_node=True)
        publication.write(final, result)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', required=True)
    args = p.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False), flush=True)


if __name__ == '__main__': main()

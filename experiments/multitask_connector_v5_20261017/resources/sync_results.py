"""Single completion-linked node1 retrieval and isolated public GitHub publication."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request

SCOPE_ID = 'multitask_connector_v5_20261017'
SECRET = re.compile(r'(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)')
PRIVATE = re.compile(r'(?:[A-Za-z]:[\\/](?:work|Users)[\\/]|(?<![\w>])/(?:data1|data|home|root|mnt|Users)/|GPU-[0-9a-f-]{8,})', re.I)
DROP = {'pid', 'pgid', 'sid', 'start_ticks', 'gpu_uuid', 'gpu_compute_processes', 'gpu_processes',
        'all_gpu_compute_process_rows_observed', 'last_resource_observation', 'resource_observation',
        'ip_address', 'remote_address', 'local_address'}
DIAGNOSTICS = {'error', 'traceback', 'exception', 'stderr', 'stdout', 'reason', 'queue_error', 'exclusion_reason'}


def publication_restriction(relative):
    """Uniform inherited corpus policy, independent of arm, score and seed."""
    parts = PurePosixPath(relative).parts
    name = parts[-1]
    if len(parts) > 1 and parts[0] == 'data' and parts[1] == 'elm':
        return 'ELM/B1 raw inputs and candidate text remain private under existing corpus publication scope'
    if name in ('description_vectors.npz', 'description_vectors.json'):
        return 'Frozen candidate-description inputs/embeddings remain private under ELM/B1 publication scope'
    if 'qualification' in parts:
        return 'TRAIN qualification artifact directory private; aggregate parent acceptance receipt remains available'
    elm = any(part == 'elm' or part.startswith('elm__') for part in parts)
    if elm and (name == 'run_manifest.json' or name.endswith('.pt') or 'prediction' in name.lower()):
        return 'ELM/B1 per-query outputs, per-sequence coverage manifests and trained weights remain private; aggregates retained'
    return None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON metadata: ' + x)))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def safe_id(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}', value): raise ValueError('Unsafe run-id')
    return value


def safe_extract(archive, destination):
    """Reject links, traversal, Windows ADS, duplicates and expansion before writing."""
    destination = Path(destination).resolve()
    if destination.exists(): raise FileExistsError('Extraction directory must be new')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = set()
        if len(members) > 200000 or sum(m.size for m in members) > 30 * 1024**3:
            raise ValueError('Export exceeds explicit30GiB/200000-member bound')
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or '\\' in member.name or ':' in member.name
                    or not (member.isfile() or member.isdir()) or path.as_posix() in names):
                raise ValueError('Unsafe archive member: ' + member.name)
            names.add(path.as_posix())
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            path = destination.joinpath(*PurePosixPath(member.name).parts)
            if not path.resolve().is_relative_to(destination): raise ValueError('Archive escapes destination')
            if member.isdir(): path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, path.open('xb') as dst: shutil.copyfileobj(src, dst)
    return destination


def verify_export(root, manifest_relative, expected_sha):
    root = Path(root).resolve()
    path = (root / manifest_relative).resolve(strict=True)
    if not path.is_relative_to(root) or sha(path) != expected_sha: raise ValueError('Export manifest binding mismatch')
    manifest = read(path)
    expected = {item['path'] for item in manifest['files']} | {manifest_relative}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != expected: raise ValueError('Unmanifested or missing export files')
    for item in manifest['files']:
        file = (root / item['path']).resolve(strict=True)
        if not file.is_relative_to(root) or file.is_symlink() or sha(file) != item['sha256'] or file.stat().st_size != item['bytes']:
            raise ValueError('Export file integrity mismatch: ' + item['path'])
    return manifest


def normalize(value, roots):
    def string(text):
        if SECRET.search(text): raise ValueError('Credential-like payload blocks publication')
        for root, label in sorted(roots.items(), key=lambda x: -len(x[0])): text = text.replace(root, label)
        # Unknown private paths fail closed rather than silently rewriting evidence.
        if PRIVATE.search(text): raise ValueError('Unregistered private metadata path or device identity')
        return text
    if isinstance(value, str): return string(value)
    if isinstance(value, list): return [normalize(v, roots) for v in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in DROP: continue
            if key in ('hostname', 'expected_hostname', 'expected_host') or (key == 'host' and item != 'node1'):
                result[key] = '<COMPUTE_HOST>'; continue
            if key == 'remaining_group_members' and item:
                result['remaining_group_member_count'] = len(item); continue
            new = string(key)
            if new in result: raise ValueError('Normalized metadata key collision')
            try: result[new] = normalize(item, roots)
            except ValueError:
                if key not in DIAGNOSTICS or not isinstance(item, str): raise
                # Never rewrite a biological field, prediction or number to
                # hide a private diagnostic. The original file remains private.
                raw = json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False).encode('utf-8')
                result[new] = {'private_diagnostic_sha256': hashlib.sha256(raw).hexdigest(),
                    'publication_note': 'Private diagnostic retained in original export; source file SHA recorded'}
        return result
    return value


def root_map(scope, manifest):
    roots = {manifest['scope_root']: '<REMOTE_SCOPE>'}
    for key, value in manifest.get('runtime_roots', {}).items():
        if not isinstance(value, str) or not (value.startswith('/') or re.match(r'^[A-Za-z]:[\\/]', value)):
            raise ValueError('Explicit observed runtime roots must be absolute paths')
        roots.setdefault(value, '<RUNTIME_' + key.upper() + '>')
    # External original inputs have stable registered tokens, never published roots.
    strings = []
    def walk(x):
        if isinstance(x, str): strings.append(x)
        elif isinstance(x, dict):
            for k, v in x.items(): strings.append(k); walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    for path in Path(scope).rglob('*.json'):
        try: walk(read(path))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Interrupted producers may leave partial metadata. Keep its raw
            # bytes privately; never let one failed case erase all45 positions.
            continue
    external = set()
    for string in strings:
        for candidate in re.findall(r'/(?:data1|data)/[^/\s"\']+/(?:project|projects|env|standard)', string):
            external.add(candidate)
    for index, value in enumerate(sorted(external)):
        roots.setdefault(value, f'<REGISTERED_EXTERNAL_ROOT_{index + 1}>')
    return roots


def checkpoint_approved(path, relative, evidence):
    proof = evidence.get(relative, {})
    names = proof.get('tensor_names', [])
    forbidden = ('base_lm.', 'esm.', 'qwen.', 'transformer.', 'model.layers.', 'model.embed_tokens.', 'lm_head.')
    return (Path(path).stat().st_size < 50 * 1024**2 and proof.get('public_safe') is True
        and proof.get('sha256') == sha(path) and proof.get('bytes') == Path(path).stat().st_size
        and proof.get('finite_CPU_trainable_state') is True and proof.get('independently_accepted_case') is True
        and bool(re.fullmatch('[0-9a-f]{64}', proof.get('producer_sha256', '')))
        and bool(names) and all(isinstance(n, str) and not n.startswith(forbidden) for n in names))


def publish_snapshot(source, destination, manifest, roots=None):
    """Reusable no-Git sanitizer for initial snapshots or completed result exports."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists(): raise FileExistsError('Public result namespace must be new')
    roots = roots or root_map(source, manifest)
    evidence = read(source / manifest['checkpoint_evidence']) if manifest.get('checkpoint_evidence') else {}
    records, excluded, curated = [], [], []
    destination.mkdir(parents=True, exist_ok=False)
    for item in manifest['files']:
        relative = item['path']; path = (source / relative).resolve(strict=True)
        if not path.is_relative_to(source) or path.is_symlink() or sha(path) != item['sha256']:
            raise ValueError('Input is outside verified export or its bytes changed')
        parts = PurePosixPath(relative).parts
        reason = publication_restriction(relative)
        if reason: pass
        elif '__pycache__' in parts or any('probe' in part.lower() for part in parts): reason = 'Private raw resource observation'
        elif 'supervision' in parts:
            reason = 'Private supervision identity; safe terminal fields curated separately'
            if path.name == 'finished.json':
                raw = read(path)
                keep = {k: raw[k] for k in ('status', 'exit_code', 'started_unix', 'finished_unix', 'elapsed_seconds') if k in raw}
                keep['remaining_group_member_count'] = len(raw.get('remaining_group_members', []))
                keep['guard'] = {k: v for k, v in raw.get('guard', {}).items() if k in ('reason', 'peak_group_rss_bytes', 'finished_unix')}
                curated.append({'source_relative': relative, 'source_sha256': item['sha256'], **keep})
        elif path.suffix in ('.log', '.zip', '.tar', '.gz', '.pyc'): reason = 'Raw logs/archive remain private'
        elif path.suffix == '.pt' and not checkpoint_approved(path, relative, evidence): reason = 'Checkpoint lacks bounded accepted trainable-state proof'
        elif path.suffix not in ('.json', '.jsonl', '.csv', '.md', '.txt', '.py', '.pt', '.npz'):
            reason = 'Unregistered binary/artifact type'
        elif path.suffix == '.npz' and path.name != 'description_vectors.npz': reason = 'Per-sequence cache payload omitted'
        if reason:
            excluded.append({'source_relative': relative, 'source_sha256': item['sha256'], 'reason': reason}); continue
        target = destination / relative; target.parent.mkdir(parents=True, exist_ok=True)
        transformed = False
        if path.suffix in ('.pt', '.npz'): shutil.copyfile(path, target)
        else:
            text = path.read_text(encoding='utf-8-sig')
            if path.suffix == '.json':
                try: original = read(path)
                except json.JSONDecodeError:
                    if PRIVATE.search(text) or SECRET.search(text):
                        excluded.append({'source_relative': relative, 'source_sha256': item['sha256'],
                            'reason': 'Incomplete private JSON cannot be normalized; original retained privately'})
                        continue
                    shutil.copyfile(path, target)
                    records.append({'source_relative': relative, 'public_relative': relative,
                        'source_sha256': item['sha256'], 'public_sha256': sha(target), 'bytes': target.stat().st_size,
                        'transformation': 'original_bytes_partial_JSON_unparsed'})
                    continue
                public = normalize(original, roots)
                transformed = original != public
                if transformed: write(target, public)
                else: shutil.copyfile(path, target)
            elif path.suffix == '.jsonl' and path.name == 'queue_events.jsonl':
                lines = []
                for number, line in enumerate(text.splitlines(), 1):
                    if not line.strip(): continue
                    try: lines.append(normalize(json.loads(line), roots))
                    except json.JSONDecodeError:
                        lines.append({'event': 'unparsed_partial_administrative_line', 'source_line_index': number,
                            'source_line_sha256': hashlib.sha256(line.encode('utf-8')).hexdigest(),
                            'publication_note': 'Interrupted raw administrative line retained privately'})
                # Administrative events carry runtime paths and identities.
                # They are distinct from immutable scientific row ledgers.
                target.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in lines), encoding='utf-8')
                transformed = True
            elif path.suffix == '.md':
                try: public = normalize(text, roots)
                except ValueError:
                    excluded.append({'source_relative': relative, 'source_sha256': item['sha256'],
                        'reason': 'Private diagnostic report retained; canonical structured analysis and scientific rows remain public'})
                    continue
                transformed = public != text
                if transformed: target.write_text(public, encoding='utf-8')
                else: shutil.copyfile(path, target)
            else:
                # Scientific predictions, histories and source are byte-exact.
                if SECRET.search(text) or PRIVATE.search(text): raise ValueError('Unsafe scientific/source bytes: ' + relative)
                shutil.copyfile(path, target)
        records.append({'source_relative': relative, 'public_relative': relative,
            'source_sha256': item['sha256'], 'public_sha256': sha(target), 'bytes': target.stat().st_size,
            'transformation': 'structured_metadata_path_and_identity_normalization' if transformed else 'original_bytes'})
    write(destination / 'CURATED_TERMINAL_RECEIPTS.json', curated)
    analysis_file = source / manifest.get('independent_analysis', '') / 'analysis.json'
    analysis = read(analysis_file) if analysis_file.is_file() else {}
    if manifest.get('plan_relative'):
        plan = read(source / manifest['plan_relative'])
        states = {}
        ledger = source / manifest['queue_relative'] / 'queue_events.jsonl'
        if ledger.is_file():
            for line in ledger.read_text(encoding='utf-8').splitlines():
                try: event = json.loads(line)
                except json.JSONDecodeError: continue
                if 'case_id' in event: states[event['case_id']] = event['status']
        write(destination / 'QUEUE_POSITIONS.json', [{**case, 'queue_status': states.get(case['case_id'], 'pending'),
            'analysis_status': next((r['analysis_status'] for r in analysis.get('cases', []) if r['case_id'] == case['case_id']), 'unavailable')}
            for case in plan['cases']])
    write(destination / 'PUBLICATION_MANIFEST.json', {'schema': 'v5-publication.v1', 'source_files': records,
        'excluded_files': excluded, 'expected_cases': 45, 'analysis_status_counts': analysis.get('analysis_status_counts'),
        'queue_status_counts': analysis.get('queue_status_counts'), 'completion_trigger': manifest.get('completion_trigger'),
        'source_export_plan_sha256': manifest.get('plan_sha256'), 'mean_SD_policy': 'Only complete3/3 fixed paired seeds; canonical mt_analyze output retained',
        'credentials_or_raw_process_inventory_published': False})
    (destination / 'STATUS.md').write_text('计划45组。独立复核状态：' + json.dumps(analysis.get('analysis_status_counts'), ensure_ascii=False)
        + '；队列状态：' + json.dumps(analysis.get('queue_status_counts'), ensure_ascii=False)
        + '。\n原失败、未启动和部分结果保留；未知状态不算成功，三个种子不完整不汇总均值／标准差。\n', encoding='utf-8')
    return destination


def call(argv, cwd=None):
    return subprocess.run([str(x) for x in argv], cwd=cwd, check=True, capture_output=True, text=True)


def transport_call(argv, *, attempts=3, runner=None, sleep=None):
    """Only bounded SSH/SCP transport recovery; never resumes training or Git."""
    runner, sleep = runner or call, sleep or time.sleep
    transient = re.compile(r'connection (?:reset|timed out|closed|refused)|broken pipe|network is unreachable|no route to host|kex_exchange_identification|lost connection', re.I)
    for index in range(attempts):
        try: return runner(argv)
        except subprocess.CalledProcessError as exc:
            if index + 1 == attempts or not transient.search(str(exc.stderr or '') + str(exc.stdout or '')): raise
            sleep(10 * (index + 1))


def isolated_worktree(repository, root, run_id):
    repository, root = Path(repository).resolve(strict=True), Path(root).resolve()
    target = root / safe_id(run_id)
    if target.exists(): raise FileExistsError('Never reuse or reset another worktree')
    root.mkdir(parents=True, exist_ok=True)
    call(['git', 'fetch', 'origin', 'main'], repository)
    call(['git', 'worktree', 'add', '--detach', str(target), 'origin/main'], repository)
    return target


def commit_and_verify(worktree, targets, run_id):
    for target in targets:
        relative = Path(target).relative_to(worktree)
        if not str(relative).replace('\\', '/').startswith('experiments/' + SCOPE_ID + '/'):
            raise ValueError('Publication path is outside owned V5 scope')
        call(['git', 'add', '-f', '--', str(relative)], worktree)
    message = ('Preserve the completed multitask evidence without disturbing active checkouts\n\n'
        'Publish one independently rescored45-case snapshot, including failures and missing positions.\n\n'
        'Constraint: Fixed data, scoring and seeds; isolated worktree only\n'
        'Tested: Private export SHA, safe extraction, metadata privacy and immutable prediction bytes\n'
        'Confidence: high\nScope-risk: narrow\n'
        'Directive: Do not replace missing seeds with historical runs\n'
        'Co-authored-by: OmX <omx@oh-my-codex.dev>')
    call(['git', 'commit', '-m', message], worktree)
    commit = call(['git', 'rev-parse', 'HEAD'], worktree).stdout.strip()
    call(['git', 'push', 'origin', 'HEAD:main'], worktree)
    remote_head = call(['git', 'ls-remote', 'origin', 'refs/heads/main'], worktree).stdout.split()[0]
    if remote_head != commit: raise RuntimeError('Remote head advanced; preserve worktree for explicit review')
    url = call(['git', 'remote', 'get-url', 'origin'], worktree).stdout.strip()
    match = re.fullmatch(r'(?:git@github.com:|https://github.com/)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?', url)
    if not match: raise ValueError('Anonymous verification requires a public GitHub origin without credentials')
    verified = []
    base = targets[0]
    for name in ('PUBLICATION_MANIFEST.json', 'STATUS.md'):
        path = base / name; relative = path.relative_to(worktree).as_posix()
        request = urllib.request.Request('https://raw.githubusercontent.com/' + match[1] + '/' + commit + '/' + urllib.parse.quote(relative), headers={'User-Agent': 'V5-public-evidence-verification'})
        with urllib.request.urlopen(request, timeout=60) as response: payload = response.read()
        if hashlib.sha256(payload).hexdigest() != sha(path): raise ValueError('Anonymous public bytes differ')
        verified.append({'relative': relative, 'sha256': sha(path)})
    return {'commit': commit, 'remote_head': remote_head, 'anonymous_verified': verified, 'run_id': run_id}


def synchronize(args):
    run_id = safe_id(args.run_id)
    project = Path(args.project_root).resolve(strict=True)
    scope = project / 'experiments' / SCOPE_ID
    local = scope / 'resources' / 'completion_exports' / run_id
    local.mkdir(parents=True, exist_ok=False)
    remote_scope = PurePosixPath(args.remote_scope)
    if not remote_scope.is_absolute() or remote_scope.name != SCOPE_ID: raise ValueError('Explicit owned remote scope required')
    queue = PurePosixPath(args.queue_root)
    if not queue.is_relative_to(remote_scope): raise ValueError('Queue outside owned remote scope')
    remote_output = remote_scope / 'resources' / 'completion_exports' / run_id
    # The existing frozen runtime is passed explicitly; no credential/config copy.
    argv = [args.remote_python, str(remote_scope / 'resources/finish_and_export.py'), '--scope-root', str(remote_scope),
            '--queue-root', str(queue), '--output', str(remote_output)] + (['--wait'] if args.wait else [])
    ssh = transport_call(['ssh', '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6', 'node1', shlex.join(argv)])
    receipt = json.loads(ssh.stdout.strip().splitlines()[-1])
    write(local / 'remote_export_receipt.json', receipt)
    expected = str(remote_output / 'results.tar.gz')
    if receipt.get('archive') != expected: raise ValueError('Unexpected remote archive location')
    archive = local / 'results.tar.gz'
    transport_call(['scp', '-q', 'node1:' + shlex.quote(expected), str(archive)])
    if sha(archive) != receipt['sha256'] or archive.stat().st_size != receipt['bytes']: raise ValueError('Downloaded archive SHA/size differs')
    source = safe_extract(archive, local / 'private')
    manifest = verify_export(source, receipt['manifest'], receipt['manifest_sha256'])
    worktree = isolated_worktree(args.public_repo, scope / 'resources/sync_worktrees', run_id)
    base = worktree / 'experiments' / SCOPE_ID
    public = publish_snapshot(source, base / 'results' / run_id, manifest)
    analysis = base / 'analysis' / run_id
    original = public / manifest['independent_analysis']
    if original.is_dir(): shutil.copytree(original, analysis)
    targets = [public] + ([analysis] if analysis.is_dir() else [])
    result = read(public / 'PUBLICATION_MANIFEST.json')
    completion_status = base / 'COMPLETION_STATUS.json'
    write(completion_status, {'run_id': run_id, 'expected_cases': 45,
        'analysis_status_counts': result['analysis_status_counts'], 'queue_status_counts': result['queue_status_counts'],
        'completion_trigger': result['completion_trigger'], 'results': 'results/' + run_id,
        'analysis': 'analysis/' + run_id if analysis.is_dir() else None,
        'all_positions_preserved': True, 'three_complete_seeds_required_for_mean_SD': True})
    targets.append(completion_status)
    verification = commit_and_verify(worktree, targets, run_id)
    write(local / 'PUBLICATION_RECEIPT.json', verification)
    return verification


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('remote-scope', 'queue-root', 'project-root', 'public-repo', 'run-id', 'remote-python'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--wait', action='store_true')
    args = p.parse_args()
    print(json.dumps(synchronize(args), ensure_ascii=False), flush=True)


if __name__ == '__main__': main()

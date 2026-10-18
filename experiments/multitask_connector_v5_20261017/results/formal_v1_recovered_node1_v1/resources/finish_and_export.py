"""One task-linked completion waiter and private export; never trains or resumes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

MAX_CHECKPOINT_BYTES = 50 * 1024**2


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def identity_alive(identity):
    """Only an absent/reused exact Linux identity is evidence of death."""
    if not isinstance(identity, dict) or not {'pid', 'pgid', 'sid', 'start_ticks'} <= set(identity):
        raise ValueError('Complete controller identity required before declaring death')
    try:
        fields = Path(f"/proc/{int(identity['pid'])}/stat").read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    current = {'pid': identity['pid'], 'pgid': int(fields[2]), 'sid': int(fields[3]), 'start_ticks': int(fields[19])}
    return fields[0] != 'Z' and all(current[k] == identity[k] for k in current)


def ready(queue, plan):
    complete = (queue / 'complete.json').exists()
    owner = queue / 'controller.json'
    dead = owner.is_file() and not identity_alive(read(owner)['identity'])
    if complete and ((queue / 'analysis_exit.json').exists() or not plan.get('analysis_command') or dead):
        return 'queue_terminal'
    if dead:
        # Controller loss does not stop its independent supervised workers.
        from src.mt_supervise import group_members
        for name in ('supervision/started.json', 'supervisor_owner.json'):
            for path in (queue / 'cases').glob('*/attempt_*/' + name):
                identity = read(path).get('identity')
                if identity and group_members(identity): return None
        return 'controller_dead_partial'
    return None


def checkpoint_state_allowed(names):
    forbidden = ('base_lm.', 'esm.', 'qwen.', 'transformer.', 'model.layers.', 'model.embed_tokens.', 'lm_head.')
    return bool(names) and all(isinstance(name, str) and not name.startswith(forbidden) for name in names)


def existing_export(scope, queue, output):
    """Recover only a completed, byte-verified export for the same requested roots."""
    receipt_path = output / 'export_receipt.json'
    if not receipt_path.is_file(): raise FileExistsError('Incomplete prior export preserved; choose a fresh output ID')
    receipt = read(receipt_path)
    archive, manifest_path = output / 'results.tar.gz', output / 'EXPORT_MANIFEST.json'
    if (receipt.get('archive') != str(archive) or not archive.is_file()
            or archive.stat().st_size != receipt.get('bytes') or sha(archive) != receipt.get('sha256')
            or sha(manifest_path) != receipt.get('manifest_sha256')):
        raise ValueError('Existing completed export SHA/size binding differs')
    manifest = read(manifest_path)
    if manifest.get('scope_root') != str(scope) or manifest.get('queue_root') != str(queue):
        raise ValueError('Existing export belongs to different scope/queue arguments')
    return {**receipt, 'reused_existing_verified_export': True}


def checkpoint_receipts(scope, queue, analysis, plan):
    """Only independently accepted formal trainable tensors can become public."""
    accepted = {r['case_id'] for r in analysis.get('cases', []) if r.get('analysis_status') == 'accepted'}
    producer = sha(scope / 'src/mt_train.py')
    pinned = plan.get('source_pins', {}).get(str(scope / 'src/mt_train.py'))
    receipts = {}
    for path in (queue / 'cases').glob('*/attempt_*/artifacts/checkpoint.pt'):
        item = {'sha256': sha(path), 'bytes': path.stat().st_size, 'public_safe': False}
        try:
            if path.parents[2].name not in accepted: raise ValueError('Case not independently accepted')
            if item['bytes'] >= MAX_CHECKPOINT_BYTES: raise ValueError('Checkpoint is not below50MiB')
            manifest = read(path.parent / 'run_manifest.json')
            if pinned != producer or manifest.get('code_sha256', {}).get('mt_train.py') != producer:
                raise ValueError('Actual frozen producer pin not proven')
            import torch
            payload = torch.load(path, map_location='cpu', weights_only=True)
            state = payload.get('trainable_state', {})
            if payload.get('schema') != 'mt-checkpoint.v1' or not checkpoint_state_allowed(state):
                raise ValueError('Checkpoint contains unexpected or backbone state')
            for key in ('scope_id', 'task', 'method', 'backbone', 'seed', 'config_sha256'):
                if payload.get('identity', {}).get(key) != manifest.get(key): raise ValueError('Checkpoint identity differs')
            if any(not isinstance(v, torch.Tensor) or v.device.type != 'cpu' or not torch.isfinite(v).all()
                   or (v.is_floating_point() and v.dtype != torch.float32) for v in state.values()):
                raise ValueError('Only finite CPU FP32 trainable tensor state is permitted')
            item.update(public_safe=True, producer_sha256=producer, tensor_names=list(state),
                        finite_CPU_trainable_state=True, independently_accepted_case=True)
        except Exception as exc:
            item['exclusion_reason'] = str(exc)
        receipts[path.relative_to(scope).as_posix()] = item
    return receipts


def finish(scope, queue, output, wait=False, poll_seconds=15):
    scope, queue, output = Path(scope).resolve(strict=True), Path(queue).resolve(strict=True), Path(output).resolve()
    if not queue.is_relative_to(scope) or not output.is_relative_to(scope): raise ValueError('Owned scope roots required')
    if output.exists(): return existing_export(scope, queue, output)
    binding = read(queue / 'queue_binding.json')
    plan_path = Path(binding['plan_path']).resolve(strict=True)
    if not plan_path.is_relative_to(scope) or sha(plan_path) != binding['plan_sha256']: raise ValueError('Frozen queue plan changed or escaped scope')
    plan = read(plan_path)
    if Path(plan['paths']['scope_root']).resolve() != scope or len(plan['cases']) != 45:
        raise ValueError('Not the owned45-case scope')
    sys.path.insert(0, str(scope))
    partial_quiet_since = None
    while True:
        state = ready(queue, plan)
        if state == 'queue_terminal': break
        if state == 'controller_dead_partial':
            if partial_quiet_since is None: partial_quiet_since = time.monotonic()
            if time.monotonic() - partial_quiet_since >= 35: break
        else:
            partial_quiet_since = None
        if not wait: raise RuntimeError('Queue not terminal; pass --wait for one blocking completion process')
        time.sleep(poll_seconds)
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    independent = output / 'independent_analysis'
    with (output / 'analysis_stdout.log').open('xb') as stdout, (output / 'analysis_stderr.log').open('xb') as stderr:
        result = subprocess.run([sys.executable, '-m', 'src.mt_analyze', '--plan', str(plan_path),
            '--queue-root', str(queue), '--data-root', plan['paths']['data_root'], '--output', str(independent)],
            cwd=scope, env=env, stdout=stdout, stderr=stderr, timeout=7200)
    analysis = read(independent / 'analysis.json') if (independent / 'analysis.json').is_file() else {}
    evidence = checkpoint_receipts(scope, queue, analysis, plan)
    write(output / 'checkpoint_publication.json', evidence)
    write(output / 'export_status.json', {'completion_trigger': state, 'analysis_exit_code': result.returncode,
        'expected_cases': 45, 'analysis_status_counts': analysis.get('analysis_status_counts'),
        'queue_status_counts': analysis.get('queue_status_counts'), 'no_retraining_or_resume': True,
        'raw_supervision_and_resource_probes_are_private': True})
    files = {plan_path, Path(__file__).resolve()}
    if (scope / 'resources/sync_results.py').is_file(): files.add(scope / 'resources/sync_results.py')
    for root in (queue, scope / 'analysis', scope / 'configs', scope / 'src', scope / 'docs'):
        if root.is_dir(): files.update(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for name in ('README.md', 'data/manifest.json'):
        path = scope / name
        if path.is_file(): files.add(path)
    cache = Path(plan['paths']['cache_root']).resolve()
    if not cache.is_relative_to(scope): raise ValueError('Cache metadata must belong to new scope')
    for name in ('manifest.json', 'cache_spec.json', 'description_vectors.json', 'description_vectors.npz'):
        if (cache / name).is_file(): files.add(cache / name)
    files.update(p for p in output.rglob('*') if p.is_file())
    if any(not p.resolve().is_relative_to(scope) for p in files): raise ValueError('Export input escapes owned scope')
    entries = [{'path': p.relative_to(scope).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(files)]
    write(output / 'EXPORT_MANIFEST.json', {'schema': 'v5-private-completion-export.v1',
        'scope_id': scope.name, 'scope_root': str(scope), 'queue_root': str(queue),
        'plan_path': str(plan_path), 'plan_relative': plan_path.relative_to(scope).as_posix(),
        'queue_relative': queue.relative_to(scope).as_posix(), 'plan_sha256': binding['plan_sha256'],
        'runtime_roots': {'sys_prefix': sys.prefix, 'sys_base_prefix': sys.base_prefix,
            'sys_base_exec_prefix': sys.base_exec_prefix, 'sys_executable': sys.executable},
        'independent_analysis': independent.relative_to(scope).as_posix(),
        'checkpoint_evidence': (output / 'checkpoint_publication.json').relative_to(scope).as_posix(),
        'completion_trigger': state, 'analysis_exit_code': result.returncode, 'files': entries,
        'omitted': ['pretrained weights', 'per-sequence feature payload cache'],
        'publication_requires_private_metadata_filter': True})
    files.add(output / 'EXPORT_MANIFEST.json')
    archive = output / 'results.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for path in sorted(files):
            if path.is_symlink(): raise ValueError('Symlink export rejected')
            tar.add(path, arcname=path.relative_to(scope).as_posix(), recursive=False)
    if any(sha(scope / item['path']) != item['sha256'] for item in entries):
        raise RuntimeError('An artifact changed during export; partial archive retained privately, not released')
    represented = {item['path'] for item in entries}
    if any(p.relative_to(scope).as_posix() not in represented for p in queue.rglob('*')
           if p.is_file() and '__pycache__' not in p.parts):
        raise RuntimeError('New queue artifacts appeared during export; archive not released')
    receipt = {'status': 'exported', 'archive': str(archive), 'sha256': sha(archive),
               'bytes': archive.stat().st_size, 'manifest': (output / 'EXPORT_MANIFEST.json').relative_to(scope).as_posix(),
               'manifest_sha256': sha(output / 'EXPORT_MANIFEST.json'), 'analysis_exit_code': result.returncode}
    write(output / 'export_receipt.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('scope-root', 'queue-root', 'output'): p.add_argument('--' + key, required=True)
    p.add_argument('--wait', action='store_true')
    args = p.parse_args()
    print(json.dumps(finish(args.scope_root, args.queue_root, args.output, args.wait), ensure_ascii=False), flush=True)


if __name__ == '__main__': main()

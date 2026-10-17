"""Create fresh standard contracts with actual file hashes, immediately before launch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

from mm_supervise import probe, sha, standard_module, write_new

FIRST = '2026-10-17T00:29:40+08:00'
DEADLINE = '2026-10-17T10:29:40+08:00'


def create(request, standard_root, output):
    """Hash expensive inputs first; compute a conservative timeout only afterwards."""
    standard = Path(standard_root).resolve()
    validator_path = standard / 'validate_reproduction.py'
    contract = json.loads((standard / 'task-contract.template.json').read_text(encoding='utf-8-sig'))
    project = Path(request['project_root']).resolve()
    scope = Path(request['scope_root']).resolve()
    if scope != project and project not in scope.parents:
        raise ValueError('Scope must be contained in its dedicated project root')
    output = Path(output).resolve()
    if scope not in output.parents:
        raise ValueError('Contract output must be inside the new scope')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # The canonical file checker requires log references to exist at admission.
    # They remain empty until the supervised worker starts; never reuse a log.
    for log_name in ('stdout.log', 'stderr.log'):
        with (output.parent / log_name).open('x'):
            pass
    # Canonical standard paths and actual model bytes are frozen, never inferred.
    pdf = project / 'papers/bench_v02/ppikb/article.pdf'
    pdf_sha = sha(pdf)
    datasets = []
    for item in request['datasets']:
        path = Path(item['path']).resolve()
        observed = sha(path)
        if observed != item['sha256']:
            raise ValueError(f'Data hash mismatch: {path}')
        datasets.append(dict(name=item['name'], version='a1_locked_v2', path=str(path),
                             sha256=observed, manifest_path=str(path), manifest_sha256=observed,
                             source_url='https://ppikb.duanlab.ac/', integrity='complete',
                             statistical_unit='frozen database measurement', notes=item.get('notes')))
    models = []
    for item in request.get('models', []):
        path = Path(item['path']).resolve()
        digest = sha(path)
        if item.get('sha256') and digest != item['sha256']:
            raise ValueError(f'Model hash mismatch: {path}')
        models.append(dict(name=item['name'], path=str(path), sha256=digest,
                           integrity='complete', coverage=item.get('coverage', 'frozen backbone')))
    if not models:
        models = [dict(name='No model required for scalar TRAIN-statistic baseline', integrity='not_applicable',
                       notes='This CPU-only contract computes TRAIN mean/median; no model inference or training.')]
    codes = []
    for pathstr in request['code_paths']:
        path = Path(pathstr).resolve()
        codes.append(dict(role='component', path=str(path), sha256=sha(path), archive_sha256=sha(path), integrity='complete'))
    lockfile = Path(request['environment_lock']).resolve()
    lock_sha = sha(lockfile)
    resources = request['resources']
    current = probe(resources['host'], resources.get('gpu'), resources['memory_limit_bytes'], scope)
    probe_path = output.parent / 'contract_resource_probe.json'
    write_new(probe_path, current)
    if current['readiness'] != 'ready':
        write_new(output.parent / 'contract_not_frozen.json', {'status': 'not_started', 'reason': 'resource_busy', 'probe': current})
        return None
    now = datetime.now(timezone.utc)
    remaining = (datetime.fromisoformat(DEADLINE) - now).total_seconds()
    # 120 s reserve covers validation/spawn. A second actual-start guard is mandatory.
    timeout = min(int(request['timeout_seconds']), int(remaining - 120))
    if timeout <= 0:
        raise RuntimeError('Deadline exhausted; cannot freeze execution contract')
    stamp = now.isoformat()
    contract.update(is_template=False, task_id='molecular_memory_v3_20261017', run_id=request['run_id'],
                    contract_status='frozen', created_at=stamp, frozen_at=stamp)
    contract['identity'].update(title='A data resource of peptide–protein interaction structures and binding affinities for peptide drug discovery',
        doi='10.1038/s41597-026-07962-1', publication_status='published', journal='Scientific Data',
        paper_type='Data provenance; new connector experiment, not a claim of author-table reproduction',
        pdf=dict(path=str(pdf), sha256=pdf_sha, version='accepted_manuscript', source_url='https://doi.org/10.1038/s41597-026-07962-1'))
    contract['execution_scope'].update(objective=request['objective'],
        authorization_reference='User requested evaluation and implementation of the attached new architecture, automatic fixed-split controlled multiseed benchmarks, node1 GPUs preferred; 2026-10-17 scope.',
        relation_to_prior_run='New architecture objective; old 20261016 contracts, failures and results remain closed and immutable.',
        safety_assessment='allowed', scope_coverage='component_only', intended_level=request.get('level', 'smoke'),
        excluded_claims=['biological sequence generation', 'affinity optimization', 'full-paper reproduction', 'new unseen confirmatory test'],
        restriction_check=dict(prior_platform_restriction=False, reason='Generic scalar prediction benchmark with fixed public measurements; no sequence generation or biological enhancement.', evidence_paths=[str(scope / 'configs/development.json')]))
    contract['timing'].update(first_substantive_started_at=FIRST, earliest_associated_preparation_at=FIRST,
        original_deadline_at=DEADLINE, effective_deadline_at=DEADLINE, prior_window_expired=False,
        retry_resets_clock=False, timing_evidence_paths=[str(scope / 'configs/development.json')])
    contract['assets'].update(datasets=datasets, code=codes, models=models)
    contract['targets'] = [dict(target_id='registered_endpoint', required=True,
        paper_endpoint='New preregistered component endpoint', metric=request.get('metric', 'finite_runtime_output'),
        metric_definition=request.get('metric_definition', 'Finite structured output with declared denominator and independently testable serialization.'),
        statistical_unit='TRAIN canary' if request.get('phase') == 'canary' else 'frozen measurement',
        denominator=request.get('denominator'), unit=request.get('unit', 'diagnostic'),
        paper_value=None, paper_value_source=None, comparison_endpoint=request['objective'],
        checkpoint_selection=request.get('checkpoint_selection', 'none; engineering canary'),
        repetitions={'seeds': request.get('seeds', [])},
        tolerance=dict(kind='not_comparable', absolute=None, relative=None,
                       reason='New component implementation; no equivalent author numerical endpoint.'))]
    contract['environment'].update(isolation='Existing task runtime, dedicated new scope; no shared environment changes',
        environment_path=request['environment_path'], lockfile_path=str(lockfile), lockfile_sha256=lock_sha,
        shared_environment_modified=False)
    contract['environment']['numerical_policy'].update(request.get('numerical_policy', {}))
    contract['resources'].update(expected_hostname=resources['host'], cpu_threads=resources.get('cpu_threads', 2),
        data_loader_workers=0, memory_limit_bytes=resources['memory_limit_bytes'], gpu_devices=[] if resources.get('gpu') is None else [resources['gpu']],
        gpu_memory_budget_bytes=resources.get('gpu_memory_budget_bytes'), resource_probe_path=str(probe_path),
        resource_probe_at=current['utc'], start_gate=current['readiness'],
        budget_basis='Fresh CPU idle>=8%, I/O wait<=10%, host RAM reserve4GiB, storage10GiB; selected GPU no compute process/util<=5%/used<=256MiB. Own process group timeout/RSS monitored.', no_preemption=True)
    contract['execution'].update(command=request['command'], working_directory=str(scope),
        seed_policy='Fixed seeds 17/42/2026, paired untrained initialization; no score-driven seed selection',
        seeds=request.get('seeds', []), timeout_seconds=timeout,
        process_control=dict(platform='Linux', mechanism='mm_supervise POSIX new session; group RSS and monotonic timeout; TERM then KILL only own PGID; exit and remaining-member verification.', whole_tree_bounded=True),
        stdout_path=str(output.parent / 'stdout.log'), stderr_path=str(output.parent / 'stderr.log'))
    contract['review'].update(status='accepted', reviewer='scope-owner automated admission with canonical validator', reviewed_at=stamp,
                             checks=['Fresh resources', 'Actual byte hashes', 'Deadline margin', 'New scope and fixed input binding'])
    for key in ('baseline_run_binding', 'training_run_binding', 'cache_run_binding'):
        if key in request:
            contract[key] = request[key]
    contract['request_sha256'] = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    write_new(output, contract)
    validation = standard_module(validator_path).validate(contract, contract_bytes=output.read_bytes(),
                                                        check_files=True, contract_directory=output.parent)
    write_new(output.parent / 'contract_validation.json', validation)
    if not validation['valid_for_execution']:
        raise RuntimeError('Standard contract validation failed; retained exact contract and receipt')
    return contract


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', required=True)
    p.add_argument('--standard-root', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    request = json.loads(Path(args.request).read_text(encoding='utf-8-sig'))
    return 0 if create(request, args.standard_root, args.output) is not None else 3


if __name__ == '__main__':
    raise SystemExit(main())

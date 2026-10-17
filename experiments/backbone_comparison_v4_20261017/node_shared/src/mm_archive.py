"""Archive existing bounded runs against the canonical result-record template.

No fit, model import, workload launch or modification of a source artifact.
Run on the execution host so canonical check_files never skips POSIX assets.
"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import statistics


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def pairs(values):
        if len({k for k, _ in values}) != len(values):
            raise ValueError('duplicate JSON key')
        return dict(values)
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    def scan(v):
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError('nonfinite JSON number')
        if isinstance(v, dict):
            for item in v.values(): scan(item)
        elif isinstance(v, list):
            for item in v: scan(item)
    scan(value)
    return value


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def stamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'timezone required')
    return result


def _pointer(value, pointer):
    if pointer == '': return value
    require(isinstance(pointer, str) and pointer.startswith('/'), 'JSON pointer required')
    for part in pointer[1:].split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def _safe_file(root, relative):
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'completion reference must be relative')
    result = (root / rel).resolve(strict=True)
    require(result.is_file() and result.is_relative_to(root), 'completion file outside artifact root')
    return result


def _scalar_complete(contract, roots):
    binding = contract['baseline_run_binding']
    directory = Path(binding['output_directory']).resolve(strict=True)
    require(directory in roots, 'scalar bound result directory must be an artifact root')
    result = read(directory / 'result.json')
    require(result.get('status') == 'complete' and result.get('fit_performed') is True, 'scalar run incomplete')
    require(result.get('request_sha256') == binding['request_sha256'], 'result request binding differs')
    request = json.loads(binding['request_canonical_json'])
    require(result['request'] == request and result['scenario'] == request['scenario'], 'scalar request/scenario mismatch')
    require(result['authorization']['contract_sha256'] == sha(contract['_archive_contract_file']), 'result authorization mismatch')
    require(set(result['models']) == set(request['methods']), 'not every requested reference is present')
    for name, expected in result['artifact_sha256'].items():
        require(sha(_safe_file(directory, name)) == expected, 'scalar artifact SHA mismatch: ' + name)
    evidence = []
    data_manifest = next(Path(x['path']) for x in contract['assets']['datasets'] if x['path'].endswith('/data_views/manifest.json'))
    view_manifest = read(data_manifest)
    view_receipt = view_manifest['scenarios'][request['scenario']]
    expected_rows = {}
    for split in ('validation', 'test'):
        view_path = _safe_file(data_manifest.parent, view_receipt['views'][split]['path'])
        require(sha(view_path) == view_receipt['views'][split]['sha256'], 'prepared-view SHA mismatch')
        expected_rows[split] = [json.loads(line) for line in view_path.read_text(encoding='utf-8').splitlines()]
    gold_path = next(Path(x['path']) for x in contract['assets']['datasets'] if x['name'] == 'test_gold.jsonl')
    gold = [json.loads(line) for line in gold_path.read_text(encoding='utf-8').splitlines()]
    require([r['sample_id'] for r in gold] == [r['sample_id'] for r in expected_rows['test']], 'source gold/input ID mismatch')
    expected_rows['test'] = [{**r, 'pkd': g['pkd']} for r, g in zip(expected_rows['test'], gold)]
    expected_test = contract['targets'][0]['denominator']
    require(result['split_counts']['test'] == expected_test, 'contract test denominator mismatch')
    for method in request['methods']:
        state = read(directory / (method + '.state.json'))
        require(state['scenario'] == request['scenario'], 'TRAIN state scenario mismatch')
        require(state['train_statistics']['train_count'] == result['split_counts']['train'], 'TRAIN denominator mismatch')
        for split in ('validation', 'test'):
            count = result['split_counts'][split]
            metrics = result['models'][method][split]
            require(metrics['n_total'] == metrics['n_finite'] == count and metrics['numeric_coverage'] == 1., 'incomplete numeric denominator')
            prediction = directory / (method + '.' + split + '.predictions.jsonl')
            rows = [json.loads(line) for line in prediction.read_text(encoding='utf-8').splitlines()]
            require(len(rows) == count and len({r['sample_id'] for r in rows}) == count, 'prediction coverage mismatch')
            expected = expected_rows[split]
            require([(r['sample_id'], r['gold_pkd']) for r in rows] == [(r['sample_id'], r['pkd']) for r in expected], 'predictions differ from frozen IDs/gold')
            require(all(r['scenario'] == request['scenario'] and r['split'] == split
                        and r['generation_performed'] is False and r['channel'] == 'numeric_predictor' for r in rows), 'prediction identity/channel mismatch')
            require(all(type(r[k]) in (int, float) and math.isfinite(r[k]) for r in rows for k in ('gold_pkd', 'prediction_pkd')), 'nonfinite numeric rows')
            errors = [abs(r['gold_pkd'] - r['prediction_pkd']) for r in rows]
            require(math.fsum(errors) / count == metrics['micro_MAE'], 'saved MAE differs from complete predictions')
            require(math.hypot(*errors) / math.sqrt(count) == metrics['RMSE'], 'saved RMSE differs from complete predictions')
            if method in ('train_mean', 'train_median'):
                require(all(r['prediction_pkd'] == state['constant'] for r in rows), 'saved constant predictions differ from TRAIN state')
                require(metrics['global_Spearman'] is None, 'constant Spearman must remain undefined')
            groups = {}
            for r, error in zip(expected, errors): groups.setdefault(r['target_sequence'], []).append(error)
            macro = statistics.mean(math.fsum(v) / len(v) for v in groups.values())
            require(macro == metrics['exact_target_macro_MAE'] and len(groups) == metrics['n_exact_targets'], 'target macro denominator/MAE mismatch')
            evidence.append(str(prediction))
    return [str(directory / 'result.json')] + evidence


def _generic_complete(roots, spec, targets):
    """Caller supplies preregistered file/pointer assertions, never fitted values.

    spec={artifact_root: absolute, assertions:[{file:relative,pointer:'/status',equals:'complete'},
          {file:relative,pointer:'/evaluation/test/rows',equals:registered_count}],
          target_counts:{target_id:{file:relative,pointer:'/evaluation/test/rows'}}}.
    A generic exit-zero receipt alone is insufficient for scientific completeness.
    """
    require(isinstance(spec, dict) and bool(spec.get('assertions')), 'generic run needs explicit completeness assertions')
    root = Path(spec['artifact_root']).resolve(strict=True)
    require(root in roots, 'completion root must be an artifact root')
    evidence = []
    for check in spec['assertions']:
        path = _safe_file(root, check['file'])
        observed = _pointer(read(path), check['pointer'])
        require(observed == check['equals'] and type(observed) is type(check['equals']), 'completion assertion failed: ' + check['file'] + check['pointer'])
        evidence.append(str(path))
    for target in targets:
        if target.get('required'):
            check = spec.get('target_counts', {}).get(target['target_id'])
            require(isinstance(check, dict), 'required contract denominator assertion missing')
            path = _safe_file(root, check['file'])
            observed = _pointer(read(path), check['pointer'])
            require(observed == target['denominator'] and type(observed) is type(target['denominator']), 'observed contract denominator mismatch')
            evidence.append(str(path))
    return list(dict.fromkeys(evidence))


def archive(contract_path, supervision_dir, artifact_roots, standard_root, output,
            *, completion_spec=None, small_file_limit_bytes=64 * 1024**2):
    """Create an exclusive archive directory; return paths, status and validator receipt.

    Unknown/incomplete runs are retained with needs_revision; no full-paper promotion.
    For scalar baselines completeness is independently reconstructed from saved rows.
    Other runs require completion_spec plus per-required-target denominator assertions.
    """
    contract_path = Path(contract_path).resolve(strict=True)
    supervision = Path(supervision_dir).resolve(strict=True)
    roots = [Path(p).resolve(strict=True) for p in artifact_roots]
    require(bool(roots) and all(p.is_dir() for p in roots), 'existing artifact directories required')
    output = Path(output).resolve()
    require(all(output != p and p not in output.parents for p in roots), 'archive output cannot be within its inventoried artifact roots')
    output.mkdir(parents=True, exist_ok=False)
    standard = Path(standard_root).resolve(strict=True)
    raw = contract_path.read_bytes()
    contract = read(contract_path)
    template = standard / 'result-record.template.json'
    validator = standard / 'validate_reproduction.py'
    record = read(template)
    errors, evidence = [], []
    supervision_receipt = supervision / 'receipt.json'
    started_file, probe_file = supervision / 'started.json', supervision / 'startup_resource_probe.json'
    receipt = read(supervision_receipt) if supervision_receipt.exists() else {}
    started = read(started_file) if started_file.exists() else {}
    probe = read(probe_file) if probe_file.exists() else {}
    actual_start = started.get('started_at', receipt.get('started_at'))
    def check_execution():
        require(receipt.get('contract_sha256') == hashlib.sha256(raw).hexdigest(), 'supervisor contract SHA mismatch')
        require(started.get('contract_sha256') == receipt['contract_sha256'], 'actual-start contract SHA mismatch')
        require(receipt.get('status') == 'succeeded' and receipt.get('exit_code') == 0 and receipt.get('termination_reason') is None, 'supervision did not succeed')
        require(receipt.get('remaining_group_members') == [] and bool(receipt.get('handles_absent_verified_at')), 'process group cleanup unverified')
        require(type(receipt.get('pid')) is int and receipt['pid'] > 0 and receipt['pid'] == receipt.get('pgid') == started.get('pid') == started.get('pgid'), 'PID/PGID evidence mismatch')
        require(probe.get('hostname') == contract['resources']['expected_hostname'] == socket.gethostname(), 'archive must verify files on actual execution host')
        require(stamp(actual_start) >= stamp(contract['frozen_at']), 'execution before contract freeze')
        require(stamp(receipt['ended_at']) <= stamp(contract['timing']['effective_deadline_at']), 'execution exceeded deadline')
        require(stamp(receipt['ended_at']) >= stamp(actual_start), 'end before actual process start')
        require(stamp(actual_start) + timedelta(seconds=contract['execution']['timeout_seconds']) <= stamp(contract['timing']['effective_deadline_at']), 'timeout exceeded remaining start window')
        require(receipt['elapsed_seconds'] <= contract['execution']['timeout_seconds'], 'supervised elapsed exceeded timeout')
        require(receipt['peak_group_rss_bytes'] <= contract['resources']['memory_limit_bytes'], 'RSS budget exceeded')
        for key in ('stdout', 'stderr'):
            require(sha(contract['execution'][key + '_path']) == receipt[key + '_sha256'], key + ' SHA mismatch')
    try:
        check_execution()
        if 'baseline_run_binding' in contract:
            evidence = _scalar_complete({**contract, '_archive_contract_file': str(contract_path)}, roots)
        else:
            evidence = _generic_complete(roots, completion_spec, contract['targets'])
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        errors.append({'code': 'execution_or_completeness_unverified', 'description': str(exc)})
    now = datetime.now(timezone.utc).isoformat()
    record.update(is_template=False, task_id=contract['task_id'], run_id=contract['run_id'],
        contract_path=str(contract_path), contract_sha256=hashlib.sha256(raw).hexdigest(), recorded_at=now,
        lifecycle_status='succeeded' if not errors else 'failed', reproduction_level='component' if started else 'static_audit',
        scope_coverage='component_only')
    execution = contract['execution']
    record['execution'].update(performed=bool(started), command=execution['command'], working_directory=execution['working_directory'],
        environment_path=contract['environment']['environment_path'], hostname=probe.get('hostname'), resource_probe_path=str(probe_file),
        resource_usage_path=str(supervision_receipt), seeds=execution.get('seeds', []), seed_notes=execution.get('seed_policy'),
        started_at=actual_start, ended_at=receipt.get('ended_at'), effective_deadline_at=contract['timing']['effective_deadline_at'],
        timeout_seconds=execution['timeout_seconds'], pid=receipt.get('pid'), pgid=receipt.get('pgid'),
        job_object_or_tree_control=execution.get('process_control'), exit_code=receipt.get('exit_code'),
        timed_out=receipt.get('termination_reason') in ('timeout', 'absolute_deadline'), termination_reason=receipt.get('termination_reason'),
        handles_absent_verified_at=receipt.get('handles_absent_verified_at'), exit_verification_path=str(supervision_receipt),
        stdout_path=execution['stdout_path'], stderr_path=execution['stderr_path'], receipt_path=str(supervision_receipt),
        notes='started_at uses the saved actual Popen-start receipt; RSS is the supervised process-group peak.',
        peak_group_rss_bytes=receipt.get('peak_group_rss_bytes'), elapsed_seconds=receipt.get('elapsed_seconds'))
    record['metric_comparisons'] = []
    for target in contract['targets']:
        comparison = {k: target.get(k) for k in ('target_id', 'paper_endpoint', 'metric', 'metric_definition', 'statistical_unit', 'denominator', 'unit', 'paper_value')}
        comparison.update(actual_value=None, baseline_value=None, absolute_difference=None, relative_difference=None,
            comparability='not_comparable', tolerance_reference=target.get('tolerance'), passed=None,
            reason='Component result vector and exact denominators are retained in artifacts; no scalar conversion or author-paper equivalence is claimed.', evidence_paths=evidence)
        record['metric_comparisons'].append(comparison)
    inventory = []
    for root in list(dict.fromkeys(roots + [supervision])):
        for path in sorted(root.rglob('*')):
            if path.is_file():
                size = path.stat().st_size
                inventory.append({'role': 'preserved_run_artifact', 'path': str(path), 'location': 'local', 'available': True,
                    'size_bytes': size, 'sha256': sha(path) if size <= small_file_limit_bytes else None,
                    'notes': None if size <= small_file_limit_bytes else 'Large artifact: indexed by size; consult original frozen manifest for byte digest.'})
    record['artifacts'] = inventory
    record['assets_integrity'] = {}
    for category in ('datasets', 'code', 'models'):
        entries = contract['assets'][category]
        applicable = [entry for entry in entries if entry.get('integrity') != 'not_applicable']
        record['assets_integrity'][category] = {'status': 'complete' if applicable else 'not_applicable', 'scope': 'exact contracted assets',
            'evidence_paths': [str(contract_path)], 'notes': 'Canonical check_files verifies actual bytes; acceptance is gated on its terminal receipt.'}
    record['assets_integrity']['environment'] = {'status': 'complete', 'scope': 'contracted runtime and environment lockfile',
        'evidence_paths': [contract['environment']['lockfile_path']], 'notes': 'No environment mutation performed by archiver.'}
    record['claims'] = [{'text': 'Bounded component execution and complete registered outputs verified.' if not errors else 'Run preserved; component completeness not accepted.',
        'evidence_type': 'actual_execution', 'evidence_paths': [str(supervision_receipt)] + evidence,
        'limitations': 'Component-only new experiment; no full-paper reproduction or paper-metric equivalence.'}]
    record['differences'] = [{'category': 'scope', 'description': 'New component implementation with vector endpoints; paper numerical equivalence is not asserted.',
        'effect_on_comparison': 'not_comparable', 'evidence_type': 'contract', 'evidence_paths': [str(contract_path)]}]
    record['incomplete_reasons'] = [{**e, 'affected_targets': [t['target_id'] for t in contract['targets']], 'evidence_paths': [str(supervision_receipt)], 'next_action': 'Review preserved source receipts; never silently rerun or promote scope.'} for e in errors]
    record['review'].update(status='pending', reviewer='mm_archive terminal canonical audit', reviewed_at=now,
        checks=[{'item': key, 'passed': not errors, 'reason': 'Actual terminal provenance and byte-bound artifacts; final canonical validation follows.'} for key in ('identity', 'assets', 'contract', 'execution', 'comparison', 'claims', 'preservation')],
        evidence_paths=[str(contract_path), str(supervision_receipt)], notes='Accepted only after canonical archive validation and local completeness checks.')
    record['archive_provenance'] = {'archiver_sha256': sha(__file__), 'validator_sha256': sha(validator), 'template_sha256': sha(template),
        'small_file_limit_bytes': small_file_limit_bytes, 'completion_spec': completion_spec, 'original_sources_modified': False}
    module_spec = importlib.util.spec_from_file_location('mm_archive_canonical', validator)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    def validate():
        return module.validate(contract, record, contract_bytes=raw, check_files=True,
            contract_directory=contract_path.parent, result_directory=output)
    preflight = validate()
    write(output / 'pre_acceptance_validation.json', preflight)
    acceptable = not errors and preflight.get('valid') is True and not preflight.get('files_skipped')
    record['review']['status'] = 'accepted' if acceptable else 'needs_revision'
    if not acceptable and not errors:
        record['lifecycle_status'] = 'partial'
        record['incomplete_reasons'] = [{'code': 'canonical_acceptance_not_met', 'description': 'Canonical validation errors or unchecked remote references remain.',
            'affected_targets': [t['target_id'] for t in contract['targets']], 'evidence_paths': [str(output / 'pre_acceptance_validation.json')], 'next_action': 'Resolve provenance; retain this record.'}]
    if not acceptable:
        for assessment in record['assets_integrity'].values():
            if assessment['status'] == 'complete': assessment['status'] = 'unknown'
        for check in record['review']['checks']:
            check.update(passed=False, reason='Terminal canonical or completeness requirements remain unresolved; inspect archive receipts.')
        record['claims'][0]['text'] = 'Existing run preserved; terminal component acceptance requirements remain unresolved.'
    terminal = validate()
    require(not acceptable or terminal.get('valid_for_acceptance') is True, 'terminal accepted record failed canonical validation')
    write(output / 'result-record.json', record)
    write(output / 'terminal_validation.json', terminal)
    result = {'status': 'accepted' if terminal.get('valid_for_acceptance') and acceptable else 'incomplete',
        'record_path': str(output / 'result-record.json'), 'record_sha256': sha(output / 'result-record.json'),
        'validation_path': str(output / 'terminal_validation.json'), 'validation_sha256': sha(output / 'terminal_validation.json'),
        'contract_sha256': hashlib.sha256(raw).hexdigest(), 'scientific_scope': 'component_only'}
    write(output / 'archive_receipt.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', required=True)
    p.add_argument('--supervision-dir', required=True)
    p.add_argument('--artifact-root', action='append', required=True)
    p.add_argument('--standard-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--completion-spec')
    args = p.parse_args()
    result = archive(args.contract, args.supervision_dir, args.artifact_root, args.standard_root, args.output,
                     completion_spec=read(args.completion_spec) if args.completion_spec else None)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['status'] == 'accepted' else 2


if __name__ == '__main__':
    raise SystemExit(main())

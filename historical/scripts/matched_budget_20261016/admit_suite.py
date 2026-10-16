"""Record parent release after checking physical review, CPU and cache evidence."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from execute_suite import ROOT, load, now, sha, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = ROOT / 'reports/benchmark_clarity_20261016'
    review = load(report / 'implementation_review.json')
    if not review.get('implementation_gate_passed') or review.get('blocking_findings'):
        raise ValueError('Independent implementation review not closed')
    for name, expected in review['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Reviewed implementation changed: ' + name)
    tests = load(report / 'parent_cpu_tests.json')
    if tests['exit_code'] != 0 or tests['tests_run'] != 64 or tests['result'] != 'OK':
        raise ValueError('Independent parent CPU tests not complete')
    for key, directory in (('source_sha256', HERE), ('test_sha256', ROOT / 'tests')):
        for name, expected in tests[key].items():
            if sha(directory / name) != expected:
                raise ValueError('CPU-tested source changed: ' + name)
    acceptance = load(report / 'protocol_acceptance.json')
    protocol_path = ROOT / acceptance['frozen_specification']['path']
    protocol_sha = acceptance['frozen_specification']['sha256']
    preservation = load(report / 'old_assets_pre_GPU_check.json')
    if not preservation['unchanged'] or preservation['files_checked'] != 1398:
        raise ValueError('Historical asset verification failed')
    if sha(report / 'locked_asset_snapshot.json') != preservation['snapshot_sha256']:
        raise ValueError('Historical snapshot changed')
    prepared = load(args.preparation / 'preparation-complete.json')
    receipt = prepared['receipt']
    if receipt['exit_code'] != 0 or not receipt['process_group_absent'] or receipt['termination_reason'] is not None:
        raise ValueError('CPU cache preparation did not terminate successfully')
    manifest_path = args.preparation / 'cache/manifest.json'
    protocol, manifest = verify_inputs(protocol_path, protocol_sha, manifest_path, prepared['manifest_sha256'])
    if 'torch' in sys.modules:
        raise RuntimeError('Parent admission unexpectedly imported torch')
    record = dict(created_utc=now(), release_issuer='parent agent under explicit user authorization',
        scope='Six supervised TRAIN-only GPU qualification canaries, then only if all pass the fixed36 trials',
        new_scope_authorization=protocol['authorization'], effective_deadline_at=protocol['timing']['effective_deadline_at'],
        protocol_path=str(protocol_path), protocol_sha256=protocol_sha,
        cache_manifest_path=str(manifest_path), cache_manifest_sha256=prepared['manifest_sha256'],
        source_review_path=str(report / 'implementation_review.json'), source_review_sha256=sha(report / 'implementation_review.json'),
        cpu_test_receipt_path=str(report / 'parent_cpu_tests.json'), cpu_test_receipt_sha256=sha(report / 'parent_cpu_tests.json'),
        old_assets_verified=preservation, fresh_generic_probe_path=str(report / 'resource_probe_pre_GPU.json'),
        generic_probe_is_not_runtime_qualification=True,
        resource_scope='Existing project Qwen1.5B and frozen ESM150M cache only; no PLM forward, new dependency, shared runtime or driver change',
        requires_each_job=['fresh task-specific resource probe', 'actual non-template frozen standard contract valid_for_execution',
                           'strict numerical policy', 'owned process-group supervision and terminal archive'],
        source_pins_verified=len(manifest['source_pins']), paused_or_restarted_old_contract=False)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'release': str(args.output), 'sha256': sha(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

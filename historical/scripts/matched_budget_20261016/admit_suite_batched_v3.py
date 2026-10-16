"""Parent admission of uniform batch2; no model loading or score selection."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from execute_suite import ROOT, load, now, sha, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-manifest', type=Path, required=True)
    parser.add_argument('--cache-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = ROOT / 'reports/benchmark_clarity_20261016'
    review = load(report / 'implementation_review_v3.json')
    if not review.get('implementation_gate_passed') or review.get('blocking_findings'):
        raise ValueError('Independent V3 implementation review not closed')
    for name, expected in review['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Reviewed V3 implementation changed: ' + name)
    tests = load(report / 'parent_cpu_tests_v3.json')
    if tests['exit_code'] != 0 or tests['tests_run'] != 126 or tests['result'] != 'OK':
        raise ValueError('Independent126 CPU tests not passed')
    for key, directory in (('source_sha256', Path(__file__).resolve().parent), ('test_sha256', ROOT / 'tests')):
        for name, expected in tests[key].items():
            if sha(directory / name) != expected:
                raise ValueError('CPU-tested V3 source changed: ' + name)
    if sha(ROOT / tests['complete_output_path']) != tests['log_sha256']:
        raise ValueError('CPU output log changed')
    preservation = load(report / 'old_assets_pre_GPU_v3_check.json')
    if not preservation['unchanged'] or preservation['files_checked'] != 1398:
        raise ValueError('Historical assets not intact')
    if sha(report / 'locked_asset_snapshot.json') != preservation['snapshot_sha256']:
        raise ValueError('Historical snapshot changed')
    protocol_path = ROOT / 'configs/matched_budget_20261016.frozen.json'
    protocol_sha = '9940d77d82e3ff23a085aa026219a4f4ee4e7ecd18bc1afe6952646a78e8b208'
    protocol, manifest = verify_inputs(protocol_path, protocol_sha, args.cache_manifest, args.cache_manifest_sha256)
    addendum = ROOT / 'reports/benchmark_clarity_20261016/engineering_addendum_v3.json'
    if manifest['source_pins'].get(addendum.relative_to(ROOT).as_posix()) != sha(addendum):
        raise ValueError('Uniform batching addendum not pinned')
    handoff = load(report / 'V1_boundary_handoff.json')
    old_suite = ROOT / 'results/matched_budget_20261016/suite_20261016_1426'
    old_state = load(old_suite / 'suite_state.json')
    complete = [j for j in old_state['jobs'] if j['case']['phase'] == 'trial' and j['complete']]
    if handoff['completed_trials'] != 2 or len(complete) != 2:
        raise ValueError('Expected exactly two complete source trainings')
    if old_state['status'] != 'failed_or_partial' or old_state['exception']['message'] != 'Supervisor stop requested before spawn':
        raise ValueError('V1 did not stop at its documented boundary')
    if 'torch' in sys.modules:
        raise RuntimeError('Parent admission unexpectedly loaded torch')
    record = dict(created_utc=now(), release_issuer='parent under explicit user authorization',
        scope='Six TRAIN-only batch1/batch2 all-four-row qualification canaries, then only if all pass uniform36 batch2 evaluations',
        authorization=protocol['authorization'], scientific_protocol_changed=False,
        protocol_path=str(protocol_path), protocol_sha256=protocol_sha,
        cache_manifest_path=str(args.cache_manifest), cache_manifest_sha256=args.cache_manifest_sha256,
        engineering_addendum_sha256=sha(addendum), implementation_review_sha256=sha(report / 'implementation_review_v3.json'),
        cpu_receipt_sha256=sha(report / 'parent_cpu_tests_v3.json'), old_assets_verified=preservation,
        training_source_suite=str(old_suite), complete_training_sources=2,
        intended_new_training_cases=34, intended_reused_training_inference_cases=2,
        current_V1_outputs_retained=True, casewise_best_version_selection=False,
        resource_scope='Existing Qwen1.5B/frozen ESM150M cache; no new PLM forward, dependency, shared runtime or driver changes',
        required_job_gates=['fresh task-specific resource probe', 'actual frozen standard contract valid_for_execution',
                            'strict common numerical policy', 'owned process-group bounds and terminal archive'],
        generic_probe_path=str(report / 'resource_probe_pre_GPU_v3.json'), generic_probe_does_not_establish_qualification=True,
        effective_deadline_at=protocol['timing']['effective_deadline_at'], window_reset=False)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'release': str(args.output), 'sha256': sha(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

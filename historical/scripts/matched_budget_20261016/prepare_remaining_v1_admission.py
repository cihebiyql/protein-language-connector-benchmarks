"""Reuse immutable messages for the exact original V1 continuation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from execute_suite import ROOT, load, now, sha, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    parent_sha = 'df857d026f99c0b44933fe6af929219edaa731a5bd0ed51d65ac5aed62d5029f'
    protocol_path = ROOT / 'configs/matched_budget_20261016.frozen.json'
    protocol_sha = '9940d77d82e3ff23a085aa026219a4f4ee4e7ecd18bc1afe6952646a78e8b208'
    _, manifest = verify_inputs(protocol_path, protocol_sha, args.parent_manifest, parent_sha)
    new_sources = ('scripts/matched_budget_20261016/execute_remaining_v1.py',
                   'scripts/matched_budget_20261016/summarize_remaining_v1.py',
                   'scripts/matched_budget_20261016/execute_suite.py',
                   'scripts/matched_budget_20261016/execute_suite_batched.py',
                   'scripts/matched_budget_20261016/summarize.py',
                   'reports/benchmark_clarity_20261016/engineering_final_decision_v1.json')
    for name in new_sources:
        manifest['source_pins'][name] = sha(ROOT / name)
    manifest['inference_implementation_admission'] = dict(
        admitted_utc=now(), version='original_V1_unchanged_batch1_all36', generation_batch_size=1,
        parent_manifest_path=args.parent_manifest.resolve().relative_to(ROOT).as_posix(),
        parent_manifest_sha256=parent_sha, scientific_protocol_sha256=protocol_sha,
        message_tensors_and_initializations_recomputed=False,
        source_training_or_test_criteria_changed=False,
        qualification='Exact original six TRAIN qualifications may be reused only with current environment, source, model, cache and policy checks; fresh per-job gates still required',
        effective_deadline_at='2026-10-16T23:20:00+08:00')
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
    verify_inputs(protocol_path, protocol_sha, args.output, sha(args.output))
    print(json.dumps({'manifest': str(args.output), 'sha256': sha(args.output),
                      'new_source_pins': {name: manifest['source_pins'][name] for name in new_sources}}))


if __name__ == '__main__':
    main()

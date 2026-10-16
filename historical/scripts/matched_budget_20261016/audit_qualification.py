"""Read actual TRAIN canary archives; record qualification without scientific scores."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from execute_suite import canary_matrix, case_id, load, now, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = []
    for case in canary_matrix(load(args.protocol)):
        job = args.suite / case_id(case)
        result = load(job / 'artifacts/result.json')
        exit_receipt = load(job / 'execution-receipt.json')
        acceptance = load(job / 'result-validation.json')
        numerical = result['numerical_policy']
        if (result['status'] != 'passed' or result['sample_split'] != 'train'
                or result['test_examples_evaluated'] != 0 or result['scientific_trial_count'] != 0
                or result['optimizer_steps'] != 1 or not acceptance['valid_for_acceptance']
                or exit_receipt['exit_code'] != 0 or not exit_receipt['performed']
                or not exit_receipt['process_group_absent'] or exit_receipt['termination_reason'] is not None
                or numerical['base_parameter_dtypes'] != ['torch.bfloat16']
                or numerical['reader_parameter_dtypes'] != ['torch.float32']
                or numerical['deterministic_algorithms'] is not True
                or numerical['deterministic_warn_only'] is not False):
            raise ValueError('TRAIN runtime qualification incomplete: ' + job.name)
        records.append(dict(case=case, result_path=str(job / 'artifacts/result.json'),
                            result_sha256=sha(job / 'artifacts/result.json'),
                            receipt_sha256=sha(job / 'execution-receipt.json'),
                            standard_acceptance_sha256=sha(job / 'result-validation.json'),
                            gpu_peak_allocated_bytes=result['gpu_peak_allocated_bytes'],
                            elapsed_seconds=exit_receipt['elapsed_seconds'],
                            test_examples_evaluated=0, optimizer_steps=1, process_group_absent=True))
    output = dict(audited_utc=now(), status='all_six_TRAIN_runtime_qualifications_passed',
                  protocol_sha256=sha(args.protocol), canaries=records,
                  formal_training_trials_counted=0, scientific_performance_claim=False,
                  meaning='Actual runtime forward/backward, generation/metric serialization and owned-process cleanup only')
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'passed_canaries': len(records), 'scientific_trials_counted': 0}))


if __name__ == '__main__':
    main()

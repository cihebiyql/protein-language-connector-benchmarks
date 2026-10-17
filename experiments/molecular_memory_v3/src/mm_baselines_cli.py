"""Resolve a fixed suite recipe; print exact binding or execute one reviewed scenario."""
import argparse
import json
from pathlib import Path

from . import mm_baselines as baseline
from . import mm_data


def resolve_request(path, project_root, scenario):
    recipe = mm_data.strict_loads(Path(path).read_text(encoding='utf-8'))
    if (set(recipe) != {'schema', 'view_manifest_relative', 'scenario_requests'}
            or recipe['schema'] != 'mm_numeric_baseline_suite.v1'
            or set(recipe['scenario_requests']) != set(mm_data.SCENARIOS)):
        raise ValueError('fixed two-scenario baseline suite required')
    for entry in recipe['scenario_requests'].values():
        if (set(entry) != {'methods', 'output_relative'}
                or entry['methods'] != ['train_mean', 'train_median']):
            raise ValueError('suite registers only TRAIN mean and median')
    if scenario not in mm_data.SCENARIOS:
        raise ValueError('unregistered scenario')
    # This is a pre-contract resolution, not execution authorization. The exact
    # resulting manifest SHA/request/code/output must be bound by the contract.
    manifest = baseline._relative(project_root, recipe['view_manifest_relative'])
    entry = recipe['scenario_requests'][scenario]
    return baseline.make_request(scenario, recipe['view_manifest_relative'],
                                 mm_data.file_sha(manifest), entry['output_relative'], entry['methods'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', required=True)
    p.add_argument('--project-root', required=True)
    p.add_argument('--scenario', required=True, choices=mm_data.SCENARIOS)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--print-binding', action='store_true')
    mode.add_argument('--execute', action='store_true')
    p.add_argument('--contract')
    p.add_argument('--validator')
    p.add_argument('--validator-sha256')
    args = p.parse_args()
    request = resolve_request(args.request, args.project_root, args.scenario)
    if args.print_binding:
        plan = baseline.run_baselines(request, args.project_root)
        result = {'resolved_request': request, 'baseline_run_binding': baseline.execution_binding(request, args.project_root),
                  'suite_sha256': mm_data.file_sha(args.request), 'cli_sha256': mm_data.file_sha(__file__), 'plan': plan}
    else:
        result = baseline.run_baselines(request, args.project_root, execute=args.execute,
            contract_path=args.contract, validator_path=args.validator, validator_sha256=args.validator_sha256)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

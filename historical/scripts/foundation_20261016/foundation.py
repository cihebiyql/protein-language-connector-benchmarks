"""One entry point for the baseline foundation's tests and evidence index.

This index never promotes a smoke result, prior training, or an unavailable gold
set to a new full benchmark result. GPU execution remains in its bounded suite.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/foundation_20261016'
PYTHON_WSL = 'LOCAL_HOME/.venvs/peptide-omnipanel-plm5080/bin/python'

def load(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf8'))
    except json.JSONDecodeError:
        return {'status': 'file_being_written_or_invalid', 'path': str(path)}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temporary.replace(path)

def build_index():
    replay_path = OUT / 'replay_predictors/summary.json'
    rt_path = OUT / 'rt_target_diagnostic/result-record.json'
    if not rt_path.exists():
        candidates = list((OUT / 'rt_target_diagnostic').glob('*result*.json'))
        rt_path = candidates[0] if candidates else rt_path
    replay = load(replay_path)
    rt = load(rt_path)
    replay_verified = bool(replay and replay.get('status') == 'passed'
                           and replay.get('actual_replayed_checkpoints') == 18
                           and replay.get('deterministic_control_fits') == 6
                           and replay.get('original_input_pins_unchanged')
                           and replay.get('exit_code') == 0)
    rt_validation = load(OUT / 'rt_target_diagnostic/result_validation.json')
    rt_verified = bool(rt and rt_validation and rt_validation.get('valid_for_acceptance') is True)
    assets = ROOT / 'sources/continuation_20261016/transfer_verification.json'
    extra_assets = ROOT / 'sources/continuation_20261016/manifest_additional_live.json'
    methods = {
        'BASE-00': {'name': 'train mean / composition and chemistry ridge / kNN',
                    'evidence_kind': 'fresh_control_refit', 'result': str(replay_path),
                    'status': 'verified_replay_and_refit' if replay_verified else 'not_verified'},
        'BASE-01': {'name': 'target-conditioned ESM residue attention predictor',
                    'evidence_kind': 'prior_checkpoint_independent_replay', 'result': str(replay_path),
                    'status': 'verified_prior_checkpoint_replay' if replay_verified else 'not_verified'},
        'BASE-02': {'name': 'sequence-text discriminative alignment',
                    'evidence_kind': 'prior_checkpoint_independent_replay', 'result': str(replay_path),
                    'status': 'verified_prior_checkpoint_replay' if replay_verified else 'not_verified'},
        'BASE-03': {'name': 'frozen predictor structured estimates to same frozen Qwen', 'evidence_kind': 'new_generation_inference'},
        'BASE-04': {'name': 'ordinary input MLP connector', 'method_key': 'input_mlp', 'evidence_kind': 'new_connector_training'},
        'BASE-05': {'name': 'Q-Former-style learned query connector', 'method_key': 'qformer_style', 'evidence_kind': 'new_connector_training'},
        'BASE-06': {'name': 'ordinary internal cross-attention', 'method_key': 'internal_cross_attention', 'evidence_kind': 'new_connector_training'},
        'CONSTANT': {'name': 'separately trained constant memory', 'method_key': 'constant', 'evidence_kind': 'new_control_training'}
    }
    new_results = []
    execution_suites = []
    for suite in sorted(OUT.glob('gpu_suite*')):
        if not suite.exists():
            continue
        state = load(suite / 'suite_state.json') or {}
        if state:
            execution_suites.append({'directory': str(suite), 'state_path': str(suite / 'suite_state.json'),
                'status': state.get('status'), 'first_substantive_started_at': state.get('first_substantive_started_at'),
                'terminal_jobs': [{'job': job.get('job'), 'phase': job.get('phase'),
                                  'complete': job.get('complete'),
                                  'exit_code': job.get('receipt', {}).get('exit_code')}
                                 for job in state.get('jobs', [])]})
        for path in sorted(suite.rglob('result.json')):
            value = load(path)
            if not value or value.get('status') != 'complete':
                continue
            registration = load(path.parent / 'run_preregistration.json') or {}
            method = value.get('method') or registration.get('method') or registration.get('configuration', {}).get('method')
            matched = next((key for key, entry in methods.items()
                            if entry.get('method_key') == method), None)
            if method == 'BASE03':
                matched = 'BASE-03'
            row = {'method': method, 'result_path': str(path), 'result_sha256': digest(path),
                   'scope': value.get('scope', registration.get('scope')), 'subset_scope': value.get('subset_scope'),
                   'scenario': value.get('scenario') or registration.get('scenario') or registration.get('configuration', {}).get('scenario'),
                   'formal_multitask_connector_result': value.get('formal_multitask_connector_result') is True,
                   'registration': str(path.parent / 'run_preregistration.json')}
            new_results.append(row)
            if matched:
                methods[matched].setdefault('actual_results', []).append(row)
                methods[matched]['status'] = 'execution_result_available_review_separate'
    for entry in methods.values():
        entry.setdefault('status', 'component_implemented_not_yet_verified_as_real_execution')

    information_results = []
    for path in sorted((OUT / 'information_diagnostics/v3_all8').glob('*/artifacts/result.json')):
        actual = load(path) or {}
        job = path.parent.parent
        acceptance = load(job / 'result-validation.json') or {}
        receipt = load(job / 'execution-receipt.json') or {}
        if (actual.get('status') != 'complete' or acceptance.get('valid_for_acceptance') is not True
                or receipt.get('exit_code') != 0 or receipt.get('process_group_absent') is not True):
            continue
        information_results.append({'method': actual.get('method'), 'scenario': actual.get('scenario'),
            'result_path': str(path), 'result_sha256': digest(path),
            'result_validation': str(job / 'result-validation.json'),
            'has_C1': 'C1' in actual, 'has_C2': 'C2' in actual,
            'scientific_claim_review_separate': True})

    panels = {
        'A1': {'status': 'broad_reference_target_panel_ready', 'same_condition_pair_gold_n': 0,
               'known_target_split_n': {'train': 2062, 'validation': 147, 'test': 152},
               'double_new_split_n': {'train': 2063, 'validation': 431, 'test': 442},
               'same_condition_affinity_claim': False},
        'A2_target_diagnostic': {'status': 'verified_target_only_diagnostic' if rt_verified else 'not_verified',
                                 'source': 'PXD064311 March, measured minutes', 'result': str(rt_path),
                                 'original_cross_condition_gate_released': False},
        'A2_original_cross_condition': {'status': 'blocked',
            'reason': 'Exact source snapshot and lossy D/E chemical identities unresolved; iRT is not minutes'},
        'B1': {'status': 'locked_predictor_checkpoint_replayed' if replay_verified else 'not_verified',
               'annotation_rows': {'train': 378, 'validation': 134, 'test': 76},
               'test_function_classes': 2, 'new_generation_connector_claim': False},
        'B2': {'status': 'blocked', 'reviewed_qa_gold_n': 0, 'evidence_registry_is_gold': False},
        'C1': {'status': 'correct_off_enabled_in_new_generation_runs',
               'same_target_independent_group_donors': 0,
               'donor_protocol': 'pre-execution revision 2; source-connected groups held independent',
               'mismatched_score_P_H': None, 'constant_control': 'see CONSTANT method'},
        'C2': {'status': 'accepted_execution_available_scientific_review_separate'
               if any(row['has_C2'] for row in information_results) else 'component_tests_only',
               'expected_method_scenario_jobs': 8, 'accepted_execution_jobs': len(information_results),
               'evaluation': 'Frozen A1 pKD linear readout; prompt-only states; not a total mutual-information estimate',
               'not_executed_in_this_stage': ['random-label control tasks', 'MLP probe capacity sweep',
                                               'memory-slot deletion curves', 'biological counterfactual or residue evidence validation']},
        'C3': {'status': 'blocked', 'reason': 'No independently adjudicated atomic-fact gold'},
        'D': {'status': 'descriptive_reliability_results_available' if replay_verified else 'not_verified',
               'validation_reused_after_checkpoint_selection': True, 'formal_coverage_guarantee': False},
        'COST': {'status': 'per_real_run_receipts', 'parameter_count_is_total_system_cost': False}
    }
    originals = {name: {'status': 'not_claimed_reproduced',
                        'reason': 'Author system runtime, exact data and applicable weights require separate admission'}
                 for name in ['OPUS-PLLM', 'ProtT3', 'ProtNote', 'DeepLC-v3.1.13', 'PepInter']}
    original_readiness = load(ROOT / 'sources/continuation_20261016/original_system_readiness.json') or {}
    for name, entry in originals.items():
        entry['individual_readiness'] = original_readiness.get('systems', {}).get(name)
    test_receipts = sorted(OUT.glob('tests_*/result.json'))
    value = {'benchmark': 'PepBridge-Bench foundation continuation 2026-10-16',
             'updated_utc': datetime.now(timezone.utc).isoformat(),
             'project_root': str(ROOT), 'old_results_overwritten': False,
             'new_results': new_results, 'information_results': information_results,
             'execution_suites': execution_suites,
             'latest_unified_test_receipt': {'path': str(test_receipts[-1]),
                                             'sha256': digest(test_receipts[-1]),
                                             'result': load(test_receipts[-1])} if test_receipts else None,
             'methods': methods, 'panels': panels,
             'published_systems_separate': originals,
             'asset_verification': load(assets), 'additional_full_assets': load(extra_assets),
             'additional_full_asset_verification': load(ROOT / 'sources/continuation_20261016/additional_full_verification.json'),
             'protnote_unpack_verification': load(ROOT / 'sources/continuation_20261016/unpack_protnote_verification.json'),
             'workbench_status_receipt': str(ROOT / 'sources/continuation_20261016/workbench_current_status_coverage_all60.json'),
             'predictor_replay_path': str(replay_path),
             'scope_notice': 'A1/B1 existing predictor results, new A1 generation foundation and RT target-only diagnostics remain separate; no unified multitask or full-paper reproduction claim'}
    write_json(OUT / 'foundation_index.json', value)
    return value

def run_tests():
    """Existing NumPy/openpyxl tests on Windows; torch-specific tests in WSL."""
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    directory = OUT / ('tests_' + stamp)
    directory.mkdir(parents=True, exist_ok=False)
    jobs = [('windows_existing', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_v02*.py', '-v'])]
    if sys.platform == 'win32':
        import shlex
        wsl_root = '/mnt/' + ROOT.drive[0].lower() + ROOT.as_posix()[2:]
        for name, pattern in [('wsl_existing_torch', 'test_v02_panel*.py'), ('wsl_foundation', 'test_foundation*.py')]:
            command = 'cd ' + shlex.quote(wsl_root) + ' && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=\'\' timeout 300 ' + shlex.quote(PYTHON_WSL) + ' -m unittest discover -s tests -p ' + shlex.quote(pattern) + ' -v'
            jobs.append((name, ['wsl.exe', '-d', 'Ubuntu-22.04', '--', 'bash', '-lc', command]))
    else:
        jobs.append(('local_foundation', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_foundation*.py', '-v']))
    records = []
    for name, command in jobs:
        started = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, text=True, encoding='utf8', errors='replace',
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=330)
        log = directory / (name + '.log')
        log.write_text(result.stdout, encoding='utf8')
        record = {'name': name, 'command': command, 'exit_code': result.returncode,
                  'elapsed_seconds': time.perf_counter() - started, 'log': str(log),
                  'log_sha256': digest(log), 'passed': result.returncode == 0}
        count = re.search(r'Ran (\d+) tests?', result.stdout)
        record['tests_discovered'] = int(count.group(1)) if count else None
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    write_json(directory / 'result.json', {'status': 'passed' if all(r['passed'] for r in records) else 'failed', 'jobs': records})
    return all(r['passed'] for r in records)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', choices=['status', 'tests', 'gpu'], default='status')
    args = parser.parse_args()
    if args.step == 'tests':
        passed = run_tests()
        build_index()
        return 0 if passed else 1
    if args.step == 'gpu':
        print('GPU suite is explicitly bounded and must freeze its own run contracts before work.', flush=True)
        controller = Path(__file__).with_name('execute_gpu_suite.py')
        if not controller.exists():
            raise SystemExit('GPU suite controller not yet available')
        if sys.platform == 'win32':
            linux_path = '/mnt/' + controller.drive[0].lower() + controller.as_posix()[2:]
            return subprocess.call(['wsl.exe', '-d', 'Ubuntu-22.04', '--', PYTHON_WSL, linux_path], cwd=ROOT)
        return subprocess.call([sys.executable, str(controller)], cwd=ROOT)
    index = build_index()
    print(json.dumps({'index': str(OUT / 'foundation_index.json'),
                      'methods': {key: value['status'] for key, value in index['methods'].items()},
                      'panels': {key: value['status'] for key, value in index['panels'].items()}}, ensure_ascii=False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

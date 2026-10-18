"""Recovery orchestration must publish only after successful frozen rescoring."""
import contextlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('recovery_finalize_contract_test', ROOT / 'resources/recovery_finalize_v1.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

class FinalizerTests(unittest.TestCase):
    def setup_fixture(self, temporary):
        scope = Path(temporary) / 'scope'
        for name in ('private/recovery_finalize_v1', 'runs/formal_v1', 'runs/recovery_retry_v1',
                     'private/node1_publication/formal_v1_node1_terminal_v1', 'configs', 'resources', 'docs'):
            (scope / name).mkdir(parents=True, exist_ok=True)
        def write(name, value): (scope / name).write_text(json.dumps(value))
        write('runs/formal_v1/complete.json', {})
        write('runs/formal_v1/analysis_exit.json', {'exit_code': 0})
        write('runs/formal_v1/queue_state.json', {'counts': {'accepted': 44, 'failed': 1}})
        write('runs/recovery_retry_v1/recovery_status.json', {'status': 'accepted'})
        write('private/node1_publication/formal_v1_node1_terminal_v1/launch.json', {'identity': {'pid': 100}})
        write('private/recovery_v1_old_publisher_retired.json', {})
        write('configs/queue_formal_v1.json', {'paths': {'python': 'test-python', 'data_root': str(scope / 'data')}})
        write('private/finalizer_config.json', {})
        (scope / 'resources/audited.py').write_text('value = 1\n')
        config = {'source_pins': {str(scope / 'resources/audited.py'): runner.mt_case.sha(scope / 'resources/audited.py')},
                  'wait_timeout_seconds': 5, 'publisher_config': str(scope / 'private/publisher.json')}
        return scope, config

    def exercise(self, *, analysis_code=0, old_alive=False):
        with tempfile.TemporaryDirectory() as temporary:
            scope, config = self.setup_fixture(temporary)
            def build(*args):
                (scope / 'runs/reconciled_recovery_v1/provenance').mkdir(parents=True)
                return {'retry_accepted': True}
            def analyze(command, **kwargs):
                self.assertIn('src.mt_analyze', command)
                self.assertIn(str(scope / 'runs/reconciled_recovery_v1'), command)
                if analysis_code == 0:
                    output = scope / 'analysis/recovery_reconciled_v1'
                    output.mkdir(parents=True)
                    (output / 'analysis.json').write_text(json.dumps({'schema': 'multitask-independent-analysis.v1',
                        'expected_cases': 45, 'paired_comparisons': [], 'arm_summaries': []}))
                return types.SimpleNamespace(returncode=analysis_code)
            with patch.object(runner, 'SCOPE', scope), patch.object(runner, 'STATE', scope / 'private/recovery_finalize_v1'), \
                 patch.object(runner, 'verify_config', return_value=config), \
                 patch.object(runner.socket, 'gethostname', return_value='node1'), \
                 patch.object(runner.mt_queue, 'lease', return_value=contextlib.nullcontext()), \
                 patch.object(runner.supervision, 'process_identity', return_value={'pid': 200}), \
                 patch.object(runner.finish_and_export, 'identity_alive', return_value=old_alive), \
                 patch.object(runner.reconcile_recovery_v1, 'build_view', side_effect=build) as merge, \
                 patch.object(runner.subprocess, 'run', side_effect=analyze) as rescore, \
                 patch.object(runner.node1_publish, 'run', return_value={'commit': 'c' * 40}) as publish:
                if analysis_code or old_alive:
                    with self.assertRaises(RuntimeError): runner.run(scope / 'private/finalizer_config.json')
                    publish.assert_not_called()
                    if old_alive:
                        merge.assert_not_called()
                        rescore.assert_not_called()
                else:
                    result = runner.run(scope / 'private/finalizer_config.json')
                    self.assertEqual(result['commit'], 'c' * 40)
                    publish.assert_called_once_with(config['publisher_config'])
                    self.assertEqual(json.loads((scope / 'runs/reconciled_recovery_v1/analysis_exit.json').read_text())['exit_code'], 0)
                    self.assertTrue((scope / 'docs/RECOVERY_M3_ANALYSIS_v1.md').is_file())
                    summary = json.loads((scope / 'runs/reconciled_recovery_v1/provenance/M3_STABILITY/summary.json').read_text(encoding='utf-8'))
                    self.assertIsNone(summary['overall_M3_vs_M0']['supports_universal_stable_advantage'])

    def test_successful_rescore_reports_missing_seeds_and_then_publishes(self): self.exercise()
    def test_failed_rescore_never_publishes(self): self.exercise(analysis_code=2)
    def test_live_previous_publisher_prevents_reconciliation(self): self.exercise(old_alive=True)

if __name__ == '__main__': unittest.main()

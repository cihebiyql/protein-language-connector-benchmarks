"""Temporary CPU archival fixtures using the installed canonical validator."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import unittest

from src import mm_archive as a

STANDARD = Path(os.environ['MM_STANDARD_ROOT']) if os.environ.get('MM_STANDARD_ROOT') else (
    Path(__file__).resolve().parents[1] / 'standard')


class ArchiveTests(unittest.TestCase):
    def fixture(self, root):
        standard = root / 'standard'
        standard.mkdir()
        for name in ('result-record.template.json', 'validate_reproduction.py'):
            shutil.copyfile(STANDARD / name, standard / name)
        contract = a.read(STANDARD / 'task-contract.template.json')
        now = datetime.now(timezone.utc)
        first, frozen = now - timedelta(seconds=30), now - timedelta(seconds=20)
        started, ended, deadline = now - timedelta(seconds=10), now - timedelta(seconds=9), now + timedelta(minutes=5)
        artifact = root / 'artifacts'
        artifact.mkdir()
        a.write(artifact / 'metrics.json', {'status': 'complete', 'rows': 2, 'scores': {'MAE': 1, 'Spearman': None}})
        inputs = root / 'inputs'
        inputs.mkdir()
        for name in ('data.json', 'code.py', 'paper.pdf', 'environment.json'):
            (inputs / name).write_bytes(b'fixture bytes\n')
        contract.update(is_template=False, task_id='fixture', run_id='run', contract_status='frozen', frozen_at=frozen.isoformat())
        contract['identity'].update(title='Fixture provenance', doi='10.0000/fixture', publication_status='published',
            pdf={'path': str(inputs / 'paper.pdf'), 'sha256': a.sha(inputs / 'paper.pdf')})
        contract['execution_scope'].update(intended_level='component', scope_coverage='component_only', objective='Fixture component',
            authorization_reference='temporary CPU unit test', safety_assessment='allowed', restriction_check={'prior_platform_restriction': False})
        contract['timing'].update(first_substantive_started_at=first.isoformat(), original_deadline_at=deadline.isoformat(), effective_deadline_at=deadline.isoformat(),
            wall_clock_cap_hours=10, additional_deadlines_at=[], earliest_associated_preparation_at=first.isoformat(), prior_window_expired=False, retry_resets_clock=False)
        datafile, codefile = inputs / 'data.json', inputs / 'code.py'
        contract['assets'].update(datasets=[{'name': 'fixture', 'path': str(datafile), 'sha256': a.sha(datafile), 'manifest_sha256': a.sha(datafile), 'integrity': 'complete'}],
            code=[{'path': str(codefile), 'sha256': a.sha(codefile), 'archive_sha256': a.sha(codefile), 'integrity': 'complete'}],
            models=[{'integrity': 'not_applicable', 'notes': 'No model fixture'}])
        contract['targets'] = [{'target_id': 'vector', 'required': True, 'paper_endpoint': 'new component', 'metric': 'MAE_Spearman_vector',
            'metric_definition': 'two unchanged metrics', 'statistical_unit': 'row', 'denominator': 2, 'unit': 'mixed', 'paper_value': None,
            'tolerance': {'kind': 'not_comparable', 'reason': 'New component'}}]
        environment = inputs / 'environment.json'
        contract['environment'].update(isolation='fixture', environment_path=str(inputs), lockfile_path=str(environment), lockfile_sha256=a.sha(environment), shared_environment_modified=False)
        supervision = root / 'supervision'
        supervision.mkdir()
        probe = {'hostname': socket.gethostname(), 'readiness': 'ready'}
        a.write(supervision / 'startup_resource_probe.json', probe)
        contract['resources'].update(expected_hostname=socket.gethostname(), cpu_threads=1, memory_limit_bytes=1024**3,
            resource_probe_path=str(supervision / 'startup_resource_probe.json'), resource_probe_at=first.isoformat(), start_gate='ready', budget_basis='fixture', no_preemption=True)
        for name in ('stdout.log', 'stderr.log'):
            (root / name).write_bytes(b'')
        contract['execution'].update(command=['fixture'], working_directory=str(root), seed_policy='none', seeds=[], timeout_seconds=30,
            process_control={'whole_tree_bounded': True}, stdout_path=str(root / 'stdout.log'), stderr_path=str(root / 'stderr.log'))
        contract['review'].update(status='accepted')
        contract_file = root / 'contract.json'
        a.write(contract_file, contract)
        digest = a.sha(contract_file)
        a.write(supervision / 'started.json', {'pid': 1234, 'pgid': 1234, 'started_at': started.isoformat(), 'contract_sha256': digest})
        a.write(supervision / 'receipt.json', {'status': 'succeeded', 'exit_code': 0, 'termination_reason': None, 'pid': 1234, 'pgid': 1234,
            'started_at': (started - timedelta(milliseconds=5)).isoformat(), 'ended_at': ended.isoformat(), 'elapsed_seconds': 1.005,
            'peak_group_rss_bytes': 20000000, 'remaining_group_members': [], 'handles_absent_verified_at': ended.isoformat(),
            'contract_sha256': digest, 'stdout_sha256': a.sha(root / 'stdout.log'), 'stderr_sha256': a.sha(root / 'stderr.log')})
        spec = {'artifact_root': str(artifact), 'assertions': [{'file': 'metrics.json', 'pointer': '/status', 'equals': 'complete'}],
                'target_counts': {'vector': {'file': 'metrics.json', 'pointer': '/rows'}}}
        return dict(contract_path=contract_file, supervision_dir=supervision, artifact_roots=[artifact], standard_root=standard, output=root / 'archive', completion_spec=spec)

    def change_json(self, path, function):
        value = a.read(path)
        function(value)
        path.write_text(json.dumps(value), encoding='utf-8')

    def test_complete_vector_component_is_accepted_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            before = {str(p): a.sha(p) for p in Path(directory).rglob('*') if p.is_file()}
            receipt = a.archive(**args)
            self.assertEqual(receipt['status'], 'accepted')
            record = a.read(receipt['record_path'])
            self.assertEqual(record['scope_coverage'], 'component_only')
            self.assertEqual(record['reproduction_level'], 'component')
            metric = record['metric_comparisons'][0]
            self.assertEqual((metric['metric'], metric['denominator']), ('MAE_Spearman_vector', 2))
            self.assertIsNone(metric['actual_value'])
            self.assertIsNone(metric['passed'])
            self.assertEqual(record['execution']['pid'], 1234)
            self.assertEqual(record['execution']['peak_group_rss_bytes'], 20000000)
            validation = a.read(receipt['validation_path'])
            self.assertTrue(validation['valid_for_acceptance'])
            self.assertFalse(validation['files_skipped'])
            for path, digest in before.items(): self.assertEqual(a.sha(path), digest)
            for artifact in record['artifacts']:
                self.assertEqual(a.sha(artifact['path']), artifact['sha256'])
            with self.assertRaises(FileExistsError): a.archive(**args)

    def test_incomplete_denominator_does_not_become_success(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            self.change_json(args['artifact_roots'][0] / 'metrics.json', lambda r: r.update(rows=1))
            receipt = a.archive(**args)
            record = a.read(receipt['record_path'])
            self.assertEqual(receipt['status'], 'incomplete')
            self.assertEqual(record['lifecycle_status'], 'failed')
            self.assertEqual(record['review']['status'], 'needs_revision')
            self.assertTrue(record['incomplete_reasons'])
            self.assertFalse(a.read(receipt['validation_path'])['valid_for_acceptance'])

    def test_exit_cleanup_contract_hash_or_log_tamper_cannot_pass(self):
        cases = [('receipt', lambda r: r.update(exit_code=2)), ('receipt', lambda r: r.update(remaining_group_members=[1234])),
                 ('receipt', lambda r: r.update(contract_sha256='0' * 64)), ('receipt', lambda r: r.update(peak_group_rss_bytes=2 * 1024**3)),
                 ('started', lambda r: r.update(pgid=999))]
        for file, change in cases:
            with self.subTest(file=file, change=change), tempfile.TemporaryDirectory() as directory:
                args = self.fixture(Path(directory))
                self.change_json(args['supervision_dir'] / (file + '.json'), change)
                receipt = a.archive(**args)
                self.assertEqual(receipt['status'], 'incomplete')
                self.assertEqual(a.read(receipt['record_path'])['review']['status'], 'needs_revision')
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            (Path(directory) / 'stdout.log').write_text('tampered')
            self.assertEqual(a.archive(**args)['status'], 'incomplete')

    def test_actual_asset_hash_mismatch_fails_canonical_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            (Path(directory) / 'inputs/code.py').write_text('tampered')
            receipt = a.archive(**args)
            self.assertEqual(receipt['status'], 'incomplete')
            validation = a.read(receipt['validation_path'])
            self.assertFalse(validation['valid_for_acceptance'])
            self.assertTrue(any('SHA256 mismatch' in e['message'] for e in validation['errors']))
            record = a.read(receipt['record_path'])
            self.assertEqual(record['assets_integrity']['code']['status'], 'unknown')

    def test_generic_exit_zero_without_completeness_spec_is_insufficient(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            args['completion_spec'] = None
            self.assertEqual(a.archive(**args)['status'], 'incomplete')

    def test_missing_target_count_or_escaping_assertion_fails(self):
        for mode in ('missing', 'escaping', 'boolean'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                args = self.fixture(Path(directory))
                if mode == 'missing': args['completion_spec']['target_counts'] = {}
                elif mode == 'escaping': args['completion_spec']['assertions'][0]['file'] = '../contract.json'
                else:
                    args['completion_spec']['assertions'][0].update(pointer='/rows', equals=True)
                self.assertEqual(a.archive(**args)['status'], 'incomplete')

    def test_archive_cannot_change_inventoried_artifact_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            args['output'] = args['artifact_roots'][0] / 'archive'
            with self.assertRaises(ValueError): a.archive(**args)


if __name__ == '__main__':
    unittest.main()

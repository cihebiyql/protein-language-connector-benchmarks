"""CPU safety checks: no live SSH, GPU work, credential use or Git publication."""
import hashlib
import importlib.util
import io
import json
import subprocess
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCOPE = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCOPE / 'resources' / (name + '.py'))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


s = module('sync_results')
f = module('finish_and_export')


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def put(self, root, name, payload):
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
        return path

    def export(self, files):
        root = self.root / 'source'; root.mkdir()
        entries = []
        for name, value in files.items():
            path = self.put(root, name, value)
            entries.append({'path': name, 'sha256': s.sha(path), 'bytes': path.stat().st_size})
        return root, {'scope_root': '/data1/user/project/experiments/' + s.SCOPE_ID,
            'files': entries, 'completion_trigger': 'controller_dead_partial'}

    def archive(self, names):
        path = self.root / 'archive.tar.gz'
        with tarfile.open(path, 'w:gz') as tar:
            for name, kind in names:
                member = tarfile.TarInfo(name)
                if kind == 'link': member.type = tarfile.SYMTYPE; member.linkname = '/outside'
                else: member.size = 3
                tar.addfile(member, io.BytesIO(b'abc') if kind != 'link' else None)
        return path

    def test_archive_rejects_traversal_absolute_links_windows_and_duplicates(self):
        for names in ([('../escape', 'file')], [('/absolute', 'file')], [('x', 'link')],
                      [('C:evil', 'file')], [('..\\escape', 'file')], [('same', 'file'), ('same', 'file')]):
            with self.subTest(names=names):
                with self.assertRaises(ValueError): s.safe_extract(self.archive(names), self.root / 'new')
                self.assertFalse((self.root / 'new').exists())

    def test_safe_extract_new_directory_and_manifest_integrity(self):
        root, manifest = self.export({'a.json': {'number': 1}})
        manifest_file = self.put(root, 'EXPORT_MANIFEST.json', manifest)
        tar = self.root / 'good.tar.gz'
        with tarfile.open(tar, 'w:gz') as archive:
            for p in root.iterdir(): archive.add(p, arcname=p.name)
        extracted = s.safe_extract(tar, self.root / 'new')
        self.assertEqual(s.verify_export(extracted, 'EXPORT_MANIFEST.json', s.sha(manifest_file)), manifest)
        self.put(extracted, 'unexpected.json', {})
        with self.assertRaises(ValueError): s.verify_export(extracted, 'EXPORT_MANIFEST.json', s.sha(manifest_file))
        with self.assertRaises(FileExistsError): s.safe_extract(tar, extracted)

    def test_metadata_normalization_preserves_scientific_types_and_drops_identities(self):
        original = {'/data1/private/project/a': {'metric': 0.12345678901234567, 'seed': 17, 'missing': None,
            'identity': {'pid': 123, 'pgid': 123, 'task': 'elm'}, 'hostname': 'private-host',
            'gpu_compute_processes': ['other-user PID'], 'remaining_group_members': [45, 46]}}
        result = s.normalize(original, {'/data1/private/project': '<PROJECT>'})
        item = result['<PROJECT>/a']
        self.assertEqual(item['metric'], original['/data1/private/project/a']['metric'])
        self.assertIsNone(item['missing']); self.assertIs(type(item['seed']), int)
        self.assertEqual(item['identity'], {'task': 'elm'})
        self.assertEqual(item['remaining_group_member_count'], 2)
        self.assertNotIn('other-user', json.dumps(result))

    def test_unknown_root_collision_and_credentials_fail_closed(self):
        for value, roots in [('/home/private/a', {}), ('hf_' + 'A' * 30, {}),
            ({'/data1/a': 1, '/data1/b': 2}, {'/data1/a': '<X>', '/data1/b': '<X>'})]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError): s.normalize(value, roots)

    def test_registered_scope_data_subdirectory_is_not_a_private_absolute_root(self):
        original = {'/data1/user/project/experiments/scope/data/meltome/train.jsonl': 'a' * 64}
        value = s.normalize(original, {'/data1/user/project/experiments/scope': '<REMOTE_SCOPE>'})
        self.assertEqual(value, {'<REMOTE_SCOPE>/data/meltome/train.jsonl': 'a' * 64})
        self.assertFalse(s.PRIVATE.search('<REMOTE_SCOPE>/nested/data/manifest.json'))
        self.assertTrue(s.PRIVATE.search('File "/data/unregistered-user/runtime/runpy.py"'))

    def test_actual_predictions_and_history_bytes_remain_exact(self):
        raw = b'{"sample_id":"x","prediction":1.2345678901234567}\r\n'
        history = b'[{"epoch":1,"loss":0.12345678901234567}]\n'
        root, manifest = self.export({'runs/formal/cases/x/attempt_001/artifacts/readout_predictions.jsonl': raw,
                                     'runs/formal/cases/x/attempt_001/artifacts/training_history.json': history})
        out = s.publish_snapshot(root, self.root / 'public', manifest, {'/data1/user': '<ROOT>'})
        for name, data in [('readout_predictions.jsonl', raw), ('training_history.json', history)]:
            self.assertEqual(next(out.rglob(name)).read_bytes(), data)
        records = s.read(out / 'PUBLICATION_MANIFEST.json')['source_files']
        self.assertTrue(all(r['source_sha256'] == r['public_sha256'] for r in records))

    def test_interrupted_safe_JSON_is_preserved_and_does_not_erase_registry(self):
        raw = b'{"epoch":1,"loss":'
        root, manifest = self.export({'runs/formal/cases/failed/artifacts/training_history.json': raw})
        out = s.publish_snapshot(root, self.root / 'public', manifest, {'/data1/user': '<ROOT>'})
        self.assertEqual(next(out.rglob('training_history.json')).read_bytes(), raw)
        record = s.read(out / 'PUBLICATION_MANIFEST.json')['source_files'][0]
        self.assertEqual(record['transformation'], 'original_bytes_partial_JSON_unparsed')

    def test_supervision_private_but_failure_cleanup_and_memory_remain_visible(self):
        root, manifest = self.export({'runs/formal/cases/x/attempt_001/supervision/finished.json': {
            'status': 'failed', 'exit_code': -15, 'identity': {'pid': 4321}, 'remaining_group_members': [],
            'guard': {'reason': 'host_rss_limit', 'peak_group_rss_bytes': 999, 'pid': 4321}}})
        out = s.publish_snapshot(root, self.root / 'public', manifest, {'/data1/user': '<ROOT>'})
        self.assertFalse(list(out.rglob('finished.json')))
        receipt = s.read(out / 'CURATED_TERMINAL_RECEIPTS.json')[0]
        self.assertEqual(receipt['exit_code'], -15); self.assertEqual(receipt['guard']['peak_group_rss_bytes'], 999)
        self.assertNotIn('4321', json.dumps(receipt))

    def test_checkpoint_requires_exact_bounded_accepted_producer_evidence(self):
        path = self.put(self.root, 'checkpoint.pt', b'opaque synthetic test bytes')
        relative = 'runs/case/artifacts/checkpoint.pt'
        proof = {'sha256': s.sha(path), 'bytes': path.stat().st_size, 'public_safe': True,
            'producer_sha256': 'a' * 64, 'finite_CPU_trainable_state': True,
            'independently_accepted_case': True, 'tensor_names': ['projections.0.weight', 'head.weight']}
        self.assertTrue(s.checkpoint_approved(path, relative, {relative: proof}))
        proof['tensor_names'] = ['base_lm.model.layers.0.weight']
        self.assertFalse(s.checkpoint_approved(path, relative, {relative: proof}))
        proof['tensor_names'] = ['head.weight']; proof['sha256'] = 'b' * 64
        self.assertFalse(s.checkpoint_approved(path, relative, {relative: proof}))
        self.assertFalse(s.checkpoint_approved(path, relative, {}))

    def test_remote_checkpoint_name_policy_never_accepts_backbone(self):
        self.assertTrue(f.checkpoint_state_allowed(['head.weight', 'conditions.weight']))
        for names in ([], ['base_lm.x'], ['model.embed_tokens.weight'], ['lm_head.weight']):
            self.assertFalse(f.checkpoint_state_allowed(names))

    def test_partial_registry_keeps_all45_positions_and_unknown_analysis(self):
        cases = [{'case_id': str(i), 'task': 'meltome'} for i in range(45)]
        root, manifest = self.export({'configs/plan.json': {'cases': cases},
            'runs/formal/queue_events.jsonl': b'{"case_id":"0","status":"failed"}\n'})
        manifest.update(plan_relative='configs/plan.json', queue_relative='runs/formal')
        out = s.publish_snapshot(root, self.root / 'public', manifest, {'/data1/user': '<ROOT>'})
        positions = s.read(out / 'QUEUE_POSITIONS.json')
        self.assertEqual(len(positions), 45); self.assertEqual(positions[0]['queue_status'], 'failed')
        self.assertTrue(all(row['analysis_status'] == 'unavailable' for row in positions))
        self.assertIsNone(s.read(out / 'PUBLICATION_MANIFEST.json')['analysis_status_counts'])

    def test_realistic45_admin_ledger_normalizes_paths_and_drops_raw_process_inventory(self):
        remote = '/data1/user/project/experiments/' + s.SCOPE_ID
        cases = [{'case_id': str(i), 'task': 'elm', 'seed': 17} for i in range(45)]
        events = [{'case_id': str(i), 'status': 'running' if i == 0 else 'pending', 'seed': 17,
            'attempt': remote + '/runs/formal/cases/' + str(i) + '/attempt_001',
            'supervisor_identity': {'pid': 123456, 'pgid': 123456, 'start_ticks': 999},
            'last_resource_observation': {'gpu_compute_processes': ['GPU-12345678-abcd, 654321, other-user']}}
            for i in range(45)]
        raw = ''.join(json.dumps(row) + '\n' for row in events).encode()
        predictions = b'{"sample_id":"row","prediction":1.2345678901234567}\r\n'
        root, manifest = self.export({'configs/plan.json': {'cases': cases},
            'runs/formal/queue_events.jsonl': raw, 'runs/formal/cases/0/artifacts/readout_predictions.jsonl': predictions})
        manifest.update(plan_relative='configs/plan.json', queue_relative='runs/formal')
        out = s.publish_snapshot(root, self.root / 'public', manifest)
        ledger = (out / 'runs/formal/queue_events.jsonl').read_text()
        self.assertNotIn('/data1/', ledger); self.assertNotIn('123456', ledger); self.assertNotIn('other-user', ledger)
        rows = [json.loads(line) for line in ledger.splitlines()]
        self.assertEqual(len(rows), 45)
        self.assertEqual([(r['case_id'], r['status'], r['seed']) for r in rows],
                         [(r['case_id'], r['status'], r['seed']) for r in events])
        self.assertEqual((out / 'runs/formal/cases/0/artifacts/readout_predictions.jsonl').read_bytes(), predictions)
        record = next(r for r in s.read(out / 'PUBLICATION_MANIFEST.json')['source_files'] if r['source_relative'].endswith('queue_events.jsonl'))
        self.assertNotEqual(record['source_sha256'], record['public_sha256'])

    def test_observed_runtime_roots_and_unknown_private_diagnostic_are_safe(self):
        prefix = '/data/synthetic_user/anaconda3/envs/fixture-runtime'
        root, manifest = self.export({'runs/formal/cases/0/artifacts/failed.json': {
            'status': 'failed', 'traceback': 'File ' + prefix + '/lib/python3.11/runpy.py',
            'error': 'Unexpected diagnostic /home/another-private-user/secret_location/file.py',
            'seed': 42, 'loss': 0.12345678901234567}})
        manifest['runtime_roots'] = {'sys_prefix': '/data1/user/task-env', 'sys_base_prefix': prefix,
            'sys_base_exec_prefix': prefix, 'sys_executable': '/data1/user/task-env/bin/python'}
        out = s.publish_snapshot(root, self.root / 'public', manifest)
        failed = s.read(next(out.rglob('failed.json')))
        self.assertIn('<RUNTIME_SYS_BASE_PREFIX>', failed['traceback'])
        self.assertIn('private_diagnostic_sha256', failed['error'])
        self.assertEqual(failed['seed'], 42); self.assertEqual(failed['loss'], 0.12345678901234567)
        self.assertNotIn('/home/', json.dumps(failed)); self.assertNotIn(prefix, json.dumps(failed))

    def test_ELM_aggregates_remain_but_perquery_inputs_vectors_and_weights_private(self):
        elm = 'runs/formal/cases/elm__M0__qwen25_1p5b__seed17/attempt_001/artifacts/'
        rt = 'runs/formal/cases/rt__M0__qwen25_1p5b__seed17/attempt_001/artifacts/'
        prediction = b'{"sample_id":"id","prediction":1.2345678901234567}\r\n'
        root, manifest = self.export({elm + 'metrics.json': {'readout': {'class_macro_mrr': 0.123456789}},
            elm + 'training_history.json': [{'epoch': 3, 'loss': 0.1}], elm + 'readout_predictions.jsonl': prediction,
            elm + 'LM_predictions.jsonl': prediction, elm + 'selection_predictions.json': [],
            elm + 'run_manifest.json': {'per_sequence_coverage': ['restricted']}, elm + 'checkpoint.pt': b'private',
            'data/elm/candidates.jsonl': b'{"description":"restricted"}\n',
            'cache/description_vectors.npz': b'private', 'cache/description_vectors.json': {'source': 'private'},
            rt + 'readout_predictions.jsonl': prediction})
        out = s.publish_snapshot(root, self.root / 'public', manifest, {'/data1/user': '<ROOT>'})
        self.assertEqual(s.read(out / (elm + 'metrics.json'))['readout']['class_macro_mrr'], 0.123456789)
        self.assertTrue((out / (elm + 'training_history.json')).is_file())
        self.assertFalse((out / (elm + 'readout_predictions.jsonl')).exists())
        self.assertFalse((out / 'data/elm/candidates.jsonl').exists())
        self.assertFalse((out / 'cache/description_vectors.npz').exists())
        self.assertFalse((out / (elm + 'checkpoint.pt')).exists())
        self.assertEqual((out / (rt + 'readout_predictions.jsonl')).read_bytes(), prediction)
        self.assertEqual(len(s.read(out / 'PUBLICATION_MANIFEST.json')['excluded_files']), 8)

    def test_transport_retry_is_bounded_and_authentication_errors_do_not_retry(self):
        calls, delays = [], []
        def runner(argv):
            calls.append(argv)
            if len(calls) < 3: raise subprocess.CalledProcessError(255, argv, stderr='Connection reset by peer')
            return 'ok'
        self.assertEqual(s.transport_call(['ssh', 'node1', 'one-completion'], runner=runner, sleep=delays.append), 'ok')
        self.assertEqual(len(calls), 3); self.assertEqual(delays, [10, 20])
        with patch.object(s, 'call', side_effect=subprocess.CalledProcessError(255, ['ssh'], stderr='Permission denied')) as call:
            with self.assertRaises(subprocess.CalledProcessError): s.transport_call(['ssh'], sleep=delays.append)
            self.assertEqual(call.call_count, 1)

    def test_isolated_worktree_fetches_fresh_base_without_touching_dirty_checkout(self):
        repo = self.root / 'repo'; repo.mkdir(); sentinel = self.put(repo, 'dirty.txt', b'unchanged')
        with patch.object(s, 'call') as call:
            target = s.isolated_worktree(repo, self.root / 'owned-worktrees', 'result_1')
        commands = [item.args[0] for item in call.call_args_list]
        self.assertEqual(commands[0], ['git', 'fetch', 'origin', 'main'])
        self.assertEqual(commands[1][-1], 'origin/main'); self.assertIn('--detach', commands[1])
        self.assertEqual(sentinel.read_bytes(), b'unchanged'); self.assertNotIn('reset', str(commands))
        self.assertTrue(target.is_relative_to(self.root / 'owned-worktrees'))

    def test_publication_never_stages_outside_new_scope(self):
        with patch.object(s, 'call') as call:
            with self.assertRaises(ValueError): s.commit_and_verify(self.root, [self.root / 'old-results'], 'x')
            call.assert_not_called()

    def test_run_id_cannot_escape_and_existing_public_namespace_never_overwritten(self):
        for value in ('../escape', '..', '/absolute', 'a/b', 'a\\b', 'a:stream'):
            with self.assertRaises(ValueError): s.safe_id(value)
        root, manifest = self.export({}); existing = self.root / 'existing'; existing.mkdir()
        with self.assertRaises(FileExistsError): s.publish_snapshot(root, existing, manifest, {})

    def test_complete_waiter_uses_terminal_and_analysis_event_not_retraining(self):
        queue = self.root / 'queue'; queue.mkdir()
        self.put(queue, 'complete.json', {}); self.put(queue, 'analysis_exit.json', {'exit_code': 0})
        self.assertEqual(f.ready(queue, {'analysis_command': ['python', '-m', 'src.mt_analyze']}), 'queue_terminal')
        with self.assertRaises(ValueError): f.identity_alive({'pid': 123})

    def test_complete_export_bundle_roundtrip_without_training_or_network(self):
        scope = self.root / s.SCOPE_ID; scope.mkdir()
        queue = scope / 'runs/formal_v1'; queue.mkdir(parents=True)
        cache = scope / 'cache'; cache.mkdir()
        producer = self.put(scope, 'src/mt_train.py', b'fixture producer\n')
        helper = self.put(scope, 'resources/finish_and_export.py', b'fixture helper\n')
        plan = {'scope_id': s.SCOPE_ID, 'cases': [{'case_id': str(i)} for i in range(45)],
                'paths': {'scope_root': str(scope), 'data_root': str(scope / 'data'), 'cache_root': str(cache)},
                'source_pins': {str(producer): f.sha(producer)}}
        plan_path = self.put(scope, 'configs/plan.json', plan)
        self.put(scope, 'data/manifest.json', {'fixture': True})
        self.put(cache, 'manifest.json', {'fixture': True})
        self.put(queue, 'queue_binding.json', {'plan_path': str(plan_path), 'plan_sha256': f.sha(plan_path)})
        self.put(queue, 'complete.json', {'case_count': 45})
        def analysis_only(argv, **kwargs):
            self.assertEqual(argv[1:3], ['-m', 'src.mt_analyze'])
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            target = Path(argv[argv.index('--output') + 1]); target.mkdir()
            self.put(target, 'analysis.json', {'cases': [], 'analysis_status_counts': {'not_accepted': 45}})
            self.put(target, 'REPORT.md', b'Fixture partial analysis, no scientific results\n')
            return SimpleNamespace(returncode=0)
        with patch.object(f, '__file__', str(helper)), patch.object(f.subprocess, 'run', side_effect=analysis_only) as process:
            receipt = f.finish(scope, queue, scope / 'resources/export_fixture')
            self.assertEqual(process.call_count, 1)
            recovered = f.finish(scope, queue, scope / 'resources/export_fixture')
            self.assertTrue(recovered['reused_existing_verified_export'])
            self.assertEqual(recovered['sha256'], receipt['sha256']); self.assertEqual(process.call_count, 1)
        self.assertEqual(f.sha(receipt['archive']), receipt['sha256'])
        extracted = s.safe_extract(receipt['archive'], self.root / 'extracted')
        manifest = s.verify_export(extracted, receipt['manifest'], receipt['manifest_sha256'])
        self.assertEqual(manifest['completion_trigger'], 'queue_terminal')
        self.assertEqual(set(manifest['runtime_roots']), {'sys_prefix', 'sys_base_prefix', 'sys_base_exec_prefix', 'sys_executable'})
        self.assertFalse(any('/sequences/' in r['path'] for r in manifest['files']))


if __name__ == '__main__': unittest.main()

"""Focused CPU checks for node-local completion; Git/network/Popen are mocked."""
from contextlib import nullcontext
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCOPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCOPE / 'resources'))
publisher = importlib.import_module('node1_publish')
launcher = importlib.import_module('launch_node1_publish')


class NodePublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scope = self.root / publisher.publication.SCOPE_ID
        self.queue = self.scope / 'runs/formal_v1'; self.queue.mkdir(parents=True)
        self.repo = self.scope / 'private/publication_repo'; self.repo.mkdir(parents=True)
        self.path = self.scope / 'private/config.json'
        self.config = {'scope_root': str(self.scope), 'queue_root': str(self.queue), 'python': sys.executable,
            'repo': str(self.repo), 'run_id': 'formal_v1_node_terminal',
            'git_ssh_command': 'ssh -F /dev/null -i synthetic_deploy_key -oIdentitiesOnly=yes',
            'git_author_name': 'Synthetic Research Publisher', 'git_author_email': 'fixture@example.test',
            'environment_overrides': {'PATH': '/synthetic/git/bin', 'GIT_EXEC_PATH': '/synthetic/git/core',
                                      'GIT_TEMPLATE_DIR': '/synthetic/git/templates'}}
        self.save()

    def save(self, path=None):
        path = path or self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.config), encoding='utf-8')
        return path

    def launch_patches(self):
        return patch.object(publisher, 'exclusive', side_effect=lambda _: nullcontext())

    def test_explicit_config_and_scope_boundaries(self):
        value = publisher.load_config(self.path)
        config = value[0] if isinstance(value, tuple) else value
        self.assertEqual(config['run_id'], self.config['run_id'])
        outside = self.root / 'outside'; outside.mkdir()
        original = dict(self.config)
        for key in ('queue_root', 'repo'):
            with self.subTest(key=key):
                self.config = {**original, key: str(outside)}; self.save()
                with self.assertRaises(ValueError): publisher.load_config(self.path)
        self.config = original; self.save()
        with self.assertRaises(ValueError): publisher.load_config(self.save(outside / 'config.json'))
        self.config['run_id'] = '../other'
        with self.assertRaises(ValueError): publisher.load_config(self.save())

    def test_git_environment_whitelist_and_single_auth_command(self):
        self.config['environment_overrides']['HOME'] = '/synthetic/forbidden'
        with self.assertRaises(ValueError): publisher.load_config(self.save())
        del self.config['environment_overrides']['HOME']
        self.config['git_ssh_command'] = 'ssh; unexpected_command'
        with self.assertRaises(ValueError): publisher.load_config(self.save())

    def test_environment_forces_CPU_threads_and_restores_process_settings(self):
        value = publisher.load_config(self.path)
        config = value[0] if isinstance(value, tuple) else value
        before = {'CUDA_VISIBLE_DEVICES': '5', 'OMP_NUM_THREADS': '9', 'MKL_NUM_THREADS': '9'}
        with patch.dict(os.environ, before):
            with publisher.environment(config):
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '')
                for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                    self.assertEqual(os.environ[name], '2')
                self.assertEqual(os.environ['GIT_SSH_COMMAND'], config['git_ssh_command'])
            for key, value in before.items(): self.assertEqual(os.environ[key], value)

    @unittest.skipUnless(os.name == 'posix', 'Linux flock required')
    def test_real_flock_rejects_second_publisher_and_releases_after_exit(self):
        lock = self.scope / 'private/fixture.lock'
        with publisher.exclusive(lock):
            with self.assertRaises(RuntimeError):
                with publisher.exclusive(lock): pass
        with publisher.exclusive(lock): pass

    def test_launch_is_detached_node_local_with_private_logs_and_no_auth_in_receipt(self):
        identity = {'pid': 12345, 'pgid': 12345, 'sid': 12345, 'start_ticks': 777}
        with self.launch_patches(), patch.object(launcher, 'identity', return_value=identity), \
             patch.object(launcher.subprocess, 'Popen', return_value=SimpleNamespace(pid=12345)) as popen:
            result = launcher.launch(self.path)
        self.assertEqual(result['status'], 'launched')
        args, kwargs = popen.call_args
        self.assertEqual(args[0][0], sys.executable)
        self.assertTrue(args[0][1].endswith('node1_publish.py'))
        self.assertEqual(args[0][-2:], ['--config', str(self.path.resolve())])
        self.assertFalse(any(x in ('ssh', 'scp', 'gh') for x in args[0]))
        self.assertIs(kwargs['start_new_session'], True)
        self.assertEqual(kwargs['stdin'], subprocess.DEVNULL)
        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            self.assertEqual(kwargs['env'][name], '2')
        self.assertNotIn('synthetic_deploy_key', json.dumps(result))
        self.assertTrue(kwargs['stdout'].closed); self.assertTrue(kwargs['stderr'].closed)
        self.assertTrue(Path(kwargs['stdout'].name).is_relative_to(self.scope / 'private'))

    def test_duplicate_live_launch_is_rejected_without_new_child(self):
        identity = {'pid': 12345, 'pgid': 12345, 'sid': 12345, 'start_ticks': 777}
        with self.launch_patches(), patch.object(launcher, 'identity', return_value=identity), \
             patch.object(launcher, 'identity_alive', return_value=True), \
             patch.object(launcher.subprocess, 'Popen', return_value=SimpleNamespace(pid=12345)) as popen:
            launcher.launch(self.path)
            with self.assertRaises(RuntimeError): launcher.launch(self.path)
            self.assertEqual(popen.call_count, 1)

    def test_previous_dead_launch_and_logs_are_preserved(self):
        directory = self.scope / 'private/node1_publication' / self.config['run_id']
        directory.mkdir(parents=True)
        original = json.dumps({'identity': {'pid': 123, 'pgid': 123, 'sid': 123, 'start_ticks': 123}}).encode()
        receipt = directory / 'launch.json'; receipt.write_bytes(original)
        with self.launch_patches(), patch.object(launcher, 'identity_alive', return_value=False), \
             patch.object(launcher.subprocess, 'Popen') as popen:
            with self.assertRaises((FileExistsError, RuntimeError)): launcher.launch(self.path)
            popen.assert_not_called()
        self.assertEqual(receipt.read_bytes(), original)

    def test_completed_receipt_never_starts_another_child(self):
        directory = self.scope / 'private/node1_publication' / self.config['run_id']
        directory.mkdir(parents=True)
        (directory / 'PUBLICATION_RECEIPT.json').write_text('{"commit":"fixture"}', encoding='utf-8')
        with self.launch_patches(), patch.object(launcher.subprocess, 'Popen') as popen:
            result = launcher.launch(self.path)
            popen.assert_not_called()
        self.assertEqual(result['status'], 'already_completed')


if __name__ == '__main__': unittest.main()

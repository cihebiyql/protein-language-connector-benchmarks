"""Bounded CPU-only regression tests; real process cleanup owns its Popen only."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from src import mm_supervise as safe_proc


class ScriptedEntry:
    def __init__(self, name, stat):
        self.name, self.stat = name, stat

    def __truediv__(self, name):
        if name != 'stat':
            raise AssertionError('Only /proc/PID/stat should be accessed')
        return self

    def read_text(self):
        if isinstance(self.stat, BaseException):
            raise self.stat
        return self.stat


class ScriptedRoot:
    def iterdir(self):
        return iter((ScriptedEntry('101', ProcessLookupError(3, 'PID exited')),
                     ScriptedEntry('102', '102 (still alive) S 1 900 900 0'),
                     ScriptedEntry('self', AssertionError('Must not read non-PID entry'))))

    def glob(self, _pattern):
        # Exact former failure point: next(glob_iterator), outside loop body.
        def disappeared_inside_glob():
            raise ProcessLookupError(3, 'PID exited inside glob iterator')
            yield
        return disappeared_inside_glob()


class ProcRegressionTests(unittest.TestCase):
    def test_disappearing_pid_does_not_escape_read_guard(self):
        root = ScriptedRoot()
        with self.assertRaises(ProcessLookupError):
            next(root.glob('[0-9]*/stat'))
        with patch.object(safe_proc, 'Path', return_value=root):
            self.assertEqual(safe_proc.group_members(900), [102])

    def test_exact_pgid_only_excludes_zombie_and_malformed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = {'201': '201 (name with ) parentheses) S 1 900 900 0',
                     '202': '202 (foreign) S 1 901 901 0',
                     '203': '203 (zombie) Z 1 900 900 0',
                     '204': 'malformed', '205': '205 (bad pgid) S 1 invalid 900 0'}
            for pid, text in stats.items():
                (root / pid).mkdir()
                (root / pid / 'stat').write_text(text)
            (root / 'self').mkdir()
            (root / '206').mkdir()  # Directory exists but stat disappeared.
            with patch.object(safe_proc, 'Path', return_value=root):
                self.assertEqual(safe_proc.group_members(900), [201])

    def test_unavailable_proc_root_fails_explicitly(self):
        class DeniedRoot:
            def iterdir(self):
                raise PermissionError(13, 'proc root inaccessible')
        with patch.object(safe_proc, 'Path', return_value=DeniedRoot()):
            with self.assertRaises(PermissionError):
                safe_proc.group_members(900)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux signal cleanup policy')
    def test_root_unavailable_does_not_disable_owned_group_cleanup(self):
        import signal
        with patch.object(safe_proc, 'group_members', side_effect=PermissionError('proc root')), \
             patch.object(safe_proc.os, 'killpg', create=True) as kill, \
             patch.object(safe_proc.time, 'monotonic', side_effect=[0, 4, 4, 7]):
            safe_proc.terminate_group(900)
        self.assertEqual(kill.call_args_list,
                         [unittest.mock.call(900, signal.SIGTERM), unittest.mock.call(900, signal.SIGKILL)])

    @unittest.skipUnless(sys.platform.startswith('linux'), 'requires Linux /proc')
    def test_real_owned_session_isolated_and_child_exit_observed(self):
        child = subprocess.Popen([sys.executable, '-B', '-c',
                                  'import time; time.sleep(10)'], start_new_session=True)
        try:
            self.assertEqual(os.getpgid(child.pid), child.pid)
            self.assertIn(child.pid, safe_proc.group_members(child.pid))
            self.assertNotIn(child.pid, safe_proc.group_members(os.getpgrp()))
        finally:
            # No killpg and no external PID: only this test's exact Popen handle.
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)
        self.assertNotIn(child.pid, safe_proc.group_members(child.pid))


if __name__ == '__main__':
    unittest.main(verbosity=2)

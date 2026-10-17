import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import mm_supervise as supervisor


class SupervisorTests(unittest.TestCase):
    def test_write_is_exclusive_and_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'one.json'
            supervisor.write_new(path, {'status': 'preserved'})
            with self.assertRaises(FileExistsError):
                supervisor.write_new(path, {'status': 'overwrite'})
            self.assertEqual(json.loads(path.read_text()), {'status': 'preserved'})
            with self.assertRaises(ValueError):
                supervisor.write_new(Path(directory) / 'bad.json', {'x': float('nan')})

    def test_hash_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data'
            path.write_bytes(b'one\r\n')
            first = supervisor.sha(path)
            path.write_bytes(b'one\n')
            self.assertNotEqual(first, supervisor.sha(path))

    @unittest.skipUnless(os.name == 'posix' and Path('/proc').is_dir(), 'Linux process-group test')
    def test_only_owned_process_tree_is_terminated(self):
        # Create an unrelated peer plus our own parent->child tree, proving group isolation.
        peer = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
        own = subprocess.Popen([sys.executable, '-c',
            'import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);print(p.pid,flush=True);time.sleep(60)'],
            stdout=subprocess.PIPE, text=True, start_new_session=True)
        try:
            child = int(own.stdout.readline())
            time.sleep(0.1)
            self.assertIn(own.pid, supervisor.group_members(own.pid))
            self.assertIn(child, supervisor.group_members(own.pid))
            self.assertGreater(supervisor.group_rss(own.pid), 0)
            supervisor.terminate_group(own.pid)
            own.wait(timeout=5)
            self.assertEqual(supervisor.group_members(own.pid), [])
            self.assertIsNone(peer.poll())
        finally:
            supervisor.terminate_group(own.pid)
            own.wait(timeout=5)
            supervisor.terminate_group(peer.pid)
            peer.wait(timeout=5)
            own.stdout.close()


if __name__ == '__main__':
    unittest.main()

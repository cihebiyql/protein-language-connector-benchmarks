import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'resources'))
spec = importlib.util.spec_from_file_location('recovery_publication_policy_test', ROOT / 'resources/recovery_publication_policy_v1.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class RecoveryPublicationTests(unittest.TestCase):
    def test_preserved_raw_ledger_stays_private_and_history_is_normalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / 'source', root / 'public'
            names = ('runs/reconciled_recovery_v1/provenance/original_queue_events.jsonl',
                     'runs/reconciled_recovery_v1/queue_events.jsonl',
                     'runs/reconciled_recovery_v1/cases/rt__M0__qwen3/artifacts/readout_predictions.jsonl')
            for name in names: (source / name).parent.mkdir(parents=True, exist_ok=True)
            fake_root = '/' + 'data1' + '/private_example/project'
            raw = (json.dumps({'case_id': 'fixed-case', 'attempt': fake_root + '/attempt_001',
                'status': 'failed'}) + '\n').encode('utf-8')
            (source / names[0]).write_bytes(raw)
            (source / names[1]).write_bytes(raw)
            predictions = b'{"sample_id":"fixed-row","prediction":1.25}\n'
            (source / names[2]).write_bytes(predictions)
            manifest = {'scope_root': fake_root, 'files': [{'path': name,
                'sha256': module.sync_results.sha(source / name), 'bytes': (source / name).stat().st_size} for name in names]}
            previous = module.sync_results.publication_restriction
            with module.policy():
                module.sync_results.publish_snapshot(source, destination, manifest)
                self.assertIsNotNone(module.sync_results.publication_restriction('data/elm/train.jsonl'))
            self.assertIs(module.sync_results.publication_restriction, previous)
            self.assertEqual((source / names[0]).read_bytes(), raw)
            self.assertFalse((destination / names[0]).exists())
            self.assertEqual((destination / names[2]).read_bytes(), predictions)
            history = json.loads((destination / names[1]).read_text(encoding='utf-8'))
            self.assertEqual(history['status'], 'failed')
            self.assertIn('<REMOTE_SCOPE>', history['attempt'])
            exported = json.loads((destination / 'PUBLICATION_MANIFEST.json').read_text(encoding='utf-8'))
            self.assertEqual(exported['excluded_files'][0]['source_sha256'], module.sync_results.sha(source / names[0]))

    def test_exceptions_restore_the_original_publication_scope(self):
        previous = module.sync_results.publication_restriction
        with self.assertRaises(RuntimeError):
            with module.policy(): raise RuntimeError('synthetic failure')
        self.assertIs(module.sync_results.publication_restriction, previous)

if __name__ == '__main__': unittest.main()

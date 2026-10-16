import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/processed/bench_v02/a1_locked_v2'


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line]


class PepbiRecoveryTests(unittest.TestCase):
    def test_official_workbook_fingerprint(self):
        path = ROOT / 'data/raw/bench_v02/ppikb/pepbi_official/browser_recovery/PEPBI.xlsx'
        self.assertEqual(path.stat().st_size, 217930)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), '0f3ae5e9021b5c0e925114c5b53a0cb441de90adc42134e5144d4ac9c4826a24')

    def test_complete_inventory_and_source_join(self):
        inventory = rows(OUT / 'pepbi_complete_sample_inventory.jsonl')
        self.assertEqual(len(inventory), 329)
        self.assertTrue(all(row['original_itc_source_dois'] for row in inventory))
        self.assertEqual(len({row['peptide_sequence'] for row in inventory}), 84)
        self.assertEqual(len({row['target_sequence'] for row in inventory}), 58)
        self.assertTrue(all(row['chemical_state']['n_terminal'] is None for row in inventory))

    def test_no_actual_pepbi_external_family_remains(self):
        forbidden = {r['sample_id'] for r in rows(OUT / 'pepbi_actual_global_broad_exclusion_audit.jsonl')}
        for scenario in ['new_peptide_known_target', 'new_peptide_new_target']:
            samples = rows(OUT / scenario / 'samples.jsonl')
            self.assertFalse(forbidden & {r['sample_id'] for r in samples})
            train_targets = {r['target_sequence'] for r in samples if r['split'] == 'train'}
            if scenario == 'new_peptide_known_target':
                self.assertTrue(all(r['target_sequence'] in train_targets for r in samples if r['split'] != 'train'))

    def test_parent_tests_preserved_and_all_pins_intact(self):
        for scenario in ['new_peptide_known_target', 'new_peptide_new_target']:
            parent = ROOT / 'data/processed/bench_v02/a1_locked' / scenario
            current = OUT / scenario
            self.assertEqual((parent / 'test_gold.jsonl').read_bytes(), (current / 'test_gold.jsonl').read_bytes())
            lock = json.loads((current / 'benchmark_lock.json').read_text(encoding='utf8'))
            for path, expected in lock['pins'].items():
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected, path)


if __name__ == '__main__':
    unittest.main()

import importlib.util
import itertools
import json
import hashlib
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'scripts/bench_v02/split_affinity.py'
SPEC = importlib.util.spec_from_file_location('a1_split', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def slow_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


class SplitTests(unittest.TestCase):
    def test_exact_edit_distance_exhaustive_short_strings(self):
        strings = [''.join(chars) for n in range(5) for chars in itertools.product('AC', repeat=n)]
        for a in strings:
            for b in strings:
                self.assertEqual(MODULE.edit_distance(a, b), slow_distance(a, b), (a, b))

    def test_composition_filter_never_removes_true_edge(self):
        strings = [''.join(chars) for chars in itertools.product('AC', repeat=5)] + ['AAAA', 'CCCC', 'AACCCA']
        actual = {(a, b) for a, b, _, _ in MODULE.peptide_matches(strings, strings)}
        expected = {(a, b) for a in strings for b in strings if slow_distance(a, b) <= int(max(len(a), len(b)) * .2 + 1e-9)}
        self.assertEqual(actual, expected)

    def test_connected_components_transitive(self):
        dsu = MODULE.DSU(['a', 'b', 'c']); dsu.union('a', 'b'); dsu.union('b', 'c')
        self.assertEqual(len(set(dsu.labels('pf:').values())), 1)

    def rows(self):
        return [{'sample_id': str(i), 'source_doi': 'doi:' + str(i), 'target_sequence': 'same' if i < 10 else 'rare',
                 'pkd': i, 'groups': {'peptide_family': 'p:' + str(i), 'target_family': 't:' + str(i // 3),
                                     'source_doi': 'doi:' + str(i), 'repeat_measurement_candidate': str(i)}} for i in range(30)]

    def test_split_does_not_read_labels(self):
        rows = self.rows(); first, _, _ = MODULE.allocate(rows, 'new_peptide_new_target')
        for row in rows: row['pkd'] = 1000 - row['pkd']
        second, _, _ = MODULE.allocate(rows, 'new_peptide_new_target')
        self.assertEqual([(r['sample_id'], r['split']) for r in first], [(r['sample_id'], r['split']) for r in second])
        self.assertTrue(MODULE.verify(first, 'new_peptide_new_target')['target_family_disjoint'])

    def test_known_targets_are_exact(self):
        samples, _, _ = MODULE.allocate(self.rows(), 'new_peptide_known_target')
        training = {r['target_sequence'] for r in samples if r['split'] == 'train'}
        self.assertTrue(all(r['target_sequence'] in training for r in samples if r['split'] != 'train'))

    def test_actual_frozen_panels_and_pinned_homology_edges(self):
        root = PATH.parents[2]
        out = root / 'data/processed/bench_v02/a1_locked'
        if not (out / 'split_summary.json').exists():
            self.skipTest('real artifacts not prepared')
        groups = json.loads((out / 'group_audit.json').read_text(encoding='utf8'))
        for scenario in MODULE.SCENARIOS:
            folder = out / scenario
            rows = MODULE.read_jsonl(folder / 'samples.jsonl')
            MODULE.verify(rows, scenario)
            lock = json.loads((folder / 'benchmark_lock.json').read_text(encoding='utf8'))
            for path, expected in lock['pins'].items():
                self.assertEqual(MODULE.sha(root / path), expected, path)
            ownership = {row['groups']['peptide_family']: row['split'] for row in rows}
            for edge in MODULE.read_jsonl(out / 'peptide_edges.jsonl'):
                a = groups['peptide_sequence_hash_to_family'][edge['query']]
                b = groups['peptide_sequence_hash_to_family'][edge['target']]
                self.assertEqual(a, b)
                if a in ownership and b in ownership:
                    self.assertEqual(ownership[a], ownership[b])
            if scenario == 'new_peptide_new_target':
                owner = {row['groups']['target_family']: row['split'] for row in rows}
                for edge in MODULE.load_alignments(out / 'target_alignments.tsv'):
                    a = groups['target_sequence_hash_to_family'][edge['query']]
                    b = groups['target_sequence_hash_to_family'][edge['target']]
                    self.assertEqual(a, b)
                    if a in owner and b in owner:
                        self.assertEqual(owner[a], owner[b])


if __name__ == '__main__':
    unittest.main()

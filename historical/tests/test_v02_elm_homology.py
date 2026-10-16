"""Evidence-level checks for the restricted B1 homology lock."""
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('isolate_elm', ROOT / 'scripts/bench_v02/isolate_elm.py')
isolation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(isolation)
OUT = ROOT / 'data/processed/bench_v02/b1_homology_locked'


class B1HomologyTests(unittest.TestCase):
    def test_short_shared_motif_is_not_full_protein_homology(self):
        hit = ['A', 'B', '1.0', '.01', '.01', '7', '1e-20']
        self.assertFalse(isolation.accepted_alignment(hit, 'parent'))
        hit[3:5] = ['1.0', '1.0']
        self.assertFalse(isolation.accepted_alignment(hit, 'context'))

    def test_exhaustive_prefilter_enumerates_every_target(self):
        for kind in ['parent', 'context']:
            work = OUT / ('mmseqs_' + kind)
            target_data = (work / 'targets.index').read_bytes()
            self.assertEqual((work / 'allpairs').read_bytes(), target_data)
            query_keys = [x.split('\t')[0] for x in (work / 'queries.index').read_text().splitlines()]
            entries = [x.split('\t') for x in (work / 'allpairs.index').read_text().splitlines()]
            self.assertEqual([x[0] for x in entries], query_keys)
            self.assertTrue(all(x[1:] == ['0', str(len(target_data))] for x in entries))
            self.assertEqual((work / 'allpairs.dbtype').read_bytes(), bytes([7, 0, 0, 0]))

    def test_physical_pins_and_restricted_external_status(self):
        lock = json.loads((OUT / 'benchmark_lock.json').read_text(encoding='utf8'))
        self.assertEqual(lock['status'], 'frozen_restricted_scope')
        self.assertFalse(lock['checks']['external_isolation_verified'])
        for relative, expected in lock['pins'].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected, relative)
        evidence = lock['homology_evidence']
        for key in [evidence['protocol_path'], evidence['group_audit_path']] + evidence['alignment_paths']:
            self.assertIn(key, lock['pins'])

    def test_independence_gate_counts_families_not_ortholog_rows(self):
        records = isolation.rows(OUT / 'b1_samples.jsonl')
        groups = defaultdict(set)
        for record in records:
            groups[record['class_id']].add(record['groups']['parent_homology'])
        self.assertTrue(groups)
        self.assertTrue(all(len(value) >= 10 for value in groups.values()))
        audits = {r['class_id']: r for r in isolation.rows(OUT / 'class_independence_audit.jsonl')}
        for class_id, value in groups.items():
            self.assertEqual(len(value), audits[class_id]['independent_parent_homology_families'])
        self.assertFalse(audits['CLV_C14_Caspase3-7']['eligible'])

    def test_all_retained_group_members_are_disjoint_and_sourced(self):
        records = isolation.rows(OUT / 'b1_samples.jsonl')
        seen = defaultdict(set)
        for r in records:
            self.assertFalse(r['homology_exclusion_reasons'])
            self.assertTrue(r['groups']['publication'])
            keys = [('sequence', r['segment_sequence']), ('class', r['class_id'])]
            keys += [(field, r['groups'][field]) for field in ['parent_homology', 'peptide_family']]
            keys += [('publication', p) for p in r['groups']['publication']]
            for key in keys:
                seen[key].add(r['split'])
        self.assertTrue(all(len(value) == 1 for value in seen.values()))

    def test_catalog_review_hides_direct_sequence_answers(self):
        catalog = {r['candidate_id']: r['description'] for r in isolation.rows(OUT / 'b1_candidates_main.jsonl')}
        classes = {r['class_id']: r for r in isolation.rows(OUT / 'b1_classes_main.jsonl')}
        reviews = isolation.rows(OUT / 'description_review.jsonl')
        self.assertEqual(len(catalog), len(reviews))
        for r in reviews:
            self.assertEqual(catalog[r['candidate_id']], r['reviewed_description'])
            self.assertEqual(classes[r['class_id']]['class_description_no_rule'], r['reviewed_description'])
            self.assertNotIn('[rule omitted]', r['reviewed_description'])
            self.assertNotRegex(r['reviewed_description'], r'\{\d|\[[A-Z^]+\]|\bposition\s*[+-]?\d|\b[A-Zx]-[A-Zx]-[A-Zx]')
        self.assertNotIn('position', catalog['candidate_0121'])
        self.assertNotIn('}', catalog['candidate_0070'])

    def test_known_positives_and_gold_are_actual_same_sequence_annotations(self):
        records = isolation.rows(OUT / 'b1_samples.jsonl')
        positive = defaultdict(set)
        for r in records:
            positive[r['segment_sequence']].add(r['class_id'])
        for r in records:
            self.assertEqual(set(r['known_positive_class_ids']), positive[r['segment_sequence']])
        self.assertEqual(isolation.rows(OUT / 'test_gold.jsonl'), [r for r in records if r['split'] == 'test'])

    def test_parent_and_context_alignment_edges_resolve_to_same_component(self):
        audit = json.loads((OUT / 'group_audit.json').read_text(encoding='utf8'))
        lookup = {}
        for kind, key in [('parent', 'parent_components'), ('context', 'peptide_family_components')]:
            lookup[kind] = {member: group['group_id'] for group in audit[key] for member in group['members']}
        for hit in isolation.rows(OUT / 'accepted_homology_edges.jsonl'):
            self.assertEqual(lookup[hit['kind']][hit['query']], lookup[hit['kind']][hit['target']])
        self.assertEqual(audit['external_retained_component_edges'], 0)


if __name__ == '__main__':
    unittest.main()

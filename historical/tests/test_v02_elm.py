"""ELM v0.2 benchmark isolation and evidence semantics."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('prepare_elm', ROOT / 'scripts/bench_v02/prepare_elm.py')
elm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(elm)


class ElmPreparationTests(unittest.TestCase):
    def test_overlap_is_one_independent_region(self):
        rows = [{'parent_accession': 'P1', 'start': s, 'end': e} for s, e in [(1, 6), (4, 9), (12, 14)]]
        self.assertEqual(elm.independent_regions(rows), 2)

    def test_rule_text_is_removed_without_losing_function(self):
        text = 'Nuclear targeting site (K-R-X-R). Binds a transport receptor.'
        clean = elm.clean_description(text)
        self.assertNotIn('K-R-X-R', clean)
        self.assertIn('transport receptor', clean)
        self.assertNotIn('[ST]', elm.clean_description('Recognizes [ST]P sites.'))

    def test_conflicting_official_fasta_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'test.fasta'
            path.write_text('>sp|P1|A\nAAA\n>sp|P1|B\nCCC\n', encoding='utf8')
            with self.assertRaises(ValueError):
                elm.read_fasta(path)

    def test_real_artifacts_are_disjoint_and_source_verified(self):
        rows = [json.loads(x) for x in (ROOT / 'data/processed/bench_v02/b1_samples.jsonl').read_text(encoding='utf8').splitlines()]
        self.assertGreater(len(rows), 0)
        fasta = elm.read_fasta(ROOT / 'data/raw/bench_v02/elm/instances.fasta')
        seen = {}
        excluded_parents, _, _ = elm.external_exclusions()
        for r in rows:
            self.assertFalse(r['readiness_reasons'])
            self.assertEqual(r['segment_sequence'], fasta[r['parent_accession']][r['start']-1:r['end']])
            self.assertNotIn(r['groups']['parent'], excluded_parents)
            for key in [('class', r['class_id']), ('parent', r['groups']['parent']), ('sequence', r['segment_sequence'])] + [('pmid', p) for p in r['groups']['publication']]:
                self.assertIn(seen.setdefault(key, r['split']), [r['split']])
            self.assertFalse(r['free_peptide_activity_claim'])

    def test_no_gold_label_or_regex_in_model_inputs(self):
        rows = [json.loads(x) for x in (ROOT / 'data/processed/bench_v02/b1_model_inputs.jsonl').read_text(encoding='utf8').splitlines()]
        for r in rows:
            self.assertFalse({'class_id', 'parent_accession', 'class_description_no_rule', 'regex'} & r.keys())

    def test_eligibility_and_review_counts(self):
        out = ROOT / 'data/processed/bench_v02'
        audits = [json.loads(x) for x in (out / 'b1_class_eligibility.jsonl').read_text(encoding='utf8').splitlines()]
        for r in audits:
            self.assertEqual(r['eligible'], r['independent_parent_regions'] >= 10)
        cards = [json.loads(x) for x in (out / 'b2_evidence_registry.jsonl').read_text(encoding='utf8').splitlines()]
        self.assertTrue(all(not c['qa_eligible'] and c['review_status'] == 'review_pending' for c in cards))
        self.assertTrue(all(not c['structural_link_is_functional_hotspot'] for c in cards))

    def test_model_eukaryotes_taxonomy_requires_exact_verified_name(self):
        names = elm.verified_taxonomy(ROOT / 'data/raw/bench_v02/elm')
        self.assertEqual(names['homo sapiens']['taxid'], 9606)
        self.assertEqual(names['saccharomyces cerevisiae s288c']['allowed_species_taxid'], 4932)
        self.assertEqual(names['schizosaccharomyces pombe 972h-']['allowed_species_taxid'], 4896)
        self.assertNotIn('human immunodeficiency virus type 1 bh10', names)
        self.assertNotIn('homo sapiens unknown isolate', names)
        self.assertNotIn('saccharomyces cerevisiae unverified strain', names)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                elm.verified_taxonomy(Path(d))

    def test_model_scope_cannot_overwrite_human_snapshot(self):
        with self.assertRaises(ValueError):
            elm.prepare(scope='model_eukaryotes')

    def test_scope_split_protocol_ratio_and_legacy_rule(self):
        assignments = []
        for residue in range(20):
            with patch.object(elm.hashlib, 'sha256', return_value=Mock(hexdigest=lambda: f'{residue:08x}')) as hashing:
                assignments.append(elm.class_split('EXAMPLE', 'model_eukaryotes'))
                self.assertIn(b'model_eukaryotes-70_15_15-v1', hashing.call_args.args[0])
        self.assertEqual(assignments, ['test'] * 3 + ['validation'] * 3 + ['train'] * 14)
        with patch.object(elm.hashlib, 'sha256', return_value=Mock(hexdigest=lambda: '00000002')) as hashing:
            self.assertEqual(elm.class_split('EXAMPLE'), 'validation')
            self.assertEqual(hashing.call_args.args[0], b'pepbridge-v02-class-20261015:EXAMPLE')

    def test_previous_prepared_scope_preservation_hashes(self):
        path = ROOT / 'data/processed/bench_v02/b1_model_eukaryotes/prepared_rule_70_10_20'
        record = json.loads((path / 'preservation_manifest.json').read_text(encoding='utf8'))
        for item in record['files']:
            self.assertEqual(elm.hashlib.sha256((path / item['file']).read_bytes()).hexdigest(), item['sha256'])

    def test_model_scope_real_taxonomy_and_split_isolation(self):
        out = ROOT / 'data/processed/bench_v02/b1_model_eukaryotes'
        rows = [json.loads(x) for x in (out / 'b1_samples.jsonl').read_text(encoding='utf8').splitlines()]
        self.assertGreater(len(rows), 567)
        seen = {}
        for r in rows:
            self.assertTrue(r['eligibility']['taxonomy_verified'])
            self.assertIn(r['taxon']['allowed_species_taxid'], elm.MODEL_EUKARYOTE_TAXIDS)
            self.assertFalse(r['readiness_reasons'])
            keys = [('class', r['class_id']), ('parent', r['groups']['parent']), ('sequence', r['segment_sequence'])]
            keys += [('pmid', p) for p in r['groups']['publication']]
            for key in keys:
                self.assertEqual(seen.setdefault(key, r['split']), r['split'])
        manifest = json.loads((out / 'b1_manifest.json').read_text(encoding='utf8'))
        self.assertEqual(manifest['scope'], 'model_eukaryotes')
        self.assertEqual(manifest['homology_cluster_audit'], 'pending_global_cross_source_audit')


if __name__ == '__main__':
    unittest.main()

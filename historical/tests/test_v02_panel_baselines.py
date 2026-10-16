"""Fixture-only tests of frozen-feature predictors and data aggregation rules."""
from pathlib import Path
import sys
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
if torch is not None:
    from bench_v02.train_panel_baselines import A1Predictor, B1Aligner, ContextEncoder, deduplicate_b1, multi_positive_loss


@unittest.skipIf(torch is None, 'Torch tests run in the existing WSL GPU environment')
class PanelBaselineTests(unittest.TestCase):
    def test_chemical_categories_accept_published_case_variants(self):
        row = {'chemical_state': {'n_terminal': 'Free', 'c_terminal': 'Amide', 'topology': 'Linear',
                                   'residue_modification': 'No'}, 'experimental_context': {}}
        encoder = ContextEncoder([row])
        encoded = encoder(row)
        self.assertEqual(encoded.shape, (ContextEncoder.dimension,))
        self.assertEqual(encoded[8].item(), 1.)
        self.assertEqual(encoded[12].item(), 1.)
        lower = {**row, 'chemical_state': {k: v.lower() for k, v in row['chemical_state'].items()}}
        self.assertTrue(torch.equal(encoded, encoder(lower)))

    def test_context_excludes_gold_and_source_metadata(self):
        row = {'chemical_state': {}, 'experimental_context': {'ph': 7.4}, 'pkd': 2., 'raw_metadata': {'answer': 2}}
        encoder = ContextEncoder([row])
        changed = {**row, 'pkd': 12., 'source_doi': 'different', 'raw_metadata': {'answer': 12}}
        self.assertTrue(torch.equal(encoder(row), encoder(changed)))
        self.assertEqual(encoder.statistics['ph']['mean'], 7.4)

    def test_multilabel_exact_sequence_becomes_one_query(self):
        rows = [{'sample_id': 'a', 'split': 'train', 'segment_sequence': 'ACD', 'class_id': 'x',
                 'known_positive_class_ids': ['x'], 'known_positive_candidate_ids': ['c1']},
                {'sample_id': 'b', 'split': 'train', 'segment_sequence': 'ACD', 'class_id': 'y',
                 'known_positive_class_ids': ['y'], 'known_positive_candidate_ids': ['c2']}]
        result = deduplicate_b1(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['known_positive_class_ids'], ['x', 'y'])
        self.assertEqual(result[0]['member_sample_ids'], ['a', 'b'])

    def test_all_documented_positive_gradients_are_positive_supervision(self):
        scores = torch.zeros(1, 3, requires_grad=True)
        multi_positive_loss(scores, torch.tensor([[True, True, False]])).sum().backward()
        self.assertLess(scores.grad[0, 0].item(), 0)
        self.assertLess(scores.grad[0, 1].item(), 0)
        self.assertGreater(scores.grad[0, 2].item(), 0)

    def test_b1_attention_and_mean_exclude_padding(self):
        torch.manual_seed(17)
        residues = torch.randn(2, 5, 640)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        descriptions = torch.randn(4, 1536)
        changed = residues.clone()
        changed[0, 3:] = 9999
        for attention in (True, False):
            model = B1Aligner(attention).eval()
            self.assertTrue(torch.allclose(model(residues, mask, descriptions), model(changed, mask, descriptions)))

    def test_a1_attention_uses_full_target(self):
        model = A1Predictor(True).eval()
        peptide, target = torch.randn(1, 3, 640), torch.randn(1, 1030, 640)
        context = torch.zeros(1, ContextEncoder.dimension)
        pm, tm = torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 1030, dtype=torch.bool)
        original = model(peptide, pm, target, tm, context)
        changed = target.clone()
        changed[:, 1022:] += torch.randn_like(changed[:, 1022:]) * 10
        self.assertGreater(float((original - model(peptide, pm, changed, tm, context)).abs().detach()), 0.)


if __name__ == '__main__':
    unittest.main()

"""Small fixtures verify feature contracts; they are never benchmark scores."""
import sys
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
if torch is not None:
    from bench_v02.cache_panel_features import chunk_spans, content_key, masked_means, protein_payload


@unittest.skipIf(torch is None, 'Torch tests run in the existing WSL GPU environment')
class PanelCacheTests(unittest.TestCase):
    def test_long_target_chunk_covers_every_residue_once(self):
        spans = chunk_spans(8525)
        self.assertEqual(spans[-1][1], 8525)
        self.assertTrue(all(b - a <= 1022 for a, b in spans))
        self.assertEqual([i for a, b in spans for i in range(a, b)], list(range(8525)))

    def test_chunk_boundaries_include_exact_limit(self):
        self.assertEqual(chunk_spans(1022), [[0, 1022]])
        self.assertEqual(chunk_spans(1023), [[0, 1022], [1022, 1023]])
        with self.assertRaises(ValueError):
            chunk_spans(0)

    def test_payload_rejects_silent_truncation(self):
        with self.assertRaises(ValueError):
            protein_payload('AAAA', torch.zeros(3, 640), 'fixture')
        with self.assertRaises(ValueError):
            protein_payload('AAAX', torch.zeros(4, 640), 'fixture')

    def test_masked_means_exclude_padding(self):
        hidden = torch.tensor([[[1., 2.], [3., 4.], [999., 999.]]])
        actual = masked_means(hidden, torch.tensor([[1, 1, 0]]))
        self.assertTrue(torch.equal(actual, torch.tensor([[2., 3.]])))

    def test_content_address_depends_only_on_actual_input(self):
        self.assertEqual(content_key('ACD'), content_key('ACD'))
        self.assertNotEqual(content_key('ACD'), content_key('ACE'))
        self.assertNotEqual(content_key('function.'), content_key('function'))


if __name__ == '__main__':
    unittest.main()

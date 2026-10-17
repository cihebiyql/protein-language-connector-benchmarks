import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.mt_extract import ESM_REVISION, ESM_PINS, SEMANTICS, FeatureStore, chunk_spans, extract_sequence
from src.mt_data import file_sha, sequence_sha


class CacheTests(unittest.TestCase):
    def test_chunk_policy(self):
        self.assertEqual(chunk_spans(2045), [(0, 1022), (1022, 2044), (2044, 2045)])
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                chunk_spans(bad)

    def test_integrity_and_coverage_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = {'schema': 'multitask-esm-cache.v1', 'model_revision': ESM_REVISION,
                    'model_pins': ESM_PINS, 'semantics': SEMANTICS,
                    'source_semantics': {'transformers_version': '4.51.3', 'verified': True}}
            (root / 'cache_spec.json').write_text(json.dumps(spec))
            (root / 'sequences').mkdir()
            sequence = 'ACDU'
            sha = sequence_sha(sequence)
            path = root / 'sequences' / (sha + '.npz')
            np.savez(path, p15=np.zeros((8, 640), dtype=np.float32), p30=np.ones((8, 640), dtype=np.float32))
            receipt = {'cache_spec_sha256': file_sha(root / 'cache_spec.json'), 'sequence_sha256': sha,
                       'length': 4, 'raw_residues_covered': 4, 'raw_full_coverage': True,
                       'chunk_spans': [[0, 4]], 'sha256': file_sha(path)}
            path.with_suffix('.json').write_text(json.dumps(receipt))
            store = FeatureStore(root)
            self.assertEqual(float(store.get(sequence)['p30'].mean()), 1.)
            self.assertEqual(store.verify_all([{'sequence': sequence}] * 2)['sequences'], 1)
            receipt['raw_residues_covered'] = 3
            path.with_suffix('.json').write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                store.get(sequence)

    def test_tiny_frozen_model_chunk_hook_and_native_U(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch fixture requires existing CPU Torch runtime')
        from types import SimpleNamespace
        class Block(torch.nn.Module):
            def forward(self, value):
                return (value + 1,)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Module()
                self.encoder.layer = torch.nn.ModuleList([Block() for _ in range(30)])
            def forward(self, input_ids, **kwargs):
                value = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
                for block in self.encoder.layer:
                    value = block(value)[0]
                return SimpleNamespace(last_hidden_state=value * 2)
        class Tokenizer:
            unk_token_id = 999
            def convert_tokens_to_ids(self, chars):
                return [ord(c) for c in chars]
            def __call__(self, sequence, **kwargs):
                return {'input_ids': torch.tensor([[1] + self.convert_tokens_to_ids(sequence) + [2]])}
        model = Model().requires_grad_(False)
        seq = 'A' * 1022 + 'U'
        pair, receipt = extract_sequence(model, Tokenizer(), seq)
        self.assertEqual(receipt['raw_residues_covered'], 1023)
        self.assertEqual(receipt['chunk_spans'], [(0, 1022), (1022, 1023)])
        self.assertEqual(pair['p15'].shape, (8, 4))
        self.assertAlmostEqual(float(pair['p15'][0, 0]), ord('A') + 15)
        self.assertAlmostEqual(float(pair['p30'][0, 0]), (ord('A') + 30) * 2)


if __name__ == '__main__':
    unittest.main()

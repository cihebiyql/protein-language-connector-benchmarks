"""CPU checks for scientific math, residue identity and immutable sparse caches."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1] / "src/motif_features.py"
SPEC = importlib.util.spec_from_file_location("motif_features", SOURCE)
features = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(features)


class FeatureTests(unittest.TestCase):
    def tiny_model(self):
        model = features.MotifAutoEncoder(4, 7)
        with torch.no_grad():
            model.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.4]))
            model.encoder.weight.copy_(torch.arange(28).reshape(7, 4) / 15 - 1)
            model.encoder.bias.copy_(torch.linspace(-0.5, 0.4, 7))
            model.decoder.weight.copy_(torch.arange(28).reshape(4, 7) / 20 - 0.5)
        return features.freeze_model(model)

    def test_published_math_matches_independent_affine_relu(self):
        model = self.tiny_model()
        x = torch.tensor([[1., -2., 3., -4.], [-1., 2., -3., 4.], [0., 0., 0., 0.]])
        raw = (x - model.bias) @ model.encoder.weight.T + model.encoder.bias
        expected = torch.maximum(raw, torch.zeros_like(raw))
        self.assertTrue(torch.equal(model.encode(x), expected))
        self.assertTrue(torch.equal(model(x), expected @ model.decoder.weight.T + model.bias))

    def test_csr_matches_dense_every_position_and_feature(self):
        model = self.tiny_model()
        x = torch.tensor([[1., -2., 3., -4.], [-1., 2., -3., 4.], [0., 0., 0., 0.]])
        arrays, diagnostics = features.encode_csr(model, x, chunk_size=2)
        features.validate_csr(arrays, len(x), activation_dim=4, dictionary_size=7)
        recovered = np.zeros((3, 7), dtype=np.float32)
        for row in range(3):
            start, end = arrays["csr_indptr"][row:row + 2]
            recovered[row, arrays["csr_indices"][start:end]] = arrays["csr_data"][start:end]
        np.testing.assert_allclose(recovered, model.encode(x).numpy(), atol=1e-6, rtol=0)
        np.testing.assert_array_equal(arrays["h33"], x.numpy())
        expected_error = (model(x) - x).double().square().mean().item()
        self.assertAlmostEqual(diagnostics["reconstruction_mse"], expected_error, places=6)
        self.assertAlmostEqual(diagnostics["l0_mean"], float((recovered > 0).sum(1).mean()))

    def test_chunking_does_not_change_features_or_rows(self):
        model = self.tiny_model()
        x = torch.arange(44).reshape(11, 4).float() / 13
        one, _ = features.encode_csr(model, x, chunk_size=1)
        many, _ = features.encode_csr(model, x, chunk_size=8)
        for name in ("csr_indices", "csr_indptr", "csr_shape", "h33"):
            np.testing.assert_array_equal(one[name], many[name])
        np.testing.assert_allclose(one["csr_data"], many["csr_data"], rtol=1e-6, atol=1e-6)

    def test_bos_eos_removed_without_position_shift(self):
        tensor = torch.arange(28).reshape(1, 7, 4).float()
        got = features.residue_representation(tensor, sequence_length=5)
        self.assertTrue(torch.equal(got, tensor[0, 1:6]))
        with self.assertRaises(ValueError):
            features.residue_representation(tensor, sequence_length=4)
        with self.assertRaises(ValueError):
            features.residue_representation(tensor.expand(2, -1, -1), sequence_length=5)

    def test_short_and_full_contexts_have_different_sequence_identities(self):
        self.assertNotEqual(features.sequence_sha256("ACDEFGH"), features.sequence_sha256("DEF"))
        with self.assertRaises(ValueError):
            features.sequence_sha256("acdef")
        with self.assertRaises(ValueError):
            features.sequence_sha256("A" * 1023)

    def test_sequence_reader_deduplicates_and_ignores_labels(self):
        rows = [
            {"parent_sequence": "ACDEFGH", "short_sequence": "DEF", "class_id": "secret-a"},
            {"parent_sequence": "ACDEFGH", "short_sequence": "EFG", "class_id": "secret-b"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "panel.jsonl"
            path.write_text("\n".join(map(json.dumps, rows)))
            first = features.read_panel_sequences(path)
            rows[0]["class_id"] = {"not_a_valid_label": [1, 2, 3]}
            rows[1]["split"] = "a_changed_label"
            path.write_text("\n".join(map(json.dumps, rows)))
            second = features.read_panel_sequences(path)
        self.assertEqual(first, second)
        self.assertEqual(len(first["full"]), 1)
        self.assertEqual(len(first["short"]), 2)

    def test_frozen_parameters_and_no_gradient_outputs(self):
        model = self.tiny_model()
        self.assertFalse(model.training)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        arrays, _ = features.encode_csr(model, torch.ones(3, 4, requires_grad=True))
        self.assertTrue(np.isfinite(arrays["csr_data"]).all())
        model.train()
        with self.assertRaises(ValueError):
            features.encode_csr(model, torch.ones(3, 4))

    def test_invalid_or_nonfinite_hidden_states_rejected(self):
        model = self.tiny_model()
        for x in (torch.ones(2, 4, dtype=torch.float64), torch.full((2, 4), float("nan")),
                  torch.ones(2, 5), torch.ones(0, 4)):
            with self.assertRaises(ValueError):
                features.encode_csr(model, x)
        with self.assertRaises(ValueError):
            features.encode_csr(model, torch.ones(2, 4), chunk_size=129)

    def test_malformed_csr_rejected(self):
        model = self.tiny_model()
        arrays, _ = features.encode_csr(model, torch.ones(2, 4))
        bad = {key: value.copy() for key, value in arrays.items()}
        bad["csr_indptr"][-1] += 1
        with self.assertRaises(ValueError):
            features.validate_csr(bad, 2, activation_dim=4, dictionary_size=7)

    def test_cache_npz_has_no_objects_and_receipt_pins_are_immutable(self):
        model = self.tiny_model()
        arrays, diagnostics = features.encode_csr(model, torch.ones(3, 4))
        pins = {"feature_source": {"sha256": "source1"}, "panel": {"sha256": "panel1"}}
        original = features.validate_csr
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "full" / (features.sequence_sha256("ACD") + ".npz")
            with patch.object(features, "validate_csr", side_effect=lambda a, n: original(a, n, 4, 7)):
                written = features.save_cache(path, "ACD", arrays, diagnostics, pins)
                with np.load(path, allow_pickle=False) as archive:
                    self.assertEqual(set(archive.files), features.ARRAY_KEYS)
                    self.assertTrue(all(archive[name].dtype != object for name in archive.files))
                self.assertEqual(written, features.read_cache(path, "ACD", pins))
                with self.assertRaises(RuntimeError):
                    features.read_cache(path, "ACD", {"feature_source": {"sha256": "changed"}})
                with self.assertRaises(RuntimeError):
                    features.save_cache(path, "ACD", arrays, diagnostics, pins)

    def test_weights_only_esm_loader_never_requests_unsafe_mode(self):
        with patch.object(torch, "load", return_value={"cfg": {}, "model": {}}) as load:
            features.safe_esm_checkpoint(Path("official.pt"))
        self.assertIs(load.call_args.kwargs["weights_only"], True)


if __name__ == "__main__":
    unittest.main()

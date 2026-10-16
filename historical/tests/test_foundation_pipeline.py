"""BASE03 fixture tests; no foundation weights or benchmark runs are loaded."""
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from foundation_20261016.run_predictor_pipeline import (
    evaluate_records, generate_records, greedy_tokens, parse_answer,
    predictor_values, prompt_text, select_predictor, structured_input,
)


def row(sample='one', gold=8.123456789):
    return {'sample_id': sample, 'split': 'test', 'pkd': gold,
            'target_name': 'SECRET_TARGET_NAME', 'peptide_sequence': 'ACD',
            'target_sequence': 'MKW', 'raw_metadata': {'answer': gold},
            'chemical_state': {'topology': 'linear', 'n_terminal': 'free', 'c_terminal': 'amide'},
            'experimental_context': {'ph': 7.4, 'assay': 'fluorescence polarization',
                                     'buffer': 'HEPES',
                                     'actual_construct_sequence': None}}


class FakeTokenizer:
    eos_token_id, pad_token_id = 3, 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return json.dumps(messages) + '<assistant>'

    def encode(self, text, add_special_tokens):
        return [1, 2]

    def decode(self, tokens, skip_special_tokens):
        return '{"pKD":6.25}'


class FakeLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids, attention_mask, position_ids, use_cache, return_dict):
        self.calls.append({'input_ids': input_ids.clone(), 'position_ids': position_ids.clone(), 'use_cache': use_cache})
        values = torch.zeros(*input_ids.shape, 8)
        # Token-dependent deterministic transitions make EOS and padding testable.
        for i in range(input_ids.shape[0]):
            for j in range(input_ids.shape[1]):
                values[i, j, min(int(input_ids[i, j]) + 1, 7)] = 1
        return SimpleNamespace(logits=values)


class FakeStore:
    def batch(self, sequences, device):
        values = torch.tensor([[[float(len(sequence))]] for sequence in sequences], device=device)
        return values, torch.ones(len(sequences), 1, dtype=torch.bool, device=device)


class FakePredictor(nn.Module):
    def forward(self, peptide, pm, target, tm, context):
        return peptide[:, 0, 0] + target[:, 0, 0] + context[:, 0]


class FoundationPipelineTests(unittest.TestCase):
    def test_validation_only_selection_ignores_better_test_metrics(self):
        candidates = [{'architecture': 'BASE01', 'best_validation_metric': 2., 'test_mae': 0.},
                      {'architecture': 'MEANPOOL', 'best_validation_metric': 1., 'test_mae': 999.}]
        self.assertEqual(select_predictor(candidates)['architecture'], 'MEANPOOL')
        candidates[0]['best_validation_metric'] = 1.
        self.assertEqual(select_predictor(candidates[::-1])['architecture'], 'BASE01')
        candidates[0]['best_validation_metric'] = float('nan')
        with self.assertRaises(ValueError):
            select_predictor(candidates)

    def test_prompt_excludes_gold_identifiers_and_untrusted_strings(self):
        tokenizer = FakeTokenizer()
        first = row()
        payload = structured_input(first, 6.25)
        text = prompt_text(payload, tokenizer)
        self.assertIn('6.25', text)
        for secret in ('SECRET_TARGET', 'ACD', 'MKW', str(first['pkd']), 'raw_metadata', 'sample_id'):
            self.assertNotIn(secret, text)
        self.assertEqual(payload['allowed_context']['assay'], 'fluorescence polarization')
        self.assertEqual(payload['allowed_context']['buffer'], 'HEPES')
        self.assertIn('fluorescence polarization', text)
        self.assertIn('HEPES', text)
        second = {**first, 'sample_id': 'other', 'pkd': -99999.123,
                  'target_name': 'OTHER_SECRET_TARGET', 'raw_metadata': {'answer': -99999.123}}
        self.assertEqual(prompt_text(structured_input(second, 6.25), tokenizer), text)
        self.assertIsNone(payload['prediction']['uncertainty'])
        self.assertTrue(payload['allowed_context']['missing']['actual_construct_sequence'])
        self.assertFalse(payload['allowed_context']['actual_construct_verified'])
        with self.assertRaises(ValueError):
            structured_input(first, float('inf'))

    def test_strict_json_parser_rejects_nonfinite_prose_and_duplicates(self):
        self.assertEqual(parse_answer('{"pKD":-1.2}')['value'], -1.2)
        for text in ('{"pKD":NaN}', '{"pKD":1e999}', '{"pKD":true}', '{"pKD":"6"}',
                     '{"pKD":1,"pKD":2}', '{"pKD":2,"reason":"x"}',
                     '```json\n{"pKD":2}\n```', '{"pKD":2} trailing', '{"pKD":'):
            with self.subTest(text=text):
                self.assertFalse(parse_answer(text)['valid'])

    def test_real_predictor_interface_uses_frozen_scale_without_gold(self):
        rows = [row('a'), row('b', gold=-100)]
        values = predictor_values(FakePredictor(), FakeStore(), lambda r: torch.tensor([1.]),
                                  rows, 'cpu', label_center=2., label_scale=3., batch_size=1)
        self.assertEqual(values, [23., 23.])

    def test_generation_uses_actual_ids_left_padding_eos_and_no_cache(self):
        model = FakeLM()
        ids = torch.tensor([[0, 1, 2], [1, 2, 3]])
        mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
        result = greedy_tokens(model, ids, mask, max_new_tokens=4, eos_token_id=[3, 5], pad_token_id=0)
        self.assertEqual(result.tolist(), [[3, 0], [4, 5]])
        self.assertFalse(model.calls[0]['use_cache'])
        self.assertEqual(model.calls[0]['input_ids'].tolist(), ids.tolist())
        self.assertEqual(model.calls[0]['position_ids'].tolist(), [[0, 0, 1], [0, 1, 2]])
        for limit in (0, 257, True):
            with self.assertRaises(ValueError):
                greedy_tokens(model, ids, mask, limit)

    def test_raw_outputs_preserved_and_metrics_share_denominator(self):
        rows = [row('a', gold=7.), row('b', gold=8.)]
        out = io.StringIO()
        records = generate_records(rows, [6.25, 6.25], FakeTokenizer(), FakeLM(), 'cpu', handle=out)
        self.assertEqual(records[0]['raw_text'], '{"pKD":6.25}')
        self.assertEqual(records[0]['input_token_ids'], [1, 2])
        self.assertEqual(len(out.getvalue().splitlines()), 2)
        metrics = evaluate_records(rows, records)
        self.assertEqual(metrics['n_valid'], 2)
        self.assertEqual(metrics['source_predictor_mae_same_all_examples'], 1.25)
        self.assertEqual(metrics['generated_pKD_mae_valid_only'], 1.25)
        self.assertEqual(metrics['parsed_estimate_preservation_mae_valid_only'], 0.)
        records[1]['parse'] = parse_answer('bad')
        metrics = evaluate_records(rows, records)
        self.assertEqual(metrics['n_failed'], 1)
        self.assertEqual(metrics['source_predictor_mae_same_all_examples'], 1.25)
        self.assertEqual(metrics['source_predictor_mae_same_valid_examples'], .75)
        self.assertEqual(metrics['generated_pKD_mae_valid_only'], .75)
        self.assertIsNone(metrics['full_sample_generated_mae'])

    def test_all_failed_predictions_are_not_silently_imputed(self):
        rows = [row('a', gold=7.)]
        records = [{'sample_id': 'a', 'source_predictor_pKD': 6., 'parse': parse_answer('bad')}]
        metrics = evaluate_records(rows, records)
        self.assertEqual(metrics['parse_valid_rate'], 0.)
        self.assertIsNone(metrics['generated_pKD_mae_valid_only'])
        self.assertIsNone(metrics['parsed_estimate_preservation_mae_valid_only'])
        records[0]['sample_id'] = 'wrong'
        with self.assertRaises(ValueError):
            evaluate_records(rows, records)


if __name__ == '__main__':
    unittest.main()

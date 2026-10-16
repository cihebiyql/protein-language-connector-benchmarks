"""Small CPU fixtures: no pretrained tokenizer/model/checkpoint is loaded."""
import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from matched_budget_20261016.token_diagnostics import (
    FORMAT, NUMERIC, MIXED, answer_token_categories, batch_token_categories, token_nll_breakdown,
)
try:
    import torch
except ImportError:
    torch = None


class FixtureFastTokenizer:
    is_fast = True
    eos_token_id = 2
    pad_token_id = 0
    all_special_ids = [0, 2]

    def __init__(self, merge=None, mutate=None):
        self.merge, self.mutate = merge, mutate
        self.pieces = {}

    def encode(self, text, add_special_tokens=False):
        return self._encoding(text)['input_ids']

    def _encoding(self, text):
        ids, offsets = [], []
        i = 0
        while i < len(text):
            piece = self.merge if self.merge and text.startswith(self.merge, i) else text[i]
            token = 10000 if piece == self.merge else ord(piece) + 10
            self.pieces[token] = piece
            ids.append(token)
            offsets.append((i, i + len(piece)))
            i += len(piece)
        return {'input_ids': ids, 'offset_mapping': offsets, 'special_tokens_mask': [0] * len(ids)}

    def __call__(self, text, **kwargs):
        result = self._encoding(text)
        if self.mutate:
            self.mutate(result)
        return result

    def decode(self, ids, **kwargs):
        return ''.join(self.pieces[i] for i in ids)


class AnswerCategoryTests(unittest.TestCase):
    def test_negative_and_decimal_precisions(self):
        for text in ['{"pKD":-2.40}', '{"pKD":3.2}', '{"pKD":3.20}', '{"pKD":0}', '{"pKD":1e-2}']:
            tokenizer = FixtureFastTokenizer()
            ids = tokenizer.encode(text) + [2]
            classes = answer_token_categories(tokenizer, text, ids)
            self.assertEqual(classes[-1], FORMAT)
            span = text.split(':')[1][:-1]
            self.assertEqual(classes.count(NUMERIC), len(span))
            self.assertEqual(len(classes), len(ids))

    def test_mixed_boundary_token_and_whitespace(self):
        text = ' {"pKD":-2.40} '
        tokenizer = FixtureFastTokenizer(merge=':-')
        categories = answer_token_categories(tokenizer, text, tokenizer.encode(text) + [2])
        self.assertEqual(categories.count(MIXED), 1)
        self.assertEqual(categories.count(NUMERIC), 4)
        self.assertEqual(categories[0], FORMAT)
        self.assertEqual(categories[-2:], [FORMAT, FORMAT])

    def test_special_token_category_overrides_numeric_span(self):
        tokenizer = FixtureFastTokenizer()
        text = '{"pKD":2}'
        tokenizer.all_special_ids = [0, 2, ord('2') + 10]
        classes = answer_token_categories(tokenizer, text, tokenizer.encode(text) + [2])
        self.assertEqual(classes.count(NUMERIC), 0)
        self.assertEqual(classes[-2:], [FORMAT, FORMAT])

    def test_reject_duplicate_extra_boolean_nonfinite_and_missing_eos(self):
        for text in ['{"pKD":1,"pKD":2}', '{"pKD":1,"x":2}', '{"pKD":true}',
                     '{"pKD":NaN}', '{"pKD":1e999}', '{"pKD":"2"}']:
            tokenizer = FixtureFastTokenizer()
            with self.assertRaises(ValueError):
                answer_token_categories(tokenizer, text, tokenizer.encode(text) + [2])
        tokenizer = FixtureFastTokenizer()
        with self.assertRaises(ValueError):
            answer_token_categories(tokenizer, '{"pKD":1}', tokenizer.encode('{"pKD":1}'))

    def test_reject_slow_tokenizer_or_id_mismatch(self):
        tokenizer = FixtureFastTokenizer()
        text = '{"pKD":1}'
        ids = tokenizer.encode(text) + [2]
        tokenizer.is_fast = False
        with self.assertRaises(ValueError):
            answer_token_categories(tokenizer, text, ids)
        tokenizer.is_fast = True
        ids[1] += 1
        with self.assertRaises(ValueError):
            answer_token_categories(tokenizer, text, ids)

    def test_reject_malformed_uncovered_and_misaligned_offsets(self):
        def shorten(d): d['offset_mapping'].pop()
        def out_of_bounds(d): d['offset_mapping'][0] = (0, 999)
        def reversed_span(d): d['offset_mapping'][2] = (0, 1)
        def missing_char(d): d['offset_mapping'][0] = (0, 0)
        def plausible_wrong_offset(d): d['offset_mapping'][0] = (0, 2)
        for mutate in [shorten, out_of_bounds, reversed_span, missing_char, plausible_wrong_offset]:
            tokenizer = FixtureFastTokenizer(mutate=mutate)
            text = '{"pKD":1}'
            with self.assertRaises(ValueError):
                answer_token_categories(tokenizer, text, tokenizer.encode(text) + [2])


@unittest.skipIf(torch is None, 'CPU torch required for NLL fixtures')
class TokenNLLTests(unittest.TestCase):
    def batch(self):
        tokenizer = FixtureFastTokenizer(merge=':-')
        texts = ['{"pKD":-2.40}', '{"pKD":3.2}']
        answers = [tokenizer.encode(t) + [2] for t in texts]
        size = max(map(len, answers)) + 3
        ids = torch.zeros((2, size), dtype=torch.long)
        labels = torch.full_like(ids, -100)
        attention = torch.zeros_like(ids)
        for i, answer in enumerate(answers):
            start = size - len(answer)
            ids[i, start - 1] = 5
            ids[i, start:] = torch.tensor(answer)
            labels[i, start:] = torch.tensor(answer)
            attention[i, start - 1:] = 1
        categories = batch_token_categories(tokenizer, texts, ids, labels, attention)
        # Includes a boundary-merged synthetic token ID; all tensors remain tiny.
        generator = torch.Generator().manual_seed(29)
        logits = torch.randn((2, size, 10001), generator=generator)
        return tokenizer, texts, ids, labels, attention, categories, logits

    def test_left_padding_eos_and_original_labels_unchanged(self):
        tokenizer, texts, ids, labels, attention, categories, _ = self.batch()
        before_ids, before_labels = ids.clone(), labels.clone()
        again = batch_token_categories(tokenizer, texts, ids, labels, attention)
        self.assertTrue(torch.equal(ids, before_ids))
        self.assertTrue(torch.equal(labels, before_labels))
        self.assertTrue(torch.equal(again, categories))
        self.assertTrue(torch.all(categories[labels == -100] == -100))
        self.assertTrue(torch.all(categories[:, -1] == FORMAT))

    def test_three_categories_exactly_compose_full(self):
        _, _, _, labels, _, categories, logits = self.batch()
        result = token_nll_breakdown(logits, labels, categories)
        names = ('format', 'numeric', 'mixed')
        self.assertEqual(sum(result[x]['count'] for x in names), result['full']['count'])
        self.assertEqual(sum(result[x]['loss_sum'] for x in names), result['full']['loss_sum'])
        self.assertAlmostEqual(sum(result[x]['contribution_nats_per_full_token'] for x in names),
                               result['full']['nll_nats_per_token'], places=12)
        self.assertGreater(result['mixed']['count'], 0)
        self.assertEqual(sum(r['full']['count'] for r in result['per_sample']), result['full']['count'])

    def test_original_supervised_nll_and_causal_alignment(self):
        from foundation_20261016.run_connectors import supervised_nll
        _, _, _, labels, _, categories, logits = self.batch()
        result = token_nll_breakdown(SimpleNamespace(logits=logits), labels, categories)
        original_sum, original_count = supervised_nll(SimpleNamespace(logits=logits), labels)
        self.assertEqual(result['reference_supervised_nll']['loss_sum'], float(original_sum))
        self.assertEqual(result['reference_supervised_nll']['count'], original_count)
        self.assertAlmostEqual(result['full']['loss_sum'], float(original_sum), delta=1e-4)
        changed = logits.clone()
        changed[:, -1] = 100.  # Final position never predicts a supervised target.
        self.assertEqual(token_nll_breakdown(changed, labels, categories)['full'], result['full'])

    def test_nonfinite_logits_and_inconsistent_categories_rejected(self):
        _, _, _, labels, _, categories, logits = self.batch()
        bad = logits.clone()
        bad[0, 0, 0] = math.nan
        with self.assertRaises(ValueError):
            token_nll_breakdown(bad, labels, categories)
        broken = categories.clone()
        broken[0, -1] = -100
        with self.assertRaises(ValueError):
            token_nll_breakdown(logits, labels, broken)
        broken = categories.clone()
        broken[0, -1] = 99
        with self.assertRaises(ValueError):
            token_nll_breakdown(logits, labels, broken)

    def test_batch_labels_and_padding_mismatch_rejected(self):
        tokenizer, texts, ids, labels, attention, _, _ = self.batch()
        wrong = ids.clone()
        wrong[0, -1] = 8
        with self.assertRaises(ValueError):
            batch_token_categories(tokenizer, texts, wrong, labels, attention)
        bad_attention = attention.clone()
        bad_attention[0, -1] = 0
        with self.assertRaises(ValueError):
            batch_token_categories(tokenizer, texts, ids, labels, bad_attention)

    def test_zero_category_and_same_denominator_mode_difference(self):
        _, _, _, labels, _, categories, logits = self.batch()
        categories = torch.where(categories == MIXED, NUMERIC, categories)
        correct = token_nll_breakdown(logits, labels, categories)
        off = token_nll_breakdown(logits + torch.linspace(0, 1, logits.shape[-1]), labels, categories)
        self.assertEqual(correct['mixed']['count'], 0)
        self.assertIsNone(correct['mixed']['nll_nats_per_token'])
        self.assertEqual(correct['mixed']['contribution_nats_per_full_token'], 0)
        gain = sum(off[x]['contribution_nats_per_full_token'] - correct[x]['contribution_nats_per_full_token']
                   for x in ('format', 'numeric', 'mixed'))
        self.assertAlmostEqual(gain, off['full']['nll_nats_per_token'] - correct['full']['nll_nats_per_token'], places=12)


if __name__ == '__main__':
    unittest.main()

"""CPU fixtures for the isolated post-selection test teacher-forcing helper."""
import copy
import unittest

import torch

from src.mm_data import answer_text, collate, prompt_text
from src.mm_data_evaluation import collate_evaluation


class Tokenizer:
    eos_token_id, pad_token_id = 2, 0

    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(m['content'] for m in messages) + '\nassistant:'

    def encode(self, text, **kwargs):
        return [3 + ord(c) % 100 for c in text]


class Store:
    def pooled(self, sequence):
        return {'H15': torch.ones(8, 640), 'H30': torch.full((8, 640), 2.)}


def test_row(pkd=7.1234):
    return {'sample_id': 'test-a', 'split': 'test', 'pkd': pkd,
        'peptide_sequence': 'ACD', 'target_sequence': 'EFGH',
        'target_sequence_role': 'reference_unverified', 'chemical_state': {},
        'experimental_context': {'assay': 'binding', 'actual_construct_verified': False}}


class EvaluationDataTests(unittest.TestCase):
    def test_default_training_collator_still_rejects_supervised_test(self):
        with self.assertRaises(ValueError):
            collate([test_row()], Tokenizer(), Store(), supervised=True)

    def test_phase_is_explicit_and_only_test_rows_are_accepted(self):
        with self.assertRaises(TypeError):
            collate_evaluation([test_row()], Tokenizer(), Store())
        with self.assertRaises(ValueError):
            collate_evaluation([test_row()], Tokenizer(), Store(), phase='fit')
        train = test_row(); train['split'] = 'train'
        with self.assertRaises(ValueError):
            collate_evaluation([train], Tokenizer(), Store(), phase='evaluate')
        for bad in (None, True, float('inf')):
            with self.assertRaises(ValueError):
                collate_evaluation([test_row(bad)], Tokenizer(), Store(), phase='evaluate')

    def test_exact_original_answer_eos_leftpadding_and_source_unchanged(self):
        rows = [test_row(), test_row(-100.25)]
        rows[1]['sample_id'] = 'test-b'
        rows[1]['experimental_context']['assay'] = 'longer binding assay description'
        snapshot = copy.deepcopy(rows)
        tokenizer = Tokenizer()
        batch = collate_evaluation(rows, tokenizer, Store(), phase='evaluate')
        self.assertEqual(rows, snapshot)
        self.assertEqual(set(batch), {'input_ids', 'attention_mask', 'labels', 'prompt_end_index', 'memory_inputs'})
        for i, row in enumerate(rows):
            prefix = tokenizer.encode(prompt_text(row, tokenizer))
            suffix = tokenizer.encode(answer_text(row)) + [tokenizer.eos_token_id]
            active = batch['attention_mask'][i].bool()
            self.assertEqual(batch['input_ids'][i, active].tolist(), prefix + suffix)
            boundary = batch['prompt_end_index'][i].item() + 1
            self.assertTrue((batch['labels'][i, :boundary] == -100).all())
            self.assertEqual(batch['labels'][i, boundary:].tolist(), suffix)

    def test_gold_changes_only_answer_not_prompt_memory_or_prompt_index(self):
        a = collate_evaluation([test_row(7.12)], Tokenizer(), Store(), phase='evaluate')
        b = collate_evaluation([test_row(8.34)], Tokenizer(), Store(), phase='evaluate')
        end = a['prompt_end_index'].item()
        self.assertTrue(torch.equal(a['prompt_end_index'], b['prompt_end_index']))
        self.assertTrue(torch.equal(a['input_ids'][:, :end + 1], b['input_ids'][:, :end + 1]))
        self.assertFalse(torch.equal(a['labels'], b['labels']))
        for key in a['memory_inputs']:
            self.assertTrue(torch.equal(a['memory_inputs'][key], b['memory_inputs'][key]))


if __name__ == '__main__':
    unittest.main()

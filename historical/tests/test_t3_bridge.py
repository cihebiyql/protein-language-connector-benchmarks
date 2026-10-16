"""CPU-only verification of real-pair scheduling, packing and bridge ablations."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from real_data import epoch_order
from run_t3_mlp import flip_training_pair
from train_t3_bridge import (accumulate_task, apply_training_flips, evaluate_panel,
                             pack_pairs, pair_donors, pair_prompts, swap_messages,
                             training_batches, validate_pairs)


def row(i, label='A', split='train', a='AC', b='DEF', target='target'):
    return {'pair_id': f'p{i}', 'group_id': f'g{i}', 'sequence_a': a,
            'sequence_b': b, 'target_id': target, 'label': label, 'split': split}


class TinyScorer(nn.Module):
    """A differentiable 2-label scorer; does not load or allocate either model."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.25))
        self.seen_targets = []

    def encode(self, task, slots, target_ids=None):
        self.seen_targets.append(target_ids)
        return [1, 2, 3] + [0] * slots, 3

    def forward(self, features, masks, prompts, start, message_mode='correct', full_vocab=False):
        if features.shape[1] == 128:
            values = (features[:, :64, 0] * masks[:, :64]).sum(-1)
            values -= (features[:, 64:, 0] * masks[:, 64:]).sum(-1)
        else:
            values = (features[:, :, 0] * masks).sum(-1)
        values = values * self.weight
        if message_mode == 'off':
            values = values * 0
        return torch.stack((values, -values), dim=-1)

    def soft_positive_loss(self, logits, positives):
        logs = logits.log_softmax(-1)
        return torch.stack([-logs[i, p].mean() for i, p in enumerate(positives)])


class StageBScheduleTests(unittest.TestCase):
    def test_dynamic_34_pairs_have_same_flip_replay_and_partial_tail_as_mlp(self):
        rows = [row(i) for i in range(34)] + [row(34, split='val'), row(35, split='test')]
        d1 = list(range(500, 500 + 1914))
        batches = training_batches(rows, d1, seed=17, epoch=3)
        self.assertEqual([len(batch['pairs']) for batch in batches], [32, 2])
        self.assertEqual([len(batch['replay']) for batch in batches], [32, 2])
        pairs = [i for batch in batches for i in batch['pairs']]
        replay = [i for batch in batches for i in batch['replay']]
        self.assertEqual(pairs, epoch_order(list(range(34)), 17, 3).tolist())
        self.assertEqual(replay, epoch_order(d1, 100017, 3).tolist()[:34])
        self.assertEqual(len(set(replay)), 34)
        for batch in batches:
            self.assertEqual(batch['flips'], [flip_training_pair(17, 3, rows[i]['pair_id']) for i in batch['pairs']])
        self.assertEqual(30 * len(batches), 60)

    def test_schedule_adapts_to_further_data_quality_exclusions(self):
        batches = training_batches([row(i) for i in range(7)], list(range(100)), 42, 1)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]['replay']), 7)
        with self.assertRaises(ValueError):
            training_batches([row(1)], [], 42, 1)

    def test_mixed_task_loss_uses_total_sample_denominator_even_partial_microbatches(self):
        features = torch.arange(1, 11, dtype=torch.float32).reshape(10, 1, 1) / 10
        masks = torch.ones(10, 1)
        positives = [[i % 2] for i in range(10)]
        actual, expected = TinyScorer(), TinyScorer()
        accumulate_task(actual, features[:5], masks[:5], [[]] * 5, 0, positives[:5], 10, 2)
        accumulate_task(actual, features[5:], masks[5:], [[]] * 5, 0, positives[5:], 10, 2)
        logits = expected(features, masks, [[]] * 10, 0)
        expected.soft_positive_loss(logits, positives).mean().backward()
        torch.testing.assert_close(actual.weight.grad, expected.weight.grad)


class ResiduePackingTests(unittest.TestCase):
    def test_full_g1_peptide_and_zero_padding_masks_are_preserved(self):
        sequences = ['AC', 'DEF', 'ACDEFGHIKLMNPQRSTV']
        cache = {'sequences': sequences, 'features15': torch.zeros(3, 64, 2), 'mask': torch.zeros(3, 64)}
        for i, seq in enumerate(sequences):
            cache['features15'][i, :len(seq)] = i + 1
            cache['mask'][i, :len(seq)] = 1
        rows = [row(0), row(1, a=sequences[2])]
        features, masks = pack_pairs(rows, cache, 15)
        self.assertEqual(tuple(features.shape), (2, 128, 2))
        self.assertEqual(masks.sum(-1).tolist(), [5, 21])
        self.assertEqual(masks[1, :64].sum().item(), 18)
        self.assertEqual(masks[:, 64:].sum(-1).tolist(), [3, 3])
        self.assertTrue(torch.all(masks[:, 18:64] == 0))
        cache['mask'][0, 3] = 1
        with self.assertRaisesRegex(ValueError, 'Cached residue mask'):
            pack_pairs(rows, cache, 15)

    def test_training_flip_exchanges_both_residue_blocks_and_labels(self):
        features = torch.zeros(2, 128, 1)
        features[:, :64] = 1
        features[:, 64:] = 2
        masks = torch.zeros(2, 128)
        masks[:, :2] = 1
        masks[:, 64:67] = 1
        labels = torch.tensor([0, 1])
        f, m, y = apply_training_flips(features, masks, labels, [True, False])
        self.assertEqual(y.tolist(), [1, 1])
        self.assertEqual(f[0, 0, 0].item(), 2)
        self.assertEqual(m[0, :64].sum().item(), 3)
        torch.testing.assert_close(f[1], features[1])
        torch.testing.assert_close(m[1], masks[1])
        ff, mm = swap_messages(*swap_messages(features, masks))
        torch.testing.assert_close(ff, features)
        torch.testing.assert_close(mm, masks)

    def test_prompt_receives_only_opaque_target_aliases(self):
        model = TinyScorer()
        rows = [row(0, target='raw_gene_SH3'), row(1, target='raw_other_gene')]
        pair_prompts(model, rows, {'raw_gene_SH3': 'D000', 'raw_other_gene': 'D001'})
        self.assertEqual(model.seen_targets, ['D000', 'D001'])


class InterventionTests(unittest.TestCase):
    def test_small_realistic_panel_uses_distinct_same_target_donors_without_filling_to_ten(self):
        rows = [row(i, label='AB'[i % 2], split='test') for i in range(6)]
        rows[5]['target_id'] = 'different_target'
        assignment = pair_donors(rows)
        self.assertEqual(assignment[5], {'indices': [], 'eligible': 0})
        for i, selected in enumerate(assignment):
            self.assertEqual(len(selected['indices']), len(set(selected['indices'])))
            self.assertLessEqual(len(selected['indices']), 3)
            for donor in selected['indices']:
                self.assertEqual(rows[i]['target_id'], rows[donor]['target_id'])
                self.assertNotEqual(rows[i]['label'], rows[donor]['label'])
                self.assertNotEqual(rows[i]['group_id'], rows[donor]['group_id'])
        with self.assertRaisesRegex(ValueError, 'held-out'):
            pair_donors([row(0)])

    def test_correct_off_swap_and_reversed_label_are_actual_distinct_forward_calls(self):
        rows = [row(0, label='A', split='test'), row(1, label='B', split='test')]
        features = torch.zeros(2, 128, 1)
        masks = torch.zeros(2, 128)
        features[0, 0, 0] = 2
        features[1, 64, 0] = 2
        masks[:, 0] = masks[:, 64] = 1
        metrics, records, logits = evaluate_panel(TinyScorer(), rows, features, masks, {'target': 'D000'}, 2)
        self.assertEqual(metrics['correct']['accuracy'], 1.)
        self.assertEqual(metrics['off']['accuracy'], .5)
        self.assertEqual(metrics['swapped_messages']['accuracy'], 0.)
        self.assertEqual(metrics['reversed_input_and_label']['accuracy'], 1.)
        self.assertEqual(metrics['prediction_reversal_rate'], 1.)
        np.testing.assert_array_equal(logits['off'], np.zeros((2, 2)))
        self.assertEqual([r['label'] for r in records], ['A', 'B'])
        self.assertEqual(metrics['mismatched_coverage']['used_counts'], [1, 1])

    def test_invalid_pair_identity_cannot_enter_cache_or_training(self):
        with self.assertRaisesRegex(ValueError, 'Unknown pair label'):
            validate_pairs([row(0, label='unknown')], {'target': 'D000'})
        with self.assertRaisesRegex(ValueError, 'standard residues'):
            validate_pairs([row(0, a='ACX')], {'target': 'D000'})


class FinalDataPreflightTests(unittest.TestCase):
    def test_frozen_real_inputs_have_valid_opaque_catalog_and_fixed_disjoint_splits(self):
        d2 = [json.loads(line) for line in (ROOT / 'data/processed/t3_d2_pairs.jsonl').read_text(encoding='utf-8').splitlines()]
        g1 = [json.loads(line) for line in (ROOT / 'data/processed/t3_g1_direction_pairs.jsonl').read_text(encoding='utf-8').splitlines()]
        catalog = json.loads((ROOT / 'data/processed/t3_target_catalog.json').read_text(encoding='utf-8'))
        target_map = {entry['target_id']: entry['opaque_id'] for entry in catalog['entries']}
        validate_pairs(d2, target_map)
        validate_pairs(g1, target_map)
        self.assertEqual({s: sum(r['split'] == s for r in d2) for s in ('train', 'val', 'test')},
                         {'train': 34, 'val': 16, 'test': 9})
        self.assertEqual(len(g1), 17)
        self.assertEqual(sum(r['sequence_source_consistent'] for r in g1), 14)
        self.assertEqual(len(set(target_map.values())), len(target_map))
        seen = {}
        for r in d2:
            self.assertEqual(seen.setdefault(r['group_id'], r['split']), r['split'])


if __name__ == '__main__':
    unittest.main()

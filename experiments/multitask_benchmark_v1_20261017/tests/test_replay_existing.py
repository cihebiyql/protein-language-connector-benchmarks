"""Integrity regressions plus read-only replay of actual stored predictions."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parents[1] / 'src/replay_existing.py'
spec = importlib.util.spec_from_file_location('replay_existing', SOURCE)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        replay.initialize(ROOT, replay.Ledger(ROOT))

    def setUp(self):
        self.gold = [{'sample_id': 'x', 'positive': ['A']}, {'sample_id': 'y', 'positive': ['B']}]
        self.pred = [{'sample_id': 'x', 'ranking': ['A', 'B'], 'label': ['A']},
                     {'sample_id': 'y', 'ranking': ['B', 'A'], 'label': ['B']}]

    def score(self, pred=None, gold=None):
        return replay.ranking_score(gold or self.gold, pred if pred is not None else self.pred,
                                    ['A', 'B'], 'positive', 'ranking', 'label')

    def test_id_reordering_is_lossless(self):
        self.assertEqual(self.score(self.pred[::-1])['mrr'], 1.0)

    def test_missing_id_rejected(self):
        with self.assertRaisesRegex(replay.IntegrityError, 'coverage'):
            self.score(self.pred[:1])

    def test_extra_unknown_id_rejected(self):
        with self.assertRaisesRegex(replay.IntegrityError, 'coverage'):
            self.score(self.pred + [{'sample_id': 'z'}])

    def test_duplicate_prediction_id_rejected(self):
        with self.assertRaisesRegex(replay.IntegrityError, 'duplicate prediction'):
            self.score(self.pred + self.pred[:1])

    def test_duplicate_gold_id_rejected(self):
        with self.assertRaisesRegex(replay.IntegrityError, 'duplicate gold'):
            self.score(gold=self.gold + self.gold[:1])

    def test_incomplete_candidate_set_rejected(self):
        pred = copy.deepcopy(self.pred); pred[0]['ranking'] = ['A']
        with self.assertRaisesRegex(replay.IntegrityError, 'candidate'):
            self.score(pred)

    def test_duplicate_candidate_rejected(self):
        pred = copy.deepcopy(self.pred); pred[0]['ranking'] = ['A', 'A']
        with self.assertRaisesRegex(replay.IntegrityError, 'candidate'):
            self.score(pred)

    def test_wrong_candidate_identity_rejected(self):
        pred = copy.deepcopy(self.pred); pred[0]['ranking'] = ['A', 'C']
        with self.assertRaisesRegex(replay.IntegrityError, 'candidate'):
            self.score(pred)

    def test_wrong_prediction_gold_rejected(self):
        pred = copy.deepcopy(self.pred); pred[0]['label'] = ['B']
        with self.assertRaisesRegex(replay.IntegrityError, 'gold/label'):
            self.score(pred)

    def test_unknown_gold_class_rejected(self):
        gold = copy.deepcopy(self.gold); gold[0]['positive'] = ['Z']
        with self.assertRaisesRegex(replay.IntegrityError, 'gold positive'):
            self.score(gold=gold)

    def test_unknown_pair_never_negative(self):
        # Ranking scores only documented positives; no binary false-negative denominator.
        score = self.score()
        self.assertNotIn('auroc', score)
        self.assertEqual(score['n'], 2)

    def test_wrong_direction_gold_and_logit_map_rejected(self):
        gold = [{'pair_id': 'x', 'label': 'A', 'target_id': 'T', 'group_id': 'g'}]
        pred = [dict(gold[0], logits={'correct': [2, 1]}, predictions={'correct': 'A'})]
        self.assertEqual(replay.direction_score(gold, pred, 'correct')['accuracy'], 1.0)
        self.assertIsNone(replay.direction_score(gold, pred, 'correct')['balanced_accuracy'])
        pred[0]['predictions']['correct'] = 'B'
        with self.assertRaisesRegex(replay.IntegrityError, 'label/logit'):
            replay.direction_score(gold, pred, 'correct')
        pred[0]['label'] = 'B'
        with self.assertRaisesRegex(replay.IntegrityError, 'gold/target'):
            replay.direction_score(gold, pred, 'correct')

    def test_actual_read_only_four_panel_replay(self):
        # Temporary output must remain under the project so manifest paths stay portable.
        base = SOURCE.parents[1] / 'results/existing_replay_v1'
        base.mkdir(parents=True, exist_ok=True)
        frozen = [ROOT / 'data/processed/benz_samples.jsonl',
                  ROOT / 'data/processed/bench_v02/b1_homology_locked/test_gold.jsonl',
                  ROOT / 'data/processed/foundation_20261016/rt_target_diagnostic/frozen_protocol.json',
                  ROOT / 'data/processed/t3_d2_pairs.jsonl']
        before = {str(p): replay.sha(p) for p in frozen}
        with tempfile.TemporaryDirectory(prefix='test_', dir=base) as tmp:
            result = replay.run(ROOT, Path(tmp))
            self.assertTrue(result['historical_replay'])
            self.assertFalse(result['current_V4_M0_M3_multitask_training'])
            self.assertEqual({k: v['status'] for k, v in result['panels'].items()},
                             dict.fromkeys(('ELM_unseen_function', 'March_RT', 'Benz_known_positive', 'Kliche_direction'), 'complete'))
            self.assertEqual([len(v['runs']) for v in result['panels'].values()], [6, 3, 3, 27])
            elm = result['panels']['ELM_unseen_function']
            self.assertEqual(elm['population']['candidates'], 31)
            self.assertTrue(all(r['metrics']['n'] == 76 for r in elm['runs']))
            self.assertTrue(all(r['original_check']['matched'] for r in elm['runs']))
            march = result['panels']['March_RT']
            self.assertEqual(march['population']['selection'], 525)
            self.assertEqual(march['population']['calibration'], 530)
            self.assertTrue(all(r['metrics']['n'] == 1063 and r['original_check']['matched'] for r in march['runs']))
            self.assertTrue(all(r['metrics']['n'] == 387 for r in result['panels']['Benz_known_positive']['runs']))
            kliche = result['panels']['Kliche_direction']['runs']
            self.assertEqual({(r['subset'], r['n']) for r in kliche},
                             {('d2_test', 9), ('d2_test_clean_after_t1_pretraining', 5), ('g1_source_consistent_direction', 14)})
            manifest = json.loads((Path(tmp) / 'manifest.json').read_text('utf8'))
            self.assertTrue(manifest['inputs_read_only'])
            self.assertTrue(all(not Path(f['path']).is_absolute() for f in manifest['files']))
            for f in manifest['files']:
                self.assertEqual(replay.sha(ROOT / f['path']), f['sha256'])
        self.assertEqual(before, {str(p): replay.sha(p) for p in frozen})


if __name__ == '__main__':
    unittest.main()

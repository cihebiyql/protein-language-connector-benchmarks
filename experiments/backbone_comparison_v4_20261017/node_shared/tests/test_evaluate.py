"""CPU/stdlib fixtures only. No fixture is a scientific model result."""
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stderr, redirect_stdout

SCOPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCOPE/'src'))
import mm_evaluate as ev


def gold_rows(split='test'):
    return [{'sample_id': name, 'split': split, 'pkd': value,
             'groups': {'target_sequence_sha256': target}, 'group_id': target,
             'experimental_context': {'assay': 'fixture'}}
            for name, value, target in [('a', 1., 'T1'), ('b', 3., 'T1'), ('c', 5., 'T2')]]


def lm_rows(rows=None):
    rows = gold_rows() if rows is None else rows
    return [{'sample_id': r['sample_id'], 'split': r['split'], 'gold_pkd': r['pkd'],
             'raw_text': json.dumps({'pKD': r['pkd']})} for r in rows]


def numeric_rows(rows=None):
    rows = gold_rows() if rows is None else rows
    return [{'sample_id': r['sample_id'], 'split': r['split'], 'gold_pkd': r['pkd'],
             'prediction_pkd': r['pkd'], 'error': None} for r in rows]


def protocol():
    return ev.read_json(SCOPE/'docs/evaluation_protocol.json')


class PureEvaluationTests(unittest.TestCase):
    def test_strict_parser_positive(self):
        for raw in ('{"pKD":1.2}', ' {"pKD":-3} ', '{"pKD":1e2}'):
            with self.subTest(raw=raw):
                self.assertTrue(ev.parse_answer(raw)['valid'])

    def test_strict_parser_negative(self):
        for raw in ('{"pKD":true}', '{"pKD":"1"}', '{"pKD":NaN}', '{"pKD":Infinity}',
                    '{"pKD":1e309}', '{"pKD":1,"pKD":2}', '{"pKD":1,"x":0}',
                    '```json\n{"pKD":1}\n```', '{"pKD":1} text', '[]', '{"pkd":1}', None):
            with self.subTest(raw=raw):
                self.assertFalse(ev.parse_answer(raw)['valid'])

    def test_no_clipping_or_two_decimal_scoring(self):
        rows = gold_rows()[:1]; rows[0]['pkd'] = 1.23456789
        pred = lm_rows(rows); pred[0]['raw_text'] = '{"pKD":-100}'
        self.assertEqual(ev.generation_metrics(rows, pred)['full_sample_mae'], 101.23456789)

    def test_bool_and_huge_integer_not_finite(self):
        self.assertFalse(ev.finite(True))
        self.assertFalse(ev.finite(10**1000))

    def test_finite_extreme_prediction_not_clipped_or_lost_to_intermediate_overflow(self):
        score = ev.regression(gold_rows(), [1e308]*3)
        self.assertEqual(score['mae'], 1e308)
        self.assertEqual(score['rmse'], 1e308)

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ev.EvaluationError):
            ev.strict_loads('{"a":1,"a":2}')

    def test_overflow_exponent_in_unused_metadata_rejected_but_raw_reason_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t)/'input.json'
            path.write_text('{"runtime":{"value":1e309}}', encoding='utf8')
            with self.assertRaises(ev.EvaluationError): ev.read_json(path)
        self.assertEqual(ev.parse_answer('{"pKD":1e309}')['reason'], 'nonfinite_or_nonnumeric_pKD')

    def test_metric_denominators_and_rank_ties(self):
        r = ev.regression(gold_rows(), [1., 1., 5.])
        self.assertAlmostEqual(r['mae'], 2/3)
        self.assertAlmostEqual(r['rmse'], math.sqrt(4/3))
        self.assertEqual(r['target_macro_mae'], .5)
        self.assertEqual(r['n_targets'], 2)
        self.assertAlmostEqual(r['spearman'], math.sqrt(3)/2)

    def test_constant_and_singleton_spearman_undefined(self):
        self.assertIsNone(ev.spearman([1, 2], [3, 3]))
        self.assertIsNone(ev.spearman([1], [3]))
        self.assertIsNone(ev.spearman([], []))

    def test_generation_failure_full_population_null(self):
        pred = lm_rows(); pred[1]['raw_text'] = 'invalid'
        score = ev.generation_metrics(gold_rows(), pred)
        self.assertIsNone(score['full_sample_mae'])
        self.assertEqual((score['n_total'], score['n_valid'], score['n_failed']), (3, 2, 1))
        self.assertEqual(score['valid_only_denominator'], 2)

    def test_all_generation_failures_remain_counted(self):
        pred = lm_rows()
        for row in pred: row['raw_text'] = 'invalid'
        score = ev.generation_metrics(gold_rows(), pred)
        self.assertEqual(score['n_failed'], 3)
        self.assertEqual(score['parse_valid_rate'], 0.)
        self.assertIsNone(score['valid_only_mae'])

    def test_real_generation_error_requires_null_or_invalid_text(self):
        pred = lm_rows(); pred[0].update(raw_text=None, generation_error='fixture_forward_error')
        self.assertEqual(ev.generation_metrics(gold_rows(), pred)['n_failed'], 1)
        pred[0]['raw_text'] = '{"pKD":1}'
        with self.assertRaises(ev.EvaluationError): ev.generation_metrics(gold_rows(), pred)

    def test_forged_parse_rejected(self):
        pred = lm_rows(); pred[0]['parse'] = {'valid': True, 'value': 99., 'reason': None}
        with self.assertRaises(ev.EvaluationError): ev.generation_metrics(gold_rows(), pred)

    def test_missing_extra_duplicate_reordered_ID_rejected(self):
        variants = [lm_rows()[:-1], lm_rows()+[lm_rows()[0]], list(reversed(lm_rows()))]
        duplicate = lm_rows(); duplicate[1]['sample_id'] = 'a'; variants.append(duplicate)
        invalid = lm_rows(); invalid[0]['sample_id'] = []; variants.append(invalid)
        for pred in variants:
            with self.subTest(n=len(pred)), self.assertRaises(ev.EvaluationError):
                ev.generation_metrics(gold_rows(), pred)

    def test_wrong_gold_split_and_unknown_key_rejected(self):
        for key, value in [('gold_pkd', 4.), ('gold_pkd', True), ('split', 'train'), ('unexpected', 1)]:
            pred = lm_rows(); pred[0][key] = value
            with self.subTest(key=key), self.assertRaises(ev.EvaluationError):
                ev.generation_metrics(gold_rows(), pred)

    def test_token_ids_and_EOS_length_validation(self):
        for ids, length in [([-1], 1), ([True], 1), ([1]*25, 25), ([1], 2)]:
            pred = lm_rows(); pred[0].update(raw_generated_token_ids=ids, first_eos_length=length)
            with self.subTest(ids=ids), self.assertRaises(ev.EvaluationError):
                ev.generation_metrics(gold_rows(), pred)

    def test_numeric_head_independent_of_LM_parse(self):
        lm = lm_rows(); lm[0]['raw_text'] = 'invalid'
        self.assertIsNone(ev.generation_metrics(gold_rows(), lm)['full_sample_mae'])
        self.assertEqual(ev.numeric_metrics(gold_rows(), numeric_rows())['full_population']['mae'], 0.)

    def test_numeric_explicit_failure_full_metrics_null(self):
        pred = numeric_rows(); pred[1].update(prediction_pkd=None, error='nonfinite_output')
        metric = ev.numeric_metrics(gold_rows(), pred)
        self.assertIsNone(metric['full_population'])
        self.assertEqual(metric['n_finite'], 2)
        self.assertEqual(metric['n_total'], 3)

    def test_numeric_nonfinite_bool_or_error_value_rejected(self):
        for value, error in [(float('nan'), None), (float('inf'), None), (True, None), (None, None), (3., 'error')]:
            pred = numeric_rows(); pred[0].update(prediction_pkd=value, error=error)
            with self.subTest(value=value), self.assertRaises(ev.EvaluationError):
                ev.numeric_metrics(gold_rows(), pred)

    def test_selector_token_weighted_not_mean_epoch_batch_losses(self):
        history = [{'epoch': 1, 'validation_answer_loss_sum': 10., 'validation_answer_token_count': 10},
                   {'epoch': 2, 'validation_answer_loss_sum': 9., 'validation_answer_token_count': 3}]
        self.assertEqual(ev.select_checkpoint(history)['best_epoch'], 1)

    def test_selector_earliest_exact_tie_and_no_tolerance_selection(self):
        h = [{'epoch': 1, 'validation_answer_loss_sum': 4., 'validation_answer_token_count': 2},
             {'epoch': 2, 'validation_answer_loss_sum': 8., 'validation_answer_token_count': 4}]
        self.assertEqual(ev.select_checkpoint(h)['best_epoch'], 1)
        h[1]['validation_answer_loss_sum'] -= 1e-13
        self.assertEqual(ev.select_checkpoint(h)['best_epoch'], 2)

    def test_selector_bad_history_rejected(self):
        for h in ([], [{'epoch': 1, 'validation_answer_loss_sum': 1., 'validation_answer_token_count': 0}],
                  [{'epoch': True, 'validation_answer_loss_sum': 1., 'validation_answer_token_count': 2}],
                  [None], [{'epoch': 1, 'validation_answer_loss_sum': 1., 'validation_answer_token_count': 2,
                             'validation_answer_token_nll': 99.}]):
            with self.subTest(history=h), self.assertRaises(ev.EvaluationError): ev.select_checkpoint(h)

    def test_seed_3of3_mean_sample_SD(self):
        stats = ev.seed_statistics({17: 1., 42: 2., 2026: 3.})
        self.assertEqual((stats['mean'], stats['sample_SD'], stats['coverage']), (2., 1., '3/3'))

    def test_missing_seed_never_partial_mean(self):
        stats = ev.seed_statistics({'17': 1., '42': None, '2026': 3.})
        self.assertEqual(stats['coverage'], '2/3')
        self.assertIsNone(stats['mean']); self.assertIsNone(stats['sample_SD'])
        with self.assertRaises(ev.EvaluationError): ev.seed_statistics({'17': 1., '42': 2.})

    def test_C1_row_mean_and_numeric_direction(self):
        rows = gold_rows()[:2]; records = []
        for row, n, delta in zip(rows, [2, 4], [1., 3.]):
            records.append({'sample_id': row['sample_id'], 'answer_tokens': n,
                'correct': {'loss_sum': 2*n, 'token_count': n}, 'off': {'loss_sum': (2+delta)*n, 'token_count': n},
                'numeric_correct_pkd': row['pkd'], 'numeric_off_pkd': row['pkd']+delta})
        score = ev.c1_metrics(rows, records)
        self.assertEqual(score['G_off'], 2.)
        self.assertNotEqual(score['G_off'], 14/6)
        self.assertEqual(score['numeric_off_minus_on_MAE'], 2.)
        self.assertIsNone(score['G_train_mean']); self.assertEqual(score['train_mean_status'], 'not_run')
        self.assertIsNone(score['P']); self.assertIsNone(score['H'])

    def test_C1_mean_requires_TRAIN_same_theta(self):
        rows = gold_rows()[:1]
        record = {'sample_id': 'a', 'answer_tokens': 2, 'correct': {'loss_sum': 2., 'token_count': 2},
            'off': {'loss_sum': 4., 'token_count': 2}, 'train_mean': {'loss_sum': 6., 'token_count': 2},
            'checkpoint_sha256': 'a'*64, 'neutral_checkpoint_sha256': 'a'*64,
            'neutral_statistic_sha256': 'b'*64, 'neutral_fit_split': 'train'}
        self.assertEqual(ev.c1_metrics(rows, [record])['G_train_mean'], 2.)
        for key, value in [('neutral_fit_split', 'test'), ('neutral_checkpoint_sha256', 'c'*64)]:
            bad = deepcopy(record); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ev.EvaluationError): ev.c1_metrics(rows, [bad])

    def test_C1_token_denominator_and_partial_mode_rejected(self):
        rows = gold_rows()[:2]
        records = [{'sample_id': r['sample_id'], 'answer_tokens': 2,
            'correct': {'loss_sum': 2., 'token_count': 2}, 'off': {'loss_sum': 4., 'token_count': 2}} for r in rows]
        records[0]['off']['token_count'] = 3
        with self.assertRaises(ev.EvaluationError): ev.c1_metrics(rows, records)
        records[0]['off']['token_count'] = 2; records[0]['numeric_correct_pkd'] = 1.
        with self.assertRaises(ev.EvaluationError): ev.c1_metrics(rows, records)

    def test_missing_conditions_report_empty_observed_strata(self):
        result = ev.missing_condition_counts(gold_rows())
        self.assertEqual(result['ph']['n_observed'], 0)
        self.assertEqual(result['ph']['comparison_status'], 'undefined_empty_stratum')

    def test_protocol_UNFROZEN_blocks_formal_only(self):
        p = protocol(); p['evaluator_source_sha256'] = ev.file_sha(ev.__file__); p['training_budget']['status'] = 'UNFROZEN'
        self.assertTrue(ev.validate_protocol(p, formal=False))
        with self.assertRaises(ev.EvaluationError): ev.validate_protocol(p, formal=True)

    def test_protocol_no_mutable_data_matrix_or_condition_contract(self):
        for key in ('data_sha256', 'split_counts', 'arms', 'condition_slots', 'generation', 'checkpoint_selector'):
            p = protocol(); p[key] = None
            with self.subTest(key=key), self.assertRaises(ev.EvaluationError): ev.validate_protocol(p)

    def test_generation_batch_positive_integer_only(self):
        for value in (0, -1, True, 1.5, '8', None):
            p = protocol(); p['generation']['batch_size'] = value
            with self.subTest(value=value), self.assertRaises(ev.EvaluationError): ev.validate_protocol(p)

    def test_real_loader_rejects_wrong_frozen_SHA(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); base = root/ev.SCENARIOS[0]; base.mkdir()
            (base/'samples.jsonl').write_text('{}\n', encoding='utf8')
            with self.assertRaises(ev.EvaluationError): ev.load_locked_scenario(root, ev.SCENARIOS[0])

    def test_summary_missing_and_development_not_formal(self):
        result = ev.summarize_runs([], 'a'*64)
        self.assertEqual((result['n_expected'], result['n_formal_complete']), (24, 0))
        self.assertTrue(all(r['metrics']['LM_full_sample_mae']['coverage'] == '0/3' for r in result['configurations']))

    def test_summary_three_seeds_and_paired_coverage(self):
        results = []
        for seed in ev.SEEDS:
            for arm in ('M0', 'M3'):
                results.append({'scope_id': ev.SCOPE_ID, 'protocol_sha256': 'a'*64,
                    'evaluation_protocol_sha256': 'b'*64, 'backbone': ev.BACKBONES[0], 'scenario': ev.SCENARIOS[0], 'arm': arm, 'seed': seed,
                    'status': 'complete', 'phase': 'formal', 'formal_eligible': True,
                    'LM': {'full_sample_mae': 2. if arm == 'M0' else 1.},
                    'numeric': {'full_population': {'mae': 1., 'rmse': 2., 'spearman': None, 'target_macro_mae': 1.}}})
        summary = ev.summarize_runs(results, 'a'*64)
        contrast = next(c for c in summary['registered_M0_contrasts'] if c['arm'] == 'M3' and c['scenario'] == ev.SCENARIOS[0] and c['metric'] == 'LM_full_sample_mae')
        self.assertEqual((contrast['mean'], contrast['coverage']), (1., '3/3'))
        results[0]['phase'] = 'development'; results[0]['formal_eligible'] = False
        summary = ev.summarize_runs(results, 'a'*64)
        contrast = next(c for c in summary['registered_M0_contrasts'] if c['arm'] == 'M3' and c['scenario'] == ev.SCENARIOS[0] and c['metric'] == 'LM_full_sample_mae')
        self.assertEqual(contrast['coverage'], '2/3'); self.assertIsNone(contrast['mean'])

    def test_interaction_uses_all_four_same_seed_runs_and_no_partial_mean(self):
        results = []
        for seed in ev.SEEDS:
            for bi,b in enumerate(ev.BACKBONES):
                for arm in ev.ARMS:
                    error = (2. if arm == 'M0' else (1.5 if bi == 0 else 1.))
                    results.append({'scope_id': ev.SCOPE_ID, 'protocol_sha256': 'a'*64,
                        'evaluation_protocol_sha256': 'b'*64, 'backbone': b, 'scenario': ev.SCENARIOS[0],
                        'arm': arm, 'seed': seed, 'status': 'complete', 'phase': 'formal', 'formal_eligible': True,
                        'LM': {'full_sample_mae': error, 'parse_valid_rate': 1.},
                        'numeric': {'full_population': {'mae': error}}})
        summary = ev.summarize_runs(results, 'a'*64)
        interaction = next(x for x in summary['backbone_connector_interactions'] if x['scenario'] == ev.SCENARIOS[0] and x['metric'] == 'LM_full_sample_mae')
        self.assertEqual((interaction['mean'],interaction['coverage']),(.5,'3/3'))
        results[0]['status']='failed';results[0]['formal_eligible']=False
        summary = ev.summarize_runs(results,'a'*64)
        interaction = next(x for x in summary['backbone_connector_interactions'] if x['scenario'] == ev.SCENARIOS[0] and x['metric'] == 'LM_full_sample_mae')
        self.assertEqual(interaction['coverage'],'2/3');self.assertIsNone(interaction['mean'])
        self.assertEqual(summary['n_expected'],24)
        self.assertFalse(summary['cross_backbone_parameter_FLOPs_matched'])

    def test_summary_rejects_duplicate_or_mixed_revision(self):
        row = {'scope_id': ev.SCOPE_ID, 'protocol_sha256': 'a'*64, 'evaluation_protocol_sha256': 'b'*64, 'backbone': ev.BACKBONES[0],
               'scenario': ev.SCENARIOS[0], 'arm': 'M0', 'seed': 17, 'phase': 'development', 'status': 'partial'}
        with self.assertRaises(ev.EvaluationError): ev.summarize_runs([row, row], 'a'*64)
        wrong = deepcopy(row); wrong['seed'] = 42; wrong['evaluation_protocol_sha256'] = 'c'*64
        with self.assertRaises(ev.EvaluationError): ev.summarize_runs([row, wrong], 'a'*64)


class RunArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        (self.root/'docs').mkdir(); (self.root/'configs').mkdir(); self.job = self.root/'job'; self.job.mkdir()
        self.protocol = protocol();
        for spec in self.protocol['backbones'].values():
            spec['revision'] = 'd'*40; spec['model_manifest_path'] = 'fixture_manifest.json'
        self.protocol['producer_source_sha256'] = 'c'*64; self.protocol['evaluator_source_sha256'] = ev.file_sha(ev.__file__); self.protocol['C1_required_for_formal'] = False
        self.protocol['training_budget']['status'] = 'UNFROZEN'
        self.protocol['training_budget']['config_sha256'] = None
        self.protocol_path = self.root/'docs/evaluation_protocol.json'
        self.rows = gold_rows(); self.splits = {'train': gold_rows('train'), 'validation': gold_rows('validation'), 'test': self.rows}
        self.checkpoint = self.job/'selected.bin'; self.checkpoint.write_bytes(b'CPU fixture only: not model weights')
        self.history = [{'epoch': 1, 'validation_answer_loss_sum': 6., 'validation_answer_token_count': 3,
            'validation_rows': 3, 'validation_ordered_ids_sha256': ev.ordered_ids_sha(self.splits['validation'])}]
        self.selection = {'best_epoch': 1, 'validation_answer_token_nll': 2.,
            'checkpoint_sha256': ev.file_sha(self.checkpoint), 'test_labels_used_for_selection': False, 'written_before_test': True}
        self.write('history.json', self.history); self.write('selection.json', self.selection)
        self.write('lm.jsonl', lm_rows(), lines=True); self.write('numeric.jsonl', numeric_rows(), lines=True)
        self.run = {'schema_version': 1, 'scope_id': ev.SCOPE_ID, 'run_id': 'CPU_fixture_only', 'phase': 'development',
            'status': 'complete', 'scenario': ev.SCENARIOS[0], 'arm': 'M0', 'seed': 17,
            'protocol_sha256': 'a'*64, 'training_budget_sha256': 'b'*64,
            'data_sha256': ev.DATA_SHA256[ev.SCENARIOS[0]], 'test_labels_used_for_fit': False,
            'backbone': ev.BACKBONES[0], 'backbone_config_sha256': ev.canonical_sha(self.protocol['backbones'][ev.BACKBONES[0]]),
            'producer_source_sha256': 'c'*64,
            'training': {'epochs_completed': 1, 'optimizer_steps': 1, 'examples_seen': 3},
            'artifacts': {'history': self.ref('history.json'), 'selection': self.ref('selection.json'),
                'checkpoint': self.ref('selected.bin'), 'lm_predictions': self.ref('lm.jsonl'),
                'numeric_predictions': self.ref('numeric.jsonl'), 'c1': None}}
        self.run_path = self.job/'run.json'
        self.save()
        self.loader = patch.object(ev, 'load_locked_scenario', return_value=self.splits); self.loader.start()

    def tearDown(self):
        self.loader.stop(); self.temp.cleanup()

    def write(self, name, data, lines=False):
        text = ''.join(json.dumps(x, allow_nan=False)+'\n' for x in data) if lines else json.dumps(data, allow_nan=False)
        (self.job/name).write_text(text, encoding='utf8')

    def ref(self, name):
        return {'path': name, 'sha256': ev.file_sha(self.job/name)}

    def save(self):
        self.protocol_path.write_text(json.dumps(self.protocol), encoding='utf8')
        self.run_path.write_text(json.dumps(self.run), encoding='utf8')

    def score(self):
        self.save(); return ev.evaluate_run(self.run_path, self.protocol_path, self.root/'unused_mock_data')

    def freeze(self):
        training = {'epochs': 1, 'batch_size': 3, 'generation_batch_size': self.protocol['generation']['batch_size']}
        config = {'training': training, 'backbones': self.protocol['backbones'], 'backbone_signatures': {b: ev.canonical_sha(v) for b,v in self.protocol['backbones'].items()}}; cpath = self.root/'configs/formal.json'; cpath.write_text(json.dumps(config), encoding='utf8')
        self.protocol['training_budget'].update(status='FROZEN', training=training,
            training_budget_sha256=ev.canonical_sha(training), config_path='configs/formal.json', config_sha256=ev.file_sha(cpath),
            expected_training={s: self.run['training'].copy() for s in ev.SCENARIOS})
        self.run.update(phase='formal', protocol_sha256=ev.file_sha(cpath), training_budget_sha256=ev.canonical_sha(training))

    def test_wrong_or_relabelled_backbone_and_producer_are_rejected(self):
        original = deepcopy(self.run)
        for updates in ({'backbone': ev.BACKBONES[1]}, {'backbone':'unknown'},
                        {'backbone_config_sha256':'0'*64}, {'producer_source_sha256':'0'*64}):
            self.run = {**original, **updates}
            with self.subTest(updates=updates),self.assertRaises(ev.EvaluationError):self.score()
        self.run = original

    def test_unpinned_backbone_and_changed_hook_are_rejected_for_formal(self):
        self.freeze()
        self.protocol['backbones'][ev.BACKBONES[0]]['revision']=None
        self.run['backbone_config_sha256']=ev.canonical_sha(self.protocol['backbones'][ev.BACKBONES[0]])
        with self.assertRaisesRegex(ev.EvaluationError,'revision_unfrozen'): self.score()
        self.protocol['backbones'][ev.BACKBONES[0]]['revision']='d'*40
        self.protocol['backbones'][ev.BACKBONES[0]]['hook_layer']=17
        with self.assertRaises(ev.EvaluationError): self.score()

    def test_complete_development_scores_but_not_formal(self):
        result = self.score()
        self.assertEqual(result['LM']['full_sample_mae'], 0.)
        self.assertFalse(result['formal_eligible'])
        self.assertNotEqual(result['protocol_sha256'], result['evaluation_protocol_sha256'])

    def test_complete_formal_requires_frozen_config_and_counts(self):
        self.freeze(); self.assertTrue(self.score()['formal_eligible'])
        self.run['training']['optimizer_steps'] += 1
        with self.assertRaises(ev.EvaluationError): self.score()

    def test_formal_matching_generation_batches_1_4_8(self):
        for batch in (1, 4, 8):
            self.protocol['generation']['batch_size'] = batch
            self.freeze()
            with self.subTest(batch=batch): self.assertTrue(self.score()['formal_eligible'])

    def test_formal_generation_batch_config_mismatch_rejected(self):
        self.freeze()
        self.protocol['generation']['batch_size'] = 8
        with self.assertRaisesRegex(ev.EvaluationError, 'generation_batch_mismatch'): self.score()

    def test_bound_development_config_generation_batch_mismatch_rejected(self):
        self.freeze(); self.run['phase'] = 'development'
        self.protocol['generation']['batch_size'] = 4
        with self.assertRaisesRegex(ev.EvaluationError, 'generation_batch_mismatch'): self.score()

    def test_formal_unfrozen_rejected(self):
        self.run['phase'] = 'formal'
        with self.assertRaisesRegex(ev.EvaluationError, 'UNFROZEN'): self.score()

    def test_config_SHA_distinct_from_evalprotocol_and_actual_checked(self):
        self.freeze(); self.assertTrue(self.score()['formal_eligible'])
        (self.root/'configs/formal.json').write_text('{}', encoding='utf8')
        with self.assertRaisesRegex(ev.EvaluationError, 'SHA'): self.score()

    def test_formal_budget_signature_tampering_rejected(self):
        self.freeze(); self.protocol['training_budget']['training']['epochs'] = 2
        with self.assertRaises(ev.EvaluationError): self.score()

    def test_actual_config_training_must_match_signature_even_when_file_SHA_rebound(self):
        self.freeze()
        config_path = self.root/'configs/formal.json'
        config_path.write_text(json.dumps({'training': {'epochs': 9, 'batch_size': 3}, 'backbones': self.protocol['backbones'], 'backbone_signatures': {b: ev.canonical_sha(v) for b,v in self.protocol['backbones'].items()}}), encoding='utf8')
        self.protocol['training_budget']['config_sha256'] = ev.file_sha(config_path)
        self.run['protocol_sha256'] = ev.file_sha(config_path)
        with self.assertRaisesRegex(ev.EvaluationError, 'training_signature'): self.score()

    def test_test_fit_and_selection_flags_block(self):
        self.run['test_labels_used_for_fit'] = True
        with self.assertRaises(ev.EvaluationError): self.score()
        self.run['test_labels_used_for_fit'] = False; self.selection['written_before_test'] = False
        self.write('selection.json', self.selection); self.run['artifacts']['selection'] = self.ref('selection.json')
        with self.assertRaises(ev.EvaluationError): self.score()

    def test_history_wrong_validation_IDs_or_rows_block(self):
        self.history[0]['validation_ordered_ids_sha256'] = 'c'*64
        self.write('history.json', self.history); self.run['artifacts']['history'] = self.ref('history.json')
        with self.assertRaises(ev.EvaluationError): self.score()

    def test_checkpoint_and_artifact_tampering_block(self):
        self.checkpoint.write_bytes(b'changed')
        with self.assertRaisesRegex(ev.EvaluationError, 'SHA'): self.score()

    def test_path_escape_block(self):
        outside = self.root/'outside.json'; outside.write_text('[]', encoding='utf8')
        self.run['artifacts']['history'] = {'path': '../outside.json', 'sha256': ev.file_sha(outside)}
        with self.assertRaisesRegex(ev.EvaluationError, 'escape'): self.score()

    def test_row_different_theta_block(self):
        pred = lm_rows(); pred[0]['checkpoint_sha256'] = 'c'*64
        self.write('lm.jsonl', pred, lines=True); self.run['artifacts']['lm_predictions'] = self.ref('lm.jsonl')
        with self.assertRaisesRegex(ev.EvaluationError, 'provenance'): self.score()

    def test_partial_retains_missing_without_fabricating_metrics(self):
        self.run['status'] = 'partial'; self.write('lm.jsonl', lm_rows()[:1], lines=True)
        self.run['artifacts']['lm_predictions'] = self.ref('lm.jsonl')
        result = self.score()
        self.assertEqual(result['partial_coverage']['lm_predictions']['n_missing'], 2)
        self.assertIsNone(result['LM']); self.assertFalse(result['formal_eligible'])

    def test_required_C1_absent_rejected_then_on_off_accepted(self):
        self.freeze(); self.protocol['C1_required_for_formal'] = True
        with self.assertRaisesRegex(ev.EvaluationError, 'C1'): self.score()
        records = [{'sample_id': r['sample_id'], 'answer_tokens': 2,
            'correct': {'loss_sum': 2., 'token_count': 2}, 'off': {'loss_sum': 4., 'token_count': 2},
            'numeric_correct_pkd': r['pkd'], 'numeric_off_pkd': r['pkd']+1.,
            'checkpoint_sha256': ev.file_sha(self.checkpoint)} for r in self.rows]
        self.write('c1.jsonl', records, lines=True); self.run['artifacts']['c1'] = self.ref('c1.jsonl')
        result = self.score()
        self.assertTrue(result['formal_eligible']); self.assertIsNone(result['C1']['G_train_mean'])

    def test_formal_C1_registered_numeric_cannot_be_omitted(self):
        self.freeze(); self.protocol['C1_required_for_formal'] = True
        records = [{'sample_id': r['sample_id'], 'answer_tokens': 2,
            'correct': {'loss_sum': 2., 'token_count': 2}, 'off': {'loss_sum': 4., 'token_count': 2},
            'checkpoint_sha256': ev.file_sha(self.checkpoint)} for r in self.rows]
        self.write('c1.jsonl', records, lines=True); self.run['artifacts']['c1'] = self.ref('c1.jsonl')
        with self.assertRaisesRegex(ev.EvaluationError, 'C1_numeric'): self.score()

    def test_CLI_exclusive_output_and_no_overwrite(self):
        output = self.job/'result.json'; self.save()
        args = ['evaluate', '--run-manifest', str(self.run_path), '--protocol', str(self.protocol_path), '--output', str(output)]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(ev.main(args), 0)
            original = output.read_bytes()
            self.assertEqual(ev.main(args), 2)
        self.assertEqual(output.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()

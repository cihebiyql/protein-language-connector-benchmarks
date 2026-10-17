"""CPU-only training/selection/artifact tests; no pretrained weights or downloads."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src import mm_train as train


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 10, 16, 17, tzinfo=timezone.utc).astimezone(tz)


def protocol():
    return {'schema_version': 'mm-v3.protocol.v1', 'status': 'draft', 'budget_signature': None,
            'scenarios': ['known', 'double'], 'variants': ['M0'], 'seeds': [17],
            'deadline': '2026-10-17T10:29:40+08:00',
            'model': {'slots': 8, 'condition_slots': 10},
            'training': {'epochs': 3, 'batch_size': 2, 'grad_accumulation': 1, 'lr': .0003,
                         'weight_decay': .01, 'max_grad_norm': 1., 'numeric_loss_weight': 1.,
                         'numeric_standardization': 'train_zscore', 'checkpoint_selection': train.SELECTION,
                         'max_new_tokens': 24, 'generation_batch_size': 2}}


def rows(split, count=5):
    return [{'sample_id': f'{split}:{index}', 'split': split, 'pkd': 3. + index,
             'target_sequence': 'ACD', 'groups': {'target_sequence_sha256': 'target'}} for index in range(count)]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_lm = nn.Linear(1, 1).requires_grad_(False).eval()
        self.connector = nn.Linear(1, 5)
        self.numeric_head = nn.Linear(1, 1)
        self.grad_modes = []

    def train(self, mode=True):
        super().train(mode)
        self.base_lm.eval()
        return self

    def forward(self, input_ids, attention_mask, prompt_end_index, memory_inputs, off=False):
        self.grad_modes.append(torch.is_grad_enabled())
        x = input_ids.float().unsqueeze(-1) / 10
        hidden = self.base_lm(x)
        logits = self.connector(hidden)
        anchor = hidden[torch.arange(len(input_ids)), prompt_end_index]
        return {'logits': logits, 'numeric_z': self.numeric_head(anchor).reshape(-1)}

    def trainable_state_dict(self):
        return {key: value for key, value in self.state_dict().items() if not key.startswith('base_lm.')}

    def load_trainable_state_dict(self, state):
        own = self.state_dict()
        own.update(state)
        self.load_state_dict(own)

    def diagnostics(self, reset=False):
        return {'gate': 0., 'routes': [1., 0.]}


class ToyTokenizer:
    eos_token_id = 4
    pad_token_id = 0

    def decode(self, tokens, skip_special_tokens=True):
        return '{"pKD":5}' if tokens else ''


def collate(rows, supervised=True, **_kwargs):
    ids = torch.tensor([[1, 2, 3] if supervised else [1, 2] for _ in rows])
    result = {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
              'prompt_end_index': torch.ones(len(rows), dtype=torch.long), 'memory_inputs': {}}
    if supervised:
        result['labels'] = torch.tensor([[-100, -100, 3] for _ in rows])
    return result


class ScriptedModel(TinyModel):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.anchors = []

    def forward(self, input_ids, attention_mask, prompt_end_index, memory_inputs, off=False):
        self.grad_modes.append(torch.is_grad_enabled())
        self.anchors.append(prompt_end_index.clone())
        logits = torch.zeros((*input_ids.shape, 5))
        # First row EOS at step1; second row emits a token then EOS.
        logits[0, -1, 4] = 10
        if len(input_ids) > 1:
            logits[1, -1, 2 if self.calls == 0 else 4] = 10
        self.calls += 1
        return {'logits': logits, 'numeric_z': torch.arange(len(input_ids), dtype=torch.float32)}


class TrainingTests(unittest.TestCase):
    def setUp(self):
        train.configure(17)
        clock = patch.object(train, 'datetime', FixedDatetime)
        clock.start()
        self.addCleanup(clock.stop)

    def test_train_only_population_zscore_and_inverse(self):
        stats = train.NumericStats.fit(rows('train', 3))
        self.assertEqual(stats.mean, 4.)
        self.assertAlmostEqual(stats.std, (2 / 3) ** .5)
        self.assertAlmostEqual(stats.inverse(stats.targets(rows('train', 3), 'cpu')[2]), 5., places=6)
        with self.assertRaises(ValueError):
            train.NumericStats.fit(rows('validation'))
        constant = [{'split': 'train', 'pkd': 2.}] * 2
        self.assertEqual(train.NumericStats.fit(constant).std, 1e-8)

    def test_formal_guard_signature_and_fixed_slots(self):
        cfg = protocol()
        with patch.object(train, 'datetime', wraps=datetime) as clock:
            clock.now.return_value = datetime.fromisoformat('2026-10-17T01:00:00+08:00')
            train.check_protocol(cfg, 'train-canary', 'known', 'M0', 17)
            with self.assertRaisesRegex(ValueError, 'frozen'):
                train.check_protocol(cfg, 'fit', 'known', 'M0', 17)
            cfg['status'] = 'frozen'; cfg['budget_signature'] = train.digest_json(cfg['training'])
            train.check_protocol(cfg, 'fit', 'known', 'M0', 17)
            cfg['training']['lr'] = .1
            with self.assertRaisesRegex(ValueError, 'signature'):
                train.check_protocol(cfg, 'fit', 'known', 'M0', 17)
            cfg['model']['slots'] = 16
            with self.assertRaisesRegex(ValueError, 'eight'):
                train.check_protocol(cfg, 'train-canary', 'known', 'M0', 17)

    def test_masked_ce_sum_count_and_frozen_gradient(self):
        model = TinyModel()
        batch = collate(rows('train', 2))
        result = model(**train.model_inputs(batch))
        loss, count = train.supervised_ce(result['logits'], batch['labels'])
        self.assertEqual(count, 2)
        loss.backward()
        self.assertIsNone(model.base_lm.weight.grad)
        self.assertGreater(model.connector.weight.grad.abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, 'no answer'):
            train.supervised_ce(result['logits'], torch.full_like(batch['labels'], -100))

    def test_canary_exact_updates_train_only_no_val_iteration(self):
        class ForbiddenValidation:
            def __iter__(self):
                raise AssertionError('canary read validation')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            model = TinyModel()
            original = model.base_lm.weight.clone()
            cfg = protocol()
            result = train.fit_model(model, rows('train'), ForbiddenValidation(), collate, cfg,
                     {'seed': 17, 'scenario': 'known'}, output, 'cpu', max_updates=4)
            self.assertEqual(result['optimizer_steps'], 4)
            self.assertFalse(result['validation_used'])
            self.assertTrue(torch.equal(original, model.base_lm.weight))
            self.assertTrue((output / 'canary.pt').exists())
            self.assertFalse((output / 'selection.json').exists())
            payload = torch.load(output / 'canary.pt', weights_only=True)
            self.assertTrue(all(not key.startswith('base_lm.') for key in payload['trainable_state']))
            self.assertTrue(payload['optimizer']['state'])
            self.assertEqual(payload['numeric_stats']['source_split'], 'train')
            self.assertIn('task_gradient_diagnostics', payload['history'][0])
            observed_steps = [record['optimizer_step_before_update'] for segment in payload['history']
                              for record in segment['task_gradient_diagnostics']]
            self.assertIn(1, observed_steps)
            self.assertIn(3, observed_steps)

    def test_full_epochs_weighted_val_earliest_tie_selected_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            model = TinyModel()
            cfg = protocol()
            # No parameter change creates an exact val tie, testing earliest epoch retention.
            cfg['training']['lr'] = 0.; cfg['training']['weight_decay'] = 0.
            result = train.fit_model(model, rows('train', 5), rows('validation', 3), collate, cfg,
                                     {'seed': 17, 'scenario': 'known'}, output, 'cpu')
            self.assertEqual(result['optimizer_steps'], 9)
            self.assertEqual(result['examples_seen'], 15)
            history = json.loads((output / 'history.json').read_text())
            self.assertEqual(len(history), 3)
            for item in history:
                self.assertEqual(item['validation_rows'], 3)
                self.assertEqual(item['validation_answer_token_count'], 3)
                self.assertEqual(item['validation_answer_token_nll'], item['validation_answer_loss_sum'] / 3)
            selected = json.loads((output / 'selection.json').read_text())
            self.assertEqual(selected['best_epoch'], 1)
            self.assertEqual(train.file_sha(output / 'epoch_1.pt'), train.file_sha(output / 'selected.pt'))
            payload = torch.load(output / 'selected.pt', weights_only=True)
            train.verify_checkpoint_identity(payload, {'seed': 17, 'scenario': 'known'})
            with self.assertRaisesRegex(ValueError, 'identity'):
                train.verify_checkpoint_identity(payload, {'seed': 17, 'scenario': 'double'})

    def test_generation_first_eos_fixed_anchor_no_grad_restored(self):
        model = ScriptedModel().train()
        stats = train.NumericStats(5., 2., 3)
        values, ledger = train.greedy_batch(model, rows('train', 2), collate, ToyTokenizer(), stats)
        self.assertEqual(values[0]['raw_generated_token_ids'], [4])
        self.assertEqual(values[1]['raw_generated_token_ids'], [2, 4])
        self.assertEqual([v['prediction_pkd'] for v in values], [5., 7.])
        self.assertTrue(model.training)
        self.assertFalse(model.base_lm.training)
        self.assertTrue(all(not value for value in model.grad_modes))
        self.assertTrue(all(torch.equal(anchor, model.anchors[0]) for anchor in model.anchors))
        self.assertEqual([step['active_generation_rows'] for step in ledger], [2, 1])
        self.assertEqual(len(ledger), 2)

    def test_weighted_validation_uses_all_answer_tokens(self):
        class VariableLoss(TinyModel):
            def forward(self, input_ids, **_kwargs):
                logits = torch.zeros(len(input_ids), 4, 5)
                logits[:, :, 3] = input_ids[:, :1].float() * 2
                return {'logits': logits}
        records = rows('validation', 2)
        def variable_collate(selected, **_kwargs):
            batch = {'input_ids': torch.tensor([[int(r['sample_id'].endswith(':1')), 1, 2, 3] for r in selected])}
            batch.update(attention_mask=torch.ones_like(batch['input_ids']),
                         prompt_end_index=torch.zeros(len(selected), dtype=torch.long), memory_inputs={})
            batch['labels'] = torch.tensor([[-100, 3, 3, 3] if r['sample_id'].endswith(':1') else [-100, -100, -100, 3] for r in selected])
            return batch
        result = train.validation_nll(VariableLoss(), records, variable_collate, 2)
        loss_one = torch.nn.functional.cross_entropy(torch.zeros(1, 5), torch.tensor([3])).item()
        loss_three = torch.nn.functional.cross_entropy(torch.tensor([[0., 0., 0., 2., 0.]]), torch.tensor([3])).item()
        self.assertEqual(result['validation_answer_token_count'], 4)
        self.assertAlmostEqual(result['validation_answer_token_nll'], (loss_one + 3 * loss_three) / 4, places=6)
        self.assertNotAlmostEqual(result['validation_answer_token_nll'], (loss_one + loss_three) / 2, places=4)

    def test_anchor_qualification_future_changes_are_detected(self):
        good = train.prompt_anchor_invariance(TinyModel(), rows('train', 2), collate)
        self.assertTrue(good['passed'])
        self.assertEqual(good['changed_future_tokens'], 2)
        class LeakingHead(TinyModel):
            def forward(self, *args, **kwargs):
                result = super().forward(*args, **kwargs)
                result['numeric_z'] = kwargs['input_ids'][:, -1].float()
                return result
        self.assertFalse(train.prompt_anchor_invariance(LeakingHead(), rows('train', 2), collate)['passed'])
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            train.prompt_anchor_invariance(TinyModel(), rows('test', 2), collate)

    def test_prediction_schema_parser_and_serialization_chain(self):
        from src.mm_evaluate import generation_metrics, numeric_metrics
        with tempfile.TemporaryDirectory() as directory:
            gold = rows('train', 2)
            refs, lm, numeric = train.write_predictions(ScriptedModel(), gold, collate, ToyTokenizer(),
                train.NumericStats(5., 2., 3), Path(directory), 'train', {}, 2, None)
            self.assertEqual(generation_metrics(gold, lm)['n_total'], 2)
            self.assertEqual(numeric_metrics(gold, numeric)['n_total'], 2)
            self.assertIn('generation_ledger', refs)
            self.assertEqual(json.loads((Path(directory) / 'train_lm_predictions.jsonl').read_text().splitlines()[0])['sample_id'], 'train:0')

    def test_c1_fixed_theta_off_nll_numeric_and_eval_capability(self):
        from src.mm_evaluate import c1_metrics
        calls = []
        def observed_collator(selected, supervised=True, evaluation_only=False):
            calls.append((supervised, evaluation_only))
            return collate(selected, supervised=supervised)
        with tempfile.TemporaryDirectory() as directory:
            model = TinyModel().train()
            state = {key: value.clone() for key, value in model.trainable_state_dict().items()}
            ref, records, seconds = train.write_c1(model, rows('test', 3), observed_collator,
                train.NumericStats(5., 2., 3), Path(directory), 'a' * 64, 2)
            metric = c1_metrics(rows('test', 3), records)
            self.assertEqual(metric['G_off'], 0.)
            self.assertIsNone(metric['G_train_mean'])
            self.assertEqual(metric['train_mean_status'], 'not_run')
            self.assertGreaterEqual(seconds, 0)
            self.assertTrue(all(torch.equal(state[key], value) for key, value in model.trainable_state_dict().items()))
            self.assertTrue(all(not mode for mode in model.grad_modes))
            self.assertTrue(model.training)
            self.assertIn((True, True), calls)
            self.assertIn((False, False), calls)

    def test_real_connector_graph_tiny_base_can_train_and_save(self):
        from tests.test_model import fixture
        model, _ = fixture()
        def molecular_collator(selected, supervised=True):
            result = collate(selected, supervised)
            memory = {key: torch.arange(len(selected) * 8 * 6, dtype=torch.float32).reshape(len(selected), 8, 6) / 50
                      for key in ('p15', 'p30', 't15', 't30')}
            memory['condition_ids'] = torch.zeros(len(selected), 10, dtype=torch.long)
            memory['observed'] = torch.zeros(len(selected), 10, dtype=torch.bool)
            result['memory_inputs'] = memory
            return result
        with tempfile.TemporaryDirectory() as directory:
            result = train.fit_model(model, rows('train', 4), None, molecular_collator, protocol(),
                {'seed': 17, 'scenario': 'known'}, Path(directory), 'cpu', max_updates=2)
            self.assertEqual(result['optimizer_steps'], 2)
            self.assertNotEqual(float(model.alpha.detach()), 0.)
            self.assertTrue(all(param.grad is None for param in model.base_lm.parameters()))
            payload = torch.load(Path(directory) / 'canary.pt', weights_only=True)
            self.assertEqual(set(payload['trainable_state']), set(model.trainable_state_dict()))

    def test_complete_canary_cli_only_opens_train_and_writes_both_scores(self):
        from src.mm_data import FROZEN_SHAS
        scenario = next(iter(FROZEN_SHAS))
        class FakeStore:
            def __init__(self, path):
                pass
            def verify(self):
                return {'manifest_sha256': 'a' * 64}
            def provenance(self):
                return self.verify()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            views = root / 'views'; views.mkdir()
            (views / 'manifest.json').write_text(json.dumps({'scenarios': {scenario: {}}}), encoding='utf-8')
            cfg = protocol(); cfg['scenarios'] = [scenario]
            cfg['training']['generation_batch_size'] = 8
            config_path = root / 'config.json'; config_path.write_text(json.dumps(cfg), encoding='utf-8')
            accessed = []
            def read_view(_views, current, split):
                self.assertEqual(current, scenario)
                accessed.append(split)
                self.assertEqual(split, 'train')
                return rows('train', 10)
            with patch('src.mm_data.read_input_view', side_effect=read_view), \
                 patch('src.mm_data.collate', side_effect=lambda selected, *_args, **kwargs: collate(selected, **kwargs)), \
                 patch('src.mm_extract.SequenceStore', FakeStore), \
                 patch.object(train, 'verify_model_assets', return_value={}), \
                 patch.object(train, 'load_components', return_value=(TinyModel(), ToyTokenizer())):
                exit_code = train.main(['--config', str(config_path), '--project-root', str(root), '--cache-root', str(root),
                    '--view-root', str(views), '--output', str(root / 'canary'), '--scenario', scenario, '--variant', 'M0',
                    '--seed', '17', '--device', 'cpu', '--phase', 'train-canary', '--max-updates', '4'])
            self.assertEqual(exit_code, 0)
            self.assertEqual(accessed, ['train'])
            manifest = json.loads((root / 'canary/run_manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['training']['optimizer_steps'], 4)
            self.assertEqual(manifest['phase'], 'development')
            self.assertTrue(manifest['artifacts']['c1'])
            metrics = json.loads((root / 'canary/train_canary_metrics.json').read_text())
            self.assertEqual(metrics['lm']['n_total'], 8)
            self.assertEqual(metrics['numeric']['n_total'], 8)
            self.assertEqual(metrics['c1']['n_test'], 8)
            qualification = json.loads((root / 'canary/train_anchor_qualification.json').read_text())
            self.assertEqual(qualification['ordered_sample_ids'], [f'train:{i}' for i in range(8)])
            ledger = json.loads((root / 'canary/train_generation_ledger.jsonl').read_text().splitlines()[0])
            self.assertEqual(ledger['forward_steps'][0]['batch_rows'], 8)

    def test_formal_fit_then_evaluate_checkpoint_and_test_open_order(self):
        from src.mm_data import FROZEN_SHAS
        scenario = next(iter(FROZEN_SHAS))
        class FakeStore:
            def __init__(self, path):
                pass
            def verify(self):
                return {'manifest_sha256': 'b' * 64}
            def provenance(self):
                return self.verify()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); views = root / 'views'; views.mkdir()
            (views / 'manifest.json').write_text(json.dumps({'scenarios': {scenario: {}}}), encoding='utf-8')
            cfg = protocol(); cfg['scenarios'] = [scenario]; cfg['status'] = 'frozen'
            cfg['budget_signature'] = train.digest_json(cfg['training'])
            config_path = root / 'config.json'; config_path.write_text(json.dumps(cfg), encoding='utf-8')
            accessed = []
            gold_opened = []
            def read_view(_views, current, split):
                accessed.append(split)
                records = rows(split, 3)
                if split == 'test':
                    records = [{k: v for k, v in row.items() if k != 'pkd'} for row in records]
                return records
            original_sha = train.file_sha
            def sha(path):
                if Path(path).name == 'test_gold.jsonl':
                    return FROZEN_SHAS[scenario]['test_gold.jsonl']
                return original_sha(path)
            def read_gold(path):
                gold_opened.append(str(path))
                return rows('test', 3)
            common = ['--config', str(config_path), '--project-root', str(root), '--cache-root', str(root),
                      '--view-root', str(views), '--scenario', scenario, '--variant', 'M0', '--seed', '17', '--device', 'cpu']
            with patch('src.mm_data.read_input_view', side_effect=read_view), \
                 patch('src.mm_data.read_jsonl', side_effect=read_gold), \
                 patch('src.mm_data.collate', side_effect=lambda selected, *_args, **kwargs: collate(selected, **kwargs)), \
                 patch('src.mm_data_evaluation.collate_evaluation', side_effect=lambda selected, *_args, **kwargs: collate(selected, supervised=True)), \
                 patch('src.mm_extract.SequenceStore', FakeStore), \
                 patch.object(train, 'verify_model_assets', return_value={}), \
                 patch.object(train, 'load_components', side_effect=lambda *_args: (TinyModel(), ToyTokenizer())), \
                 patch.object(train, 'file_sha', side_effect=sha), \
                 patch('builtins.print'):
                self.assertEqual(train.main(common + ['--output', str(root / 'fit'), '--phase', 'fit']), 0)
                self.assertEqual(accessed, ['train', 'validation'])
                self.assertFalse(gold_opened)
                self.assertFalse((root / 'fit/test_lm_predictions.jsonl').exists())
                self.assertEqual(train.main(common + ['--output', str(root / 'evaluation'), '--phase', 'evaluate',
                    '--checkpoint', str(root / 'fit/selected.pt')]), 0)
            self.assertEqual(accessed, ['train', 'validation', 'validation', 'test'])
            self.assertEqual(len(gold_opened), 1)
            result = json.loads((root / 'evaluation/run_manifest.json').read_text())
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['execution_phase'], 'evaluate')
            self.assertEqual(set(result['artifacts']), {'history', 'selection', 'checkpoint', 'lm_predictions', 'numeric_predictions', 'c1'})
            self.assertEqual(train.file_sha(root / 'fit/selected.pt'), train.file_sha(root / 'evaluation/selected.pt'))
            self.assertEqual(len((root / 'evaluation/c1_predictions.jsonl').read_text().splitlines()), 3)

    def test_failed_cli_receipt_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'draft.json'
            config.write_text(json.dumps(protocol()), encoding='utf-8')
            output = root / 'run'
            with self.assertRaisesRegex(ValueError, 'frozen'):
                train.main(['--config', str(config), '--project-root', str(root), '--cache-root', str(root),
                            '--output', str(output), '--scenario', 'known', '--variant', 'M0', '--seed', '17',
                            '--device', 'cpu', '--phase', 'fit'])
            self.assertEqual(json.loads((output / 'failed.json').read_text())['status'], 'failed')
            self.assertFalse((output / 'receipt.json').exists())
            with self.assertRaises(FileExistsError):
                train.main(['--config', str(config), '--project-root', str(root), '--cache-root', str(root),
                            '--output', str(output), '--scenario', 'known', '--variant', 'M0', '--seed', '17',
                            '--device', 'cpu', '--phase', 'fit'])


if __name__ == '__main__':
    unittest.main()

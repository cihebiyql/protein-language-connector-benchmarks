"""CPU fixtures for runner integrity, never reported as biological results."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

try:
    import torch
    from torch import nn
    import transformers
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
if torch is not None:
    from foundation_20261016 import run_connectors as runner
    from foundation_20261016.connectors import FrozenConnectorLM


def row(identifier='one', split='train', label=5.2, target_length=12):
    return {'sample_id': identifier, 'split': split, 'group_id': identifier,
            'peptide_sequence': 'ACD', 'target_sequence': 'E' * target_length,
            'pkd': label, 'source_doi': 'forbidden_doi', 'target_name': 'forbidden_target',
            'raw_metadata': {'answer': 'forbidden_metadata'},
            'chemical_state': {'topology': 'Linear', 'c_terminal': 'Amide'},
            'experimental_context': {'ph': 7.4, 'assay': 'Radioligand binding', 'buffer': 'saline'},
            'target_sequence_role': 'database_reference_sequence_actual_assay_construct_unverified',
            'groups': {'target_sequence_sha256': 'target-' + str(target_length)}}


class FakeTokenizer:
    pad_token_id, eos_token_id = 0, 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return json.dumps(messages, sort_keys=True) + '\nassistant:'

    def encode(self, text, add_special_tokens=False):
        return [ord(character) + 3 for character in text]

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(value - 3) for value in ids if value >= 3)


@unittest.skipIf(torch is None, 'Run in the existing Torch/Transformers environment')
class RunnerTests(unittest.TestCase):
    def test_default_dataset_is_v2_and_old_output_requires_explicit_new_path(self):
        args = runner.parser().parse_args(['--scenario', runner.SCENARIOS[0], '--method', 'input_mlp', '--output', 'new'])
        self.assertEqual(args.data_root.name, 'a1_locked_v2')
        self.assertIsNone(args.train_limit)
        self.assertIsNone(args.validation_limit)
        self.assertIsNone(args.test_limit)
        self.assertEqual(args.device, 'cpu')

    def test_prompt_excludes_answer_identifiers_and_sequences(self):
        original = row()
        text = runner.prompt_text(original, FakeTokenizer())
        for forbidden in ('forbidden_', 'ACD', 'EEEEEEEEEEEE', '"pKD":5.2', '"sample_id"', '"source_doi"'):
            self.assertNotIn(forbidden, text)
        changed = copy.deepcopy(original)
        changed.update({'sample_id': 'different', 'pkd': 123., 'source_doi': 'other', 'target_name': 'other'})
        changed['raw_metadata'] = {'affinity': 123.}
        self.assertEqual(runner.prompt_text(changed, FakeTokenizer()), text)
        self.assertEqual(runner.context_view(original)['chemical_state']['c_terminal'], 'Amide')
        self.assertIn('Radioligand binding', text)
        self.assertIn('saline', text)

    def test_context_exactly_matches_existing_predictor_information_allowlist(self):
        from bench_v02.train_panel_baselines import ContextEncoder
        original = row()
        original['chemical_state'].update({'chemical_modification': 'acetylated', 'explicitly_representable': True})
        original['experimental_context'].update({'temperature_kelvin': 298., 'actual_construct_verified': True,
                                                'actual_construct_sequence': 'FORBIDDEN_CONSTRUCT',
                                                'comparable_condition_group': 'FORBIDDEN_GROUP'})
        view = runner.context_view(original)
        hashed_fields = {'chemical_state': view['chemical_state'], 'assay': view['assay'],
                         'buffer': view['buffer'], 'target_sequence_role': view['target_sequence_role']}
        expected_hashed = {'chemical_state': original['chemical_state'],
                           'assay': original['experimental_context']['assay'],
                           'buffer': original['experimental_context']['buffer'],
                           'target_sequence_role': original['target_sequence_role']}
        self.assertEqual(hashed_fields, expected_hashed)
        encoder = ContextEncoder([original])
        for key in ('temperature_kelvin', 'ph'):
            self.assertEqual(view['experimental_context'][key], encoder.statistics[key]['mean'])
        encoded = encoder(original)
        keys = ('temperature_kelvin', 'ph', 'buffer', 'assay', 'actual_construct_sequence')
        self.assertEqual([float(view['missing'][key]) for key in keys], encoded[2:7].tolist())
        self.assertEqual(float(view['actual_construct_verified']), float(encoded[7]))
        self.assertNotIn('FORBIDDEN_CONSTRUCT', json.dumps(view))
        self.assertNotIn('FORBIDDEN_GROUP', json.dumps(view))
        self.assertEqual(set(view), {'chemical_state', 'experimental_context', 'missing',
                                     'actual_construct_verified', 'assay', 'buffer', 'target_sequence_role'})

    def test_strict_finite_parser_rejects_extra_text_wrong_fields_and_nan(self):
        self.assertEqual(runner.parse_answer(' {"pKD": 5.1}\n')['value'], 5.1)
        for text in ('5.1', '{"pKD":"5.1"}', '{"pKD":true}', '{"pKD":NaN}',
                     '{"pKD":Infinity}', '{"pKD":1e999}', '{"pKD":5,"why":1}',
                     '{"pKD":5} because', '```json\n{"pKD":5}\n```', '{"pKD":5,"pKD":6}'):
            with self.subTest(text=text):
                self.assertFalse(runner.parse_answer(text)['valid'])
        self.assertEqual(runner.parse_answer('{"pKD":-1.2}')['value'], -1.2)

    def test_failure_denominator_never_disappears(self):
        rows = [row('a', 'test', 5.), row('b', 'test', 7.)]
        predictions = [{'sample_id': 'a', 'parse': runner.parse_answer('{"pKD":6}')},
                       {'sample_id': 'b', 'parse': runner.parse_answer('I cannot answer')}]
        metrics = runner.generation_metrics(rows, predictions)
        self.assertEqual((metrics['n_total'], metrics['n_valid'], metrics['n_failed']), (2, 1, 1))
        self.assertEqual(metrics['parse_valid_rate'], .5)
        self.assertEqual(metrics['regression_valid_only']['mae'], 1.)
        self.assertIsNone(metrics['full_sample_mae'])
        metrics = runner.generation_metrics(rows, [{'sample_id': r['sample_id'], 'parse': runner.parse_answer('bad')} for r in rows])
        self.assertIsNone(metrics['regression_valid_only']['mae'])
        self.assertEqual(metrics['n_failed'], 2)

    def test_select_splits_preserves_order_and_explicit_subset_scope(self):
        rows = [row(split + str(i), split) for split in ('train', 'validation', 'test') for i in range(3)]
        splits, manifest = runner.select_splits(rows, {'train': 2, 'validation': None, 'test': 1})
        self.assertEqual([r['sample_id'] for r in splits['train']], ['train0', 'train1'])
        self.assertEqual(manifest['validation']['available_rows'], 3)
        self.assertEqual(manifest['validation']['selected_rows'], 3)
        with self.assertRaises(ValueError):
            runner.select_splits(rows, {'train': 0})

    def test_collation_covers_full_target_and_labels_only_answer(self):
        tokenizer, store = FakeTokenizer(), FakeStore()
        rows = [row('a', target_length=1030), row('b', target_length=5)]
        batch = runner.collate(rows, tokenizer, store, 'cpu')
        self.assertEqual(batch['memory'].shape, (2, 1033, 640))
        self.assertEqual(batch['memory_mask'].sum(-1).tolist(), [1033, 8])
        self.assertEqual(batch['memory_type_ids'][0, :3].tolist(), [0, 0, 0])
        self.assertTrue(torch.all(batch['memory_type_ids'][0, 3:] == 1))
        for i, record in enumerate(rows):
            supervised = batch['labels'][i][batch['labels'][i] != -100].tolist()
            self.assertEqual(tokenizer.decode(supervised), runner.answer_text(record))
            prefix_length = len(tokenizer.encode(runner.prompt_text(record, tokenizer)))
            self.assertTrue(torch.all(batch['labels'][i, :prefix_length] == -100))
        without_gold = runner.collate(rows, tokenizer, store, 'cpu', supervised=False)
        self.assertNotIn('labels', without_gold)
        self.assertLess(without_gold['input_ids'].shape[1], batch['input_ids'].shape[1])

    def test_mock_training_checkpoints_connector_only_and_generates_raw_outputs(self):
        torch.manual_seed(17)
        model, tokenizer, store = FakeConnector(), FakeTokenizer(), FakeStore()
        frozen = {key: value.clone() for key, value in model.base_lm.state_dict().items()}
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(output=Path(temporary), learning_rate=1e-3, weight_decay=.01,
                                   seed=17, epochs=2, batch_size=1, device='cpu', max_grad_norm=1.,
                                   max_new_tokens=24, message_modes=['correct', 'off'])
            splits = {part: [row(part, part)] for part in ('train', 'validation', 'test')}
            logs = []
            outcome = runner.train_and_evaluate(model, tokenizer, splits, store, args, {'fixture': True}, logs.append)
            self.assertEqual(outcome['optimizer_steps'], 2)
            self.assertEqual(outcome['test_generation']['correct']['n_total'], 1)
            saved = torch.load(args.output / 'best_connector.pt', weights_only=True)
            self.assertTrue(saved['connector_state'])
            self.assertFalse(any(key.startswith('base_lm.') for key in saved['connector_state']))
            self.assertEqual(outcome['best_epoch'], min((item['validation_answer_token_nll'], item['epoch'])
                                                       for item in json.loads((args.output / 'history.json').read_text()))[1])
            raw = json.loads((args.output / 'test_predictions_correct.jsonl').read_text())
            self.assertEqual(raw['raw_text'], '{"pKD":5.000000}')
            self.assertTrue(raw['raw_generated_token_ids'])
            self.assertTrue(any(item['event'] == 'optimizer_step' for item in logs))
        for key, value in model.base_lm.state_dict().items():
            self.assertTrue(torch.equal(value, frozen[key]))

    def test_nll_is_answer_token_weighted_and_rejects_nonfinite(self):
        logits = torch.zeros(1, 4, 5, requires_grad=True)
        labels = torch.tensor([[-100, -100, 1, 2]])
        loss, count = runner.supervised_nll(SimpleNamespace(logits=logits), labels)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(float(loss.detach()), 2 * float(torch.log(torch.tensor(5.))), places=5)
        with self.assertRaises(ValueError):
            runner.supervised_nll(SimpleNamespace(logits=logits * float('nan')), labels)

    def test_actual_tiny_qwen_supports_answer_loss_and_connector_checkpoint(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.manual_seed(17)
        config = Qwen2Config(vocab_size=131, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                             max_position_embeddings=2048, pad_token_id=0, eos_token_id=1,
                             attention_dropout=0.)
        base = Qwen2ForCausalLM(config).eval().requires_grad_(False)
        model = FrozenConnectorLM(base, 'internal_cross_attention', memory_dim=8,
                                  num_memory_tokens=2, insertion_layer=1)
        batch = runner.collate([row()], FakeTokenizer(), FakeStore(), 'cpu')
        summed, count = runner.supervised_nll(model(**batch), batch['labels'])
        (summed / count).backward()
        runner.assert_frozen(base)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.parameters() if p.requires_grad))
        original = runner.connector_state(model)
        runner.load_connector_state(model, original)
        self.assertTrue(all(torch.equal(original[key], value) for key, value in runner.connector_state(model).items()))


if torch is not None:
    class FakeStore:
        def residues(self, sequence):
            return torch.full((len(sequence), 640), .1)


    class FakeConnector(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_lm = nn.Sequential(nn.Embedding(131, 8), nn.Linear(8, 131)).requires_grad_(False)
            self.connector = nn.Linear(640, 131)

        def forward(self, input_ids, attention_mask, memory, memory_mask, memory_type_ids=None, labels=None):
            logits = self.base_lm(input_ids)
            weights = memory_mask.float().unsqueeze(-1)
            pooled = (memory * weights).sum(1) / weights.sum(1)
            return SimpleNamespace(logits=logits + self.connector(pooled).unsqueeze(1))

        def greedy_generate(self, input_ids, **kwargs):
            text = '{"pKD":5.000000}'
            return torch.tensor([FakeTokenizer().encode(text)] * len(input_ids))


if __name__ == '__main__':
    unittest.main()

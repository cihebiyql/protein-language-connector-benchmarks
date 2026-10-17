"""CPU regressions for optimizer weighting and split/answer boundaries."""
import json
from pathlib import Path
import random
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src import mt_train as train


def rows(count=5, split='train', task='rt'):
    return [{'sample_id': str(i), 'task': task, 'split': split, 'sequence': 'ACDE',
             'condition_values': [None] * 10, 'target': i / 3 + .123456789,
             'positive_ids': [], 'group_id': str(i), 'x': i + 1., 'answer_tokens': i % 3 + 1}
            for i in range(count)]


def config():
    return {'model': {'memory_dim': 128}, 'training': {'epochs': 1, 'lr': .0003, 'weight_decay': .01,
            'max_grad_norm': 1., 'micro_batch_size': 1, 'grad_accumulation': 4,
            'effective_batch_size': 4, 'readout_batch_size': 8}, 'backbones': {}, 'numerical_policy': {}}


class DummyCollator:
    def __call__(self, batch, supervised=False):
        x = torch.tensor([row['x'] for row in batch])
        ids = torch.tensor([[0, 1, 2, 0, 1, 2]] * len(batch))
        result = {'input_ids': ids, 'x': x}
        if supervised:
            labels = torch.full_like(ids, -100)
            for index, row in enumerate(batch):
                labels[index, -row['answer_tokens']:] = ids[index, -row['answer_tokens']:]
            result['labels'] = labels
        return result


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.13, -.17]))

    def forward(self, input_ids=None, x=None, memory_inputs=None, **kwargs):
        if x is None:
            x = memory_inputs['p15'][:, 0, 0]
        result = {'numeric_z': self.weight[0] * x + self.weight[1]}
        if input_ids is not None:
            basis = torch.tensor([-.4, .7, .2])
            result['logits'] = self.weight[0] * basis[None, None, :] * (input_ids.float()[:, :, None] + 1)
            result['logits'] = result['logits'] + self.weight[1] * basis.flip(0)[None, None, :]
        return result

    def trainable_state_dict(self):
        return {'weight': self.weight}

    def load_trainable_state_dict(self, state):
        self.load_state_dict(state)

    def capacity_report(self):
        return {'trainable_parameters': 2}


def test_train_stats_preserve_original_precision_and_reject_other_splits():
    source = rows()
    stats = train.NumericStats.fit(source)
    assert stats.mean == pytest.approx(sum(row['target'] for row in source) / 5, abs=1e-15)
    assert stats.inverse(stats.targets(source, 'cpu')[0]) == pytest.approx(source[0]['target'], abs=1e-7)
    with pytest.raises(ValueError, match='TRAIN'):
        train.NumericStats.fit(rows(split='test'))


def test_supervised_answer_is_train_only_and_does_not_clip_original_gold():
    source = rows(1)[0]
    assert train.answer_text(source, []) == '{"retention_time_min":0.1235}'
    assert source['target'] == .123456789
    with pytest.raises(ValueError, match='TRAIN'):
        train.answer_text({**source, 'split': 'validation'}, [])


def test_elm_answer_uses_whole_semantic_description_in_catalog_order():
    description = 'Long functional description. ' * 100
    catalog = [{'candidate_id': 'c0', 'description': description}, {'candidate_id': 'c1', 'description': 'other'}]
    source = {**rows(1, task='elm')[0], 'positive_ids': ['c1', 'c0'], 'target': None}
    assert json.loads(train.answer_text(source, catalog)) == {'function': description}


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)

    def encode(self, text, **kwargs):
        return [3 + ord(char) % 20 for char in text]


def test_prompt_does_not_serialize_sequence_label_split_or_identity():
    source = {**rows(1)[0], 'sample_id': 'SECRET_SAMPLE', 'sequence': 'UNIQUESEQUENCE',
              'group_id': 'SECRET_GROUP', 'target': 192837.12345, 'positive_ids': ['SECRET_CLASS']}
    prompt = train.prompt_text(source, Tokenizer())
    assert all(secret not in prompt for secret in ('SECRET_SAMPLE', 'UNIQUESEQUENCE', 'SECRET_GROUP', '192837', 'SECRET_CLASS', 'train'))
    assert 'retention_time_min' in prompt


def test_collator_rejects_test_supervision_and_preserves_feature_only_sequence_input():
    store = types.SimpleNamespace(get=lambda sequence: {'p15': np.zeros((8, 640), np.float32),
                                                        'p30': np.ones((8, 640), np.float32)})
    collator = train.Collator(store, Tokenizer())
    result = collator(rows(2), supervised=True)
    assert result['memory_inputs']['p15'].shape == (2, 8, 640)
    assert result['memory_inputs']['condition_ids'].shape == (2, 10)
    assert (result['labels'][:, :10] == -100).all()
    with pytest.raises(ValueError, match='non-TRAIN'):
        collator(rows(1, split='test'), supervised=True)


def test_known_positive_ranking_loss_uses_all_31_descriptions():
    logits = torch.arange(31, dtype=torch.float32)[None].requires_grad_()
    ids = [str(i) for i in range(31)]
    source = [{'positive_ids': ['1', '5']}]
    loss = train.ranking_loss_sum(logits, source, ids)
    assert loss.item() == pytest.approx(float(-F.log_softmax(logits, -1)[0, [1, 5]].mean()))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match='31'):
        train.ranking_loss_sum(logits[:, :2], source, ids[:2])
    with pytest.raises(ValueError, match='known positives'):
        train.ranking_loss_sum(logits, [{'positive_ids': ['unknown']}], ids)


def test_micro_accumulation_equals_full_effective_objective_including_partial_tail(tmp_path):
    source, validation, cfg = rows(5), rows(2, split='selection'), config()
    identity = {'seed': 17, 'method': 'M0'}
    actual, reference = DummyModel(), DummyModel()
    collator = DummyCollator()
    result, stats = train.fit_model(actual, source, validation, collator, cfg, identity, tmp_path,
                                    'rt', [], progress=None)
    optimizer = torch.optim.AdamW(reference.parameters(), lr=.0003, weight_decay=.01)
    order = list(range(5))
    random.Random(18).shuffle(order)
    for start in range(0, 5, 4):
        group = [source[index] for index in order[start:start + 4]]
        batch = collator(group, supervised=True)
        output = train.forward(reference, batch)
        ce, tokens = train.supervised_ce(output['logits'], batch['labels'])
        loss = ce / tokens + F.mse_loss(output['numeric_z'], stats.targets(group, 'cpu'), reduction='mean')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        optimizer.step()
    assert torch.allclose(actual.weight, reference.weight, atol=2e-8, rtol=0)
    assert result['optimizer_steps'] == 2 and result['examples_seen'] == 5
    payload = torch.load(tmp_path / 'checkpoint.pt', map_location='cpu', weights_only=True)
    assert set(payload['trainable_state']) == {'weight'}
    assert payload['trainable_state']['weight'].device.type == 'cpu'


def test_canary_consumes_exactly_four_updates_sixteen_train_examples(tmp_path):
    result, _ = train.fit_model(DummyModel(), rows(20), None, DummyCollator(), config(),
                                {'seed': 42, 'method': 'M0'}, tmp_path, 'rt', [], max_updates=4)
    assert (result['optimizer_steps'], result['examples_seen'], result['epochs_completed']) == (4, 16, 0)
    with pytest.raises(ValueError, match='must not receive validation'):
        train.fit_model(DummyModel(), rows(20), rows(2, 'selection'), DummyCollator(), config(),
                        {'seed': 42, 'method': 'M0'}, tmp_path, 'rt', [], max_updates=4)


def test_checkpoint_earliest_tie_and_task_specific_selection_split():
    best = {'value': .5, 'maximize': True}
    assert not train.is_better(dict(best), best)
    assert train.is_better({'value': .6, 'maximize': True}, best)
    with pytest.raises(ValueError, match='wrong checkpoint selection split'):
        train.selection_value('rt', rows(2, 'calibration'), [], [])


def test_formal_requires_exact_frozen_training_signature():
    cfg = config()
    with pytest.raises(ValueError, match='frozen'):
        train.check_protocol(cfg, 'rt', 'M0', 'qwen25_1p5b', 17, 'formal', None)
    cfg.update(status='frozen', training_signature=train.digest_json(cfg['training']))
    train.check_protocol(cfg, 'rt', 'M0', 'qwen25_1p5b', 17, 'formal', None)
    cfg['training']['epochs'] += 1
    with pytest.raises(ValueError, match='signature'):
        train.check_protocol(cfg, 'rt', 'M0', 'qwen25_1p5b', 17, 'formal', None)


@pytest.mark.parametrize(('task', 'limit'), [('rt', 24), ('elm', 256)])
def test_greedy_generation_preserves_whole_fixed_token_budget(task, limit):
    class Endless(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            logits = torch.zeros((*input_ids.shape, 5))
            logits[..., 3] = 1.
            return {'logits': logits}
    tokenizer = types.SimpleNamespace(eos_token_id=2, pad_token_id=0,
                                      decode=lambda values, **kwargs: ''.join(map(str, values)))
    def collator(selected, supervised=False):
        return {'input_ids': torch.ones((len(selected), 2), dtype=torch.long),
                'attention_mask': torch.ones((len(selected), 2), dtype=torch.long),
                'prompt_end_index': torch.ones(len(selected), dtype=torch.long)}
    predicted = train.language_predictions(Endless(), rows(2, task=task), collator, tokenizer, task, batch_size=2)
    assert len(predicted) == 2
    assert all(row['text'] == '3' * limit and row['generated_tokens'] == limit for row in predicted)


def test_greedy_batch_handles_eos_independently_without_padding_as_output():
    class Finite(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, input_ids, **kwargs):
            logits = torch.zeros((*input_ids.shape, 6))
            logits[0, -1, 3 if self.calls == 0 else 2] = 1.
            logits[1, -1, 4 if self.calls < 2 else 2] = 1.
            self.calls += 1
            return {'logits': logits}
    tokenizer = types.SimpleNamespace(eos_token_id=2, pad_token_id=0,
                                      decode=lambda values, **kwargs: ''.join(map(str, values)))
    def collator(selected, supervised=False):
        return {'input_ids': torch.ones((2, 2), dtype=torch.long),
                'attention_mask': torch.ones((2, 2), dtype=torch.long),
                'prompt_end_index': torch.ones(2, dtype=torch.long)}
    predicted = train.language_predictions(Finite(), rows(2), collator, tokenizer, 'rt', batch_size=2)
    assert [row['text'] for row in predicted] == ['3', '44']


@pytest.mark.parametrize('leak_location', [None, 'numeric_z', 'prompt_hidden', 'routes.view'])
def test_anchor_guard_uses_same_shape_future_values_and_checks_hidden_routes(leak_location):
    class ShapeSensitive(torch.nn.Module):
        def forward(self, input_ids, prompt_end_index, **kwargs):
            # Numerical shape sensitivity is independent of every future token's value.
            prefix = input_ids[:, :2].float().sum(-1)
            offset = input_ids.shape[1] * .02
            numeric = prefix + offset
            hidden = torch.stack((prefix + offset, prefix * .5 + offset), dim=-1)
            view = torch.stack((prefix * 0 + .3 + offset, prefix * 0 + .7 - offset), dim=-1)
            bank = view.clone()
            future_sum = input_ids[:, 2:].float().sum(-1)
            if leak_location == 'numeric_z':
                numeric = numeric + future_sum
            elif leak_location == 'prompt_hidden':
                hidden = hidden + future_sum[:, None]
            elif leak_location == 'routes.view':
                view = view + future_sum[:, None]
            return {'numeric_z': numeric, 'prompt_hidden': hidden, 'routes': {'view': view, 'bank': bank}}
    def collator(selected, supervised=False):
        return {'input_ids': torch.tensor([[3, 4]]), 'attention_mask': torch.ones((1, 2), dtype=torch.long),
                'prompt_end_index': torch.tensor([1])}
    result = train.prompt_anchor_check(ShapeSensitive(), rows(1), collator, 'rt')
    assert result['passed'] is (leak_location is None)
    assert result['atol'] == result['rtol'] == 1e-5
    assert result['append_shape_diagnostic']['used_for_pass'] is False
    assert result['append_shape_diagnostic']['max_abs_difference']['numeric_z'] > 1e-5
    if leak_location is None:
        assert result['max_abs_difference'] == 0
    else:
        assert result['components'][leak_location]['passed'] is False


def test_cli_canary_never_opens_validation_calibration_or_test(tmp_path, monkeypatch):
    from src import mt_data, mt_extract
    data, cache, output = tmp_path / 'data', tmp_path / 'cache', tmp_path / 'run'
    data.mkdir()
    cache.mkdir()
    (data / 'manifest.json').write_text('{}')
    (cache / 'cache_spec.json').write_text('{}')
    opened = []
    source = rows(20)
    def load_task(root, task, split):
        opened.append(split)
        assert split == 'train'
        return source
    class Store:
        def __init__(self, root):
            pass
        def verify_all(self, selected):
            assert all(row['split'] == 'train' for row in selected)
            return {'status': 'complete'}
        def get(self, sequence):
            return {'p15': np.ones((8, 640), np.float32), 'p30': np.ones((8, 640), np.float32)}
    monkeypatch.setattr(mt_data, 'load_task', load_task)
    monkeypatch.setattr(mt_data, 'load_manifest', lambda root: {})
    monkeypatch.setattr(mt_extract, 'FeatureStore', Store)
    monkeypatch.setattr(train, 'load_components', lambda *args: (DummyModel(), None, None))
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config()))
    result = train.main(['--config', str(path), '--task', 'rt', '--method', 'ESM', '--backbone', 'shared',
                         '--seed', '17', '--data-root', str(data), '--cache-root', str(cache), '--output', str(output),
                         '--device', 'cpu', '--phase', 'train-canary', '--max-updates', '4'])
    assert result == 0 and opened == ['train']
    assert not (output / 'model_frozen_before_test.json').exists()


def test_cli_freezes_checkpoint_before_calibration_or_test_open(tmp_path, monkeypatch):
    from src import mt_data, mt_extract
    data, cache, output = tmp_path / 'data', tmp_path / 'cache', tmp_path / 'run'
    data.mkdir()
    cache.mkdir()
    (data / 'manifest.json').write_text('{}')
    (cache / 'manifest.json').write_text('{}')
    opened = []
    def load_task(root, task, split):
        opened.append(split)
        if split in ('test', 'calibration'):
            receipt = json.loads((output / 'model_frozen_before_test.json').read_text())
            assert receipt['test_opened'] is False
            assert receipt['checkpoint_sha256'] == train.file_sha(output / 'checkpoint.pt')
        result = rows(5 if split == 'train' else 2, split)
        for row in result:
            row['sample_id'] = split + '_' + row['sample_id']
        return result
    class Store:
        def __init__(self, root):
            pass
        def verify_all(self, selected):
            return {'status': 'complete'}
        def get(self, sequence):
            return {'p15': np.ones((8, 640), np.float32), 'p30': np.ones((8, 640), np.float32)}
    monkeypatch.setattr(mt_data, 'load_task', load_task)
    monkeypatch.setattr(mt_data, 'load_manifest', lambda root: {})
    monkeypatch.setattr(mt_extract, 'FeatureStore', Store)
    monkeypatch.setattr(train, 'load_components', lambda *args: (DummyModel(), None, None))
    cfg = config()
    cfg.update(status='frozen', training_signature=train.digest_json(cfg['training']))
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg))
    result = train.main(['--config', str(path), '--task', 'rt', '--method', 'ESM', '--backbone', 'shared',
                         '--seed', '17', '--data-root', str(data), '--cache-root', str(cache), '--output', str(output),
                         '--device', 'cpu', '--phase', 'formal'])
    assert result == 0 and opened == ['train', 'selection', 'calibration', 'test']
    manifest = json.loads((output / 'run_manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['metrics']['language'] is None
    assert manifest['metrics']['C1_off'] is None
    assert manifest['metrics']['calibration']['calibration_groups'] == 2

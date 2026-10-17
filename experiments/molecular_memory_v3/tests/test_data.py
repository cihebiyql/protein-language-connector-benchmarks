"""CPU-only data leakage, coverage, paired-layer and causal-label fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from src import mm_data as data
from src import mm_extract as extract

torch.set_num_threads(2)


def row(split='train', pkd=7.12345):
    return {'sample_id': 'PRIVATE_ID', 'split': split, 'peptide_sequence': 'ACD',
        'target_sequence': 'EFGH', 'pkd': pkd, 'source_doi': 'PRIVATE_DOI',
        'target_sequence_role': 'database_reference_sequence_actual_assay_construct_unverified',
        'chemical_state': {'n_terminal': 'Free', 'explicitly_representable': False},
        'experimental_context': {'assay': 'Binding', 'temperature_kelvin': None,
            'ph': None, 'buffer': None, 'actual_construct_sequence': None, 'actual_construct_verified': False},
        'raw_metadata': {'gold': 99}, 'groups': {'private': 'GROUP'}}


class Tokenizer:
    eos_token_id, pad_token_id = 2, 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return '\n'.join(m['content'] for m in messages) + '\nassistant:'

    def encode(self, text, add_special_tokens=False):
        return [3 + ord(c) % 100 for c in text]

    def __call__(self, sequence, **_kwargs):
        return {'input_ids': torch.tensor([[0] + [3 + 'ACDEFGHIKLMNPQRSTVWY'.index(c) for c in sequence] + [2]]),
                'attention_mask': torch.ones(1, len(sequence) + 2, dtype=torch.long)}


class Store:
    def pooled(self, sequence):
        return {'H15': torch.full((8, 640), float(len(sequence))),
                'H30': torch.full((8, 640), float(len(sequence) + 1))}


class Block(nn.Module):
    def __init__(self, width, index):
        super().__init__()
        self.register_buffer('delta', torch.arange(width).float() / width + (index + 1) / 100)

    def forward(self, x):
        return (x + self.delta,)


class TinyESM(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.embedding = nn.Embedding(33, width)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([Block(width, i) for i in range(30)])
        self.encoder.emb_layer_norm_after = nn.LayerNorm(width)
        self.fail = False
        self.calls = 0
        self.requires_grad_(False).eval()

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        x = self.embedding(input_ids)
        for block in self.encoder.layer:
            x = block(x)[0]
        if self.fail:
            raise RuntimeError('fixture failure after hook')
        return types.SimpleNamespace(last_hidden_state=self.encoder.emb_layer_norm_after(x))


class DataTests(unittest.TestCase):
    def test_strict_json_duplicates_nonfinite_overflow_and_blank_rows(self):
        for value in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.assertRaises(ValueError):
                data.strict_loads(value)
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / 'bad.jsonl'
            p.write_text('{}\n\n')
            with self.assertRaises(ValueError):
                data.read_jsonl(p)

    def test_prompt_and_conditions_ignore_gold_ids_nested_unlisted_metadata(self):
        original = row()
        altered = copy.deepcopy(original)
        altered.update(pkd=-90, sample_id='OTHER', source_doi='OTHER', raw_metadata={'pkd': 42})
        altered['chemical_state']['gold'] = {'pkd': 80}
        altered['experimental_context']['source_record_id'] = 'OTHER'
        self.assertEqual(data.prompt_text(original, Tokenizer()), data.prompt_text(altered, Tokenizer()))
        self.assertEqual(data.condition_encoding(original), data.condition_encoding(altered))
        prompt = data.prompt_text(original, Tokenizer())
        for hidden in ('PRIVATE_ID', 'PRIVATE_DOI', '7.12345', 'PRIVATE', 'GROUP'):
            self.assertNotIn(hidden, prompt)

    def test_allowlisted_nested_object_rejected(self):
        r = row(); r['chemical_state']['n_terminal'] = {'pkd': 3}
        with self.assertRaises(ValueError):
            data.context_view(r)
        for value in (True, float('inf'), '7'):
            r = row(); r['experimental_context']['ph'] = value
            with self.assertRaises(ValueError):
                data.context_view(r)

    def test_ten_fixed_conditions_missing_false_and_real_zero(self):
        r = row()
        ids, mask = data.condition_encoding(r)
        self.assertEqual(len(ids), 10)
        self.assertTrue(mask[3])  # actual false value is observed, not absent
        self.assertFalse(mask[8])
        self.assertTrue(mask[9])
        r['experimental_context']['ph'] = 0.0
        zero_ids, zero_mask = data.condition_encoding(r)
        self.assertTrue(zero_mask[8])
        self.assertTrue(all(0 <= i < 4096 for i in zero_ids))
        self.assertNotEqual(ids[8], zero_ids[8])

    def test_filtered_view_preserves_missing_semantics_but_no_construct_text(self):
        r = row(); r['experimental_context']['actual_construct_sequence'] = 'SECRET'
        view = data.input_view(r)
        self.assertNotIn('SECRET', json.dumps(view))
        self.assertEqual(data.context_view(r), data.context_view(view))
        test = data.input_view(row('test'))
        self.assertNotIn('pkd', test)
        with self.assertRaises(ValueError):
            data.input_view(row('test'), labelled=True)

    def test_causal_labels_prompt_end_and_prompt_only(self):
        rows = [row(), row(pkd=-100.25)]
        rows[1]['experimental_context']['assay'] = 'A much longer assay description'
        batch = data.collate(rows, Tokenizer(), Store())
        self.assertEqual(set(batch['memory_inputs']), {'p15', 'p30', 't15', 't30', 'condition_ids', 'observed'})
        for i, r in enumerate(rows):
            end = batch['prompt_end_index'][i].item()
            self.assertTrue((batch['labels'][i, :end + 1] == -100).all())
            suffix = Tokenizer().encode(data.answer_text(r)) + [2]
            self.assertEqual(batch['labels'][i, end + 1:].tolist(), suffix)
            self.assertEqual(batch['input_ids'][i, end + 1:].tolist(), suffix)
            self.assertEqual(int(batch['attention_mask'][i, end]), 1)
        self.assertEqual(batch['memory_inputs']['observed'].dtype, torch.bool)
        self.assertEqual(batch['memory_inputs']['condition_ids'].shape, (2, 10))
        test_batch = data.collate([data.input_view(row('test'))], Tokenizer(), Store(), supervised=False)
        self.assertNotIn('labels', test_batch)
        self.assertEqual(test_batch['prompt_end_index'].item(), test_batch['input_ids'].shape[1] - 1)
        with self.assertRaises(ValueError):
            data.collate([row('test')], Tokenizer(), Store(), supervised=True)

    def test_gold_change_leaves_prompt_prefix_memory_invariant(self):
        a = data.collate([row(pkd=7)], Tokenizer(), Store())
        b = data.collate([row(pkd=-100)], Tokenizer(), Store())
        ai, bi = a['prompt_end_index'].item(), b['prompt_end_index'].item()
        self.assertTrue(torch.equal(a['input_ids'][0, :ai + 1], b['input_ids'][0, :bi + 1]))
        for key in a['memory_inputs']:
            self.assertTrue(torch.equal(a['memory_inputs'][key], b['memory_inputs'][key]))

    def test_registered_root_escape_and_hash_tamper(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / 'a'; p.write_text('old')
            digest = data.file_sha(p)
            data.verify_sha(data.registered_path(root, 'a'), digest)
            p.write_text('new')
            with self.assertRaises(ValueError): data.verify_sha(p, digest)
            for path in ('../a', '/etc/passwd', 'dir\\a'):
                with self.assertRaises(ValueError): data.registered_path(root, path)
            outside = Path(root).parent / 'outside'
            link = Path(root) / 'link'
            try:
                link.symlink_to(Path('/etc/passwd'))
            except (OSError, NotImplementedError):
                pass
            else:
                with self.assertRaises(ValueError): data.registered_path(root, 'link')

    def test_frozen_loader_verifies_transitive_pin_and_test_order(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); base = root / data.DATA_RELATIVE / data.SCENARIOS[0]; base.mkdir(parents=True)
            rows = [row(), row('validation'), row('test')]
            for i, r in enumerate(rows): r['sample_id'] += str(i)
            source = root / 'closure'; source.write_text('frozen')
            (base / 'samples.jsonl').write_text(''.join(data.canonical(r) + '\n' for r in rows))
            (base / 'test_gold.jsonl').write_text(data.canonical(rows[-1]) + '\n')
            (base / 'manifest.json').write_text('{}')
            (base / 'benchmark_lock.json').write_text(data.canonical({'pins': {'closure': data.file_sha(source)}}))
            pins = {p.name: data.file_sha(p) for p in base.iterdir()}
            with patch.dict(data.FROZEN_SHAS, {data.SCENARIOS[0]: pins}), patch.dict(data.SPLIT_COUNTS, {data.SCENARIOS[0]: {'train': 1, 'validation': 1, 'test': 1}}):
                actual, receipt = data.load_frozen_scenario(root, data.SCENARIOS[0])
                self.assertEqual(len(actual), 3)
                self.assertIn('closure', receipt['pins'])
                source.write_text('tampered')
                with self.assertRaises(ValueError): data.load_frozen_scenario(root, data.SCENARIOS[0])


class ExtractionTests(unittest.TestCase):
    def test_disjoint_chunks_cover_long_sequence_and_reject_sizes(self):
        self.assertEqual(extract.chunk_spans(2045), [(0, 1022), (1022, 2044), (2044, 2045)])
        for size in (0, 1023):
            with self.assertRaises(ValueError): extract.chunk_spans(20, size)

    def test_paired_same_pass_raw_full_coverage_pool_semantics(self):
        torch.manual_seed(17)
        model = TinyESM()
        tensors, receipt = extract.extract_sequence(model, Tokenizer(), 'ACDEFGH', max_residues=3)
        self.assertEqual(model.calls, 3)
        self.assertEqual(receipt['raw_residues_covered'], 7)
        self.assertTrue(receipt['raw_full_coverage'])
        self.assertFalse(receipt['whole_sequence_joint_attention'])
        self.assertTrue(receipt['pooling_is_lossy'])
        self.assertEqual(tensors['H15'].shape, (8, 8))
        raw15, raw30 = [], []
        for a, b in extract.chunk_spans(7, 3):
            x = model.embedding(Tokenizer()('ACDEFGH'[a:b])['input_ids'])
            for index, block in enumerate(model.encoder.layer):
                x = block(x)[0]
                if index == 14: raw15.append(x[0, 1:-1])
            raw30.append(model.encoder.emb_layer_norm_after(x)[0, 1:-1])
        for key, raw in (('H15', raw15), ('H30', raw30)):
            expected = F.adaptive_avg_pool1d(torch.cat(raw).T[None], 8)[0].T
            self.assertTrue(torch.equal(tensors[key], expected))
        self.assertFalse(torch.equal(tensors['H15'], tensors['H30']))
        self.assertEqual(len(model.encoder.layer[14]._forward_hooks), 0)

    def test_failed_forward_always_cleans_hook_and_requires_frozen(self):
        model = TinyESM(); model.fail = True
        with self.assertRaises(RuntimeError): extract.extract_sequence(model, Tokenizer(), 'ACD')
        self.assertEqual(len(model.encoder.layer[14]._forward_hooks), 0)
        model.fail = False; model.embedding.weight.requires_grad_(True)
        with self.assertRaises(ValueError): extract.extract_sequence(model, Tokenizer(), 'ACD')

    def test_partial_failure_has_no_final_manifest(self):
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent) / 'failed'
            model = TinyESM(); model.fail = True
            with self.assertRaises(RuntimeError):
                extract.extract_cache(model, Tokenizer(), ['ACD'], root, {'verified': True}, extract.MODEL_PINS)
            self.assertFalse((root / 'manifest.json').exists())
            partial = data.strict_loads((root / 'manifest.partial.json').read_text())
            self.assertEqual(partial['cache_status'], 'failed')
            self.assertEqual(partial['expected_sequences'], 1)
            self.assertEqual(partial['completed_sequences'], 0)

    def test_canary_only_reads_train_selects_fixed_extremes_and_is_not_formal(self):
        samples = [{'peptide_sequence': 'A' * n, 'target_sequence': 'C' * (n + 1)} for n in (2, 4, 8)]
        with patch.object(extract, 'read_input_view', return_value=samples) as reader:
            selected = extract.select_sequences('/unused', 'train-canary', 4)
            self.assertEqual(len(selected), 4)
            self.assertIn('AA', selected)
            self.assertIn('C' * 9, selected)
            reader.assert_called_once_with('/unused', data.SCENARIOS[0], 'train')
        with self.assertRaises(ValueError): extract.select_sequences('/unused', 'full', 4)
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent) / 'canary'
            semantics = {'verified': True, 'source_sha256': extract.REMOTE_TF4481_SOURCE_SHA,
                         'transformers_version': '4.48.1'}
            manifest = extract.extract_cache(TinyESM(640), Tokenizer(), ['ACD'], root,
                semantics, extract.MODEL_PINS, phase='train-canary')
            self.assertEqual(manifest['phase'], 'train-canary')
            with self.assertRaises(ValueError): extract.SequenceStore(root)

    def test_cache_hash_shape_and_coverage_fail_closed(self):
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent) / 'new'
            model = TinyESM(width=640)
            semantics = {'verified': True, 'fixture': True, 'source_sha256': extract.REMOTE_TF4481_SOURCE_SHA,
                         'transformers_version': '4.48.1'}
            extract.extract_cache(model, Tokenizer(), ['ACD'], root, semantics, extract.MODEL_PINS)
            store = extract.SequenceStore(root)
            self.assertEqual(store.pooled('ACD')['H15'].shape, (8, 640))
            self.assertEqual(len(store.verify()['verified_payloads']), 1)
            entry = next(iter(store.manifest['sequences'].values()))
            entry['chunk_spans'][0]['start'] = 1
            with self.assertRaises(ValueError): store.pooled('ACD')
            entry['chunk_spans'][0]['start'] = 0
            path = root / entry['path']; path.write_bytes(path.read_bytes() + b'bad')
            with self.assertRaises(ValueError): store.pooled('ACD')
            with self.assertRaises(FileExistsError):
                extract.extract_cache(model, Tokenizer(), ['ACD'], root, semantics, extract.MODEL_PINS)

    def test_real_tiny_cpu_esm_matches_hidden_state_layer15_and_normalized_final(self):
        from transformers import EsmConfig, EsmModel
        torch.manual_seed(42)
        config = EsmConfig(vocab_size=33, hidden_size=8, num_hidden_layers=30,
            num_attention_heads=2, intermediate_size=16, max_position_embeddings=1026,
            pad_token_id=1, mask_token_id=32, hidden_dropout_prob=0, attention_probs_dropout_prob=0,
            position_embedding_type='rotary', token_dropout=False)
        model = EsmModel(config, add_pooling_layer=False).requires_grad_(False).eval()
        tensors, receipt = extract.extract_sequence(model, Tokenizer(), 'ACDE')
        with torch.no_grad():
            output = model(**Tokenizer()('ACDE'), output_hidden_states=True, return_dict=True)
        expected15 = F.adaptive_avg_pool1d(output.hidden_states[15][0, 1:-1].T[None], 8)[0].T
        expected30 = F.adaptive_avg_pool1d(output.last_hidden_state[0, 1:-1].T[None], 8)[0].T
        self.assertTrue(torch.equal(tensors['H15'], expected15))
        self.assertTrue(torch.equal(tensors['H30'], expected30))
        self.assertTrue(receipt['whole_sequence_joint_attention'])


if __name__ == '__main__':
    unittest.main()

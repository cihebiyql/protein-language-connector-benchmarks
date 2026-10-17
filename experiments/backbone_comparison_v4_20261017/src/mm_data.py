"""Pinned A1 input views, bounded conditions and causal answer collation.

Only prepare_views reads the original mixed-split source. Training should read
the newly written train/validation files, never the original samples.jsonl.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

SCENARIOS = ('new_peptide_known_target', 'new_peptide_new_target')
CHEMICAL_KEYS = ('c_terminal', 'chemical_modification', 'cyclization_method',
                 'explicitly_representable', 'n_terminal', 'residue_modification', 'topology')
CONDITION_SLOTS = CHEMICAL_KEYS + ('assay', 'measurement_context', 'target_context')
DATA_RELATIVE = Path('data/processed/bench_v02/a1_locked_v2')
FROZEN_SHAS = {
    SCENARIOS[0]: dict(zip(('samples.jsonl', 'test_gold.jsonl', 'benchmark_lock.json', 'manifest.json'), (
        'd1d9ce44db4aeefcc829f5ce7b50b3e288ad03a28af9d15bb9f1fd550a6b657e',
        '6f50e85443d1056ef35d509b5229dfb0bc364e266da2aebfa3fa78dcdb07c07b',
        'bf15cc6dfc1a62c3b97a3201ff5961167f0ac0d7afcf7872c46aa1ff57e65249',
        'f90f94866baa34f05af2b5d9416c653847fe4beb0cebbd4614592399e12c190e'))),
    SCENARIOS[1]: dict(zip(('samples.jsonl', 'test_gold.jsonl', 'benchmark_lock.json', 'manifest.json'), (
        '7ff5a67203efaa94db29fad1f7d2774b5f5e5f8a9155fc47a090e0f2a3baee71',
        '388e50ac4776bd122dc3fb60a688ee715173a8703c890ac7d0ad38eaab82065f',
        '329ebdf75e792eabe80d2edf423c53990ea31551fae5112e20b463c30b27b90f',
        '508aa3835d652aab4d115a61fa90bc6aa157e3c195e035803fb808b0cbed88c5'))),
}
SPLIT_COUNTS = {SCENARIOS[0]: {'train': 2062, 'validation': 147, 'test': 152},
                SCENARIOS[1]: {'train': 2063, 'validation': 431, 'test': 442}}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def strict_loads(text):
    def pairs(items):
        if len({k for k, _ in items}) != len(items):
            raise ValueError('duplicate JSON key')
        return dict(items)
    value = json.loads(text, object_pairs_hook=pairs,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    def finite_tree(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError('overflow/nonfinite JSON number')
        if isinstance(item, dict):
            for child in item.values():
                finite_tree(child)
        elif isinstance(item, list):
            for child in item:
                finite_tree(child)
    finite_tree(value)
    return value


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = strict_loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{path}:{number}: invalid JSONL') from exc
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{number}: row must be object')
            rows.append(row)
    return rows


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def registered_path(root, relative):
    """Reject absolute references, traversal and resolved symlink escapes."""
    root = Path(root).resolve(strict=True)
    if isinstance(relative, str) and '\\' in relative:
        raise ValueError('artifact references must use portable forward slashes')
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or ':' in relative.as_posix():
        raise ValueError('non-relative or escaping artifact path')
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('artifact outside registered root or not file')
    return path


def verify_sha(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or file_sha(path) != expected:
        raise ValueError(f'asset SHA mismatch: {path}')


def finite_real(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _category(value):
    if value is None or value == '':
        return None
    if not isinstance(value, (str, bool)):
        raise ValueError('allowlisted categorical field must be scalar string/bool/null')
    return value


def context_view(row):
    chemical = row.get('chemical_state') or {}
    context = row.get('experimental_context') or {}
    if not isinstance(chemical, dict) or not isinstance(context, dict):
        raise ValueError('chemistry/context must be objects')
    numbers = {}
    for key in ('temperature_kelvin', 'ph'):
        value = context.get(key)
        if value is not None and value != '' and not finite_real(value):
            raise ValueError('observed numeric context must be finite real')
        numbers[key] = float(value) if finite_real(value) else None
    verified = context.get('actual_construct_verified', False)
    if not isinstance(verified, bool):
        raise ValueError('actual_construct_verified must be boolean')
    construct_missing = context.get('actual_construct_sequence_missing', context.get('actual_construct_sequence') in (None, ''))
    if not isinstance(construct_missing, bool):
        raise ValueError('actual_construct_sequence_missing must be boolean')
    assay, buffer = (_category(context.get(k)) for k in ('assay', 'buffer'))
    return {'chemical_state': {k: _category(chemical.get(k)) for k in CHEMICAL_KEYS},
            'experimental_context': numbers,
            'missing': {**{k: v is None for k, v in numbers.items()},
                        'assay': assay is None, 'buffer': buffer is None,
                        'actual_construct_sequence': construct_missing},
            'actual_construct_verified': verified, 'assay': assay, 'buffer': buffer,
            'target_sequence_role': _category(row.get('target_sequence_role'))}


def condition_encoding(row):
    """Fixed categorical hash, no TRAIN/test fitted vocabulary or numeric imputation."""
    view = context_view(row)
    measurement = {**view['experimental_context'], 'buffer': view['buffer']}
    target = {k: view[k] for k in ('target_sequence_role', 'actual_construct_verified')}
    target['actual_construct_sequence_missing'] = view['missing']['actual_construct_sequence']
    values = [view['chemical_state'][k] for k in CHEMICAL_KEYS] + [view['assay'], measurement, target]
    observed = [v is not None for v in values[:8]] + [any(v is not None for v in measurement.values()),
                view['target_sequence_role'] is not None or view['actual_construct_verified']
                or not view['missing']['actual_construct_sequence']]
    ids = [int.from_bytes(hashlib.sha256((name + '\0' + canonical(value)).encode()).digest(), 'big') % 4096
           if present else 0 for name, value, present in zip(CONDITION_SLOTS, values, observed)]
    return ids, observed


def input_view(row, labelled=False):
    view = context_view(row)
    result = {key: row[key] for key in ('sample_id', 'split', 'peptide_sequence', 'target_sequence')}
    for key in ('peptide_sequence', 'target_sequence'):
        if not isinstance(result[key], str) or not result[key] or not set(result[key]) <= set('ACDEFGHIKLMNPQRSTVWY'):
            raise ValueError('sequence must be nonempty uppercase residue string')
    result['target_sequence_role'] = view['target_sequence_role']
    result['chemical_state'] = view['chemical_state']
    result['experimental_context'] = {**view['experimental_context'], 'assay': view['assay'],
        'buffer': view['buffer'], 'actual_construct_verified': view['actual_construct_verified'],
        'actual_construct_sequence_missing': view['missing']['actual_construct_sequence']}
    if labelled:
        if result['split'] not in ('train', 'validation') or not finite_real(row.get('pkd')):
            raise ValueError('labelled views require TRAIN/VAL finite label')
        result['pkd'] = float(row['pkd'])
    return result


def load_frozen_scenario(project_root, scenario):
    if scenario not in SCENARIOS:
        raise ValueError('unknown scenario')
    base = DATA_RELATIVE / scenario
    paths = {name: registered_path(project_root, base / name) for name in FROZEN_SHAS[scenario]}
    for name, path in paths.items():
        verify_sha(path, FROZEN_SHAS[scenario][name])
    lock = strict_loads(paths['benchmark_lock.json'].read_text(encoding='utf-8'))
    pin_receipts = {}
    for relative, expected in lock['pins'].items():
        path = registered_path(project_root, relative)
        verify_sha(path, expected)
        pin_receipts[relative] = {'sha256': expected, 'bytes': path.stat().st_size}
    rows, gold = read_jsonl(paths['samples.jsonl']), read_jsonl(paths['test_gold.jsonl'])
    if dict(Counter(row['split'] for row in rows)) != SPLIT_COUNTS[scenario]:
        raise ValueError('frozen split counts differ')
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('duplicate sample identity within scenario')
    test = [row for row in rows if row['split'] == 'test']
    if [(r['sample_id'], r['pkd']) for r in test] != [(r['sample_id'], r['pkd']) for r in gold]:
        raise ValueError('test gold differs in identity/order/value')
    if any(not finite_real(row.get('pkd')) for row in rows):
        raise ValueError('nonfinite/bool gold')
    return rows, {'scenario': scenario, 'source_shas': FROZEN_SHAS[scenario], 'pins': pin_receipts}


def prepare_views(project_root, output_root):
    """One-off audit process; exclusive new output directory, never overwrite."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 'mm_a1_views_v1', 'condition_slots': CONDITION_SLOTS, 'scenarios': {}}
    for scenario in SCENARIOS:
        rows, receipt = load_frozen_scenario(project_root, scenario)
        directory = output_root / scenario
        directory.mkdir()
        receipt['views'] = {}
        for split in ('train', 'validation', 'test'):
            name = 'test_inputs.jsonl' if split == 'test' else split + '.jsonl'
            path = directory / name
            selected = [input_view(row, labelled=split != 'test') for row in rows if row['split'] == split]
            path.write_text(''.join(canonical(row) + '\n' for row in selected), encoding='utf-8', newline='\n')
            receipt['views'][split] = {'path': f'{scenario}/{name}', 'rows': len(selected), 'sha256': file_sha(path)}
        manifest['scenarios'][scenario] = receipt
    (output_root / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest


def read_input_view(view_root, scenario, split):
    manifest = strict_loads(registered_path(view_root, 'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('schema') != 'mm_a1_views_v1' or manifest.get('condition_slots') != list(CONDITION_SLOTS)
            or manifest['scenarios'][scenario]['source_shas'] != FROZEN_SHAS[scenario]):
        raise ValueError('view source identity mismatch')
    source = manifest['scenarios'][scenario]['views'][split]
    path = registered_path(view_root, source['path'])
    verify_sha(path, source['sha256'])
    rows = read_jsonl(path)
    if len(rows) != SPLIT_COUNTS[scenario][split] or any(r['split'] != split for r in rows):
        raise ValueError('view split/count mismatch')
    if split == 'test' and any('pkd' in row for row in rows):
        raise ValueError('gold leaked into test input view')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate input-view identity')
    expected = {'sample_id', 'split', 'peptide_sequence', 'target_sequence', 'target_sequence_role',
                'chemical_state', 'experimental_context'} | ({'pkd'} if split != 'test' else set())
    if any(set(row) != expected or (split != 'test' and not finite_real(row.get('pkd'))) for row in rows):
        raise ValueError('input view schema/label mismatch')
    return rows


def prompt_text(row, tokenizer):
    user = ('Predict experimental binding affinity from the supplied peptide and target residue memories. '
        'Memory type 0 is the peptide; memory type 1 is the full target reference protein. '
        'The reference protein can differ from the unverified experimental construct. '
        'The endpoint is pKD = -log10(KD in molar). Return exactly one JSON object '
        'with one finite numeric field pKD, without explanations or additional fields.\n'
        'Allowed chemistry and experimental context:\n' + json.dumps(context_view(row), sort_keys=True,
            separators=(',', ':'), allow_nan=False))
    template_kwargs = {'enable_thinking': False} if getattr(tokenizer, '_mm_backbone_id', '') == 'qwen3_4b_2507' else {}
    return tokenizer.apply_chat_template([
        {'role': 'system', 'content': 'You estimate protein-peptide affinity from residue representations.'},
        {'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True, **template_kwargs)


def answer_text(row):
    if not finite_real(row.get('pkd')):
        raise ValueError('finite TRAIN/VAL gold required')
    return '{"pKD":' + format(float(row['pkd']), '.2f') + '}'


def collate(rows, tokenizer, store, device='cpu', supervised=True):
    """Tensor-only model kwargs. Identity/gold envelopes remain outside the model.

    store.pooled(sequence) returns {'H15': [8,640], 'H30': [8,640]}.
    """
    import torch
    if not rows or (supervised and any(r['split'] not in ('train', 'validation') for r in rows)):
        raise ValueError('empty batch or supervised test input')
    prefixes = [tokenizer.encode(prompt_text(row, tokenizer), add_special_tokens=False) for row in rows]
    suffixes = [tokenizer.encode(answer_text(row), add_special_tokens=False) + [tokenizer.eos_token_id]
                if supervised else [] for row in rows]
    if any(not p for p in prefixes) or (supervised and any(len(s) < 2 for s in suffixes)):
        raise ValueError('empty prompt/answer tokenization')
    length = max(len(p) + len(s) for p, s in zip(prefixes, suffixes))
    ids = torch.full((len(rows), length), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    end = []
    memories = {key: [] for key in ('p15', 'p30', 't15', 't30')}
    conditions, observations = [], []
    for i, (row, prefix, suffix) in enumerate(zip(rows, prefixes, suffixes)):
        offset = length - len(prefix) - len(suffix)
        ids[i, offset:] = torch.tensor(prefix + suffix)
        attention[i, offset:] = 1
        if supervised:
            labels[i, offset + len(prefix):] = torch.tensor(suffix)
        end.append(offset + len(prefix) - 1)
        for obj, field in (('p', 'peptide_sequence'), ('t', 'target_sequence')):
            values = store.pooled(row[field])
            for layer in ('15', '30'):
                tensor = values['H' + layer].detach().float()
                if tensor.shape != (8, 640) or not torch.isfinite(tensor).all():
                    raise ValueError('paired ESM pooled cache shape/finite mismatch')
                memories[obj + layer].append(tensor)
        condition, observed = condition_encoding(row)
        conditions.append(condition); observations.append(observed)
    memory_inputs = {key: torch.stack(value).to(device) for key, value in memories.items()}
    memory_inputs.update(condition_ids=torch.tensor(conditions, dtype=torch.long, device=device),
                         observed=torch.tensor(observations, dtype=torch.bool, device=device))
    result = {'input_ids': ids.to(device), 'attention_mask': attention.to(device),
              'prompt_end_index': torch.tensor(end, dtype=torch.long, device=device), 'memory_inputs': memory_inputs}
    if supervised:
        result['labels'] = labels.to(device)
    return result


def main():
    parser = argparse.ArgumentParser(description='Prepare pinned, gold-separated A1 input views; no models')
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    manifest = prepare_views(args.project_root, args.output_root)
    print(json.dumps({'schema': manifest['schema'], 'output_root': str(Path(args.output_root).resolve())}))


if __name__ == '__main__':
    main()

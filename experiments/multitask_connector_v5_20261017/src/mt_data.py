"""Frozen task views. Identity, labels and grouping never enter condition tensors."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

TASKS = ('meltome', 'elm', 'rt')
SPLITS = {'meltome': ('train', 'validation', 'test'),
          'elm': ('train', 'validation', 'test'),
          'rt': ('train', 'selection', 'calibration', 'test')}
CONDITION_FIELDS = ('task', 'molecule_scope', 'pH', 'phase', 'stationary_phase',
                    'mobile_phase_A', 'mobile_phase_B', 'sample', 'stereochemistry', 'modifications')
SOURCE_PINS = {
 'experiments/multitask_benchmark_v1_20261017/data/meltome/human.csv': 'cfafab67d5e29188fe0846ab48cdf6aff36aaf4ccd635b7b66bb4002a939822c',
 'data/processed/bench_v02/b1_homology_locked/b1_samples.jsonl': '8ec8591fe51ea001683da01c0b91a3b3f226e9772097bbc5c2f16b8151424d8d',
 'data/processed/bench_v02/b1_homology_locked/b1_candidates_main.jsonl': 'f4bb1c6ec0c31fa6cb93319b65afb8a487766edee45780f5ad7761922c4b42ec',
 'data/processed/bench_v02/b1_homology_locked/benchmark_lock.json': '5b8ffbe6a148bf3569c552402e900d33de70877546c6aba0a82ba958c76ddef8',
 'data/processed/foundation_20261016/rt_target_diagnostic/frozen_protocol.json': '8a15cce4d1daf5d7e7ddb2aa1f85750975a3b7a60ad4bca36d535d81962520ed',
 'data/processed/foundation_20261016/rt_target_diagnostic/train.jsonl': 'b16cf3165746d827da9bdbfe7ce2e4d7592e71e0ec2bdac82cfdb2b56cfb1a50',
 'data/processed/foundation_20261016/rt_target_diagnostic/validation.jsonl': 'f61f2ae4ccad234a3ef631987af8cc2bc7ca204e259219f61e1e2f103c42645c',
 'data/processed/foundation_20261016/rt_target_diagnostic/selection.jsonl': '763d2b6c0ed2b1c87df1a34054057591d1cf7e7185fb3954ffee491afb9a9984',
 'data/processed/foundation_20261016/rt_target_diagnostic/calibration.jsonl': '498df1e684d11c4bd5813f5f430e0a35a08cbac3f14a73fc9fa740ee6414fe61',
 'data/processed/foundation_20261016/rt_target_diagnostic/test.jsonl': '03ed93ce1a963510ff50e4bf8bb76e10b4d24b022f9af6be7a60b81d03e57b5c'}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(piece)
    return digest.hexdigest()


def sequence_sha(sequence):
    return hashlib.sha256(sequence.encode('ascii')).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _write_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')


def condition_tensors_values(row):
    """Stable field-aware buckets 1..4095, with zero reserved for absent values."""
    values = row['condition_values']
    if not isinstance(values, list) or len(values) != 10:
        raise ValueError('exactly ten registered conditions required')
    ids, observed = [], []
    for field, value in zip(CONDITION_FIELDS, values):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError('condition value must be nonempty string or null')
        present = value is not None
        observed.append(present)
        ids.append(1 + int.from_bytes(hashlib.sha256((field + '\0' + value).encode()).digest()[:8], 'big') % 4095 if present else 0)
    return ids, observed


def _row(sample_id, task, split, sequence, target, positive_ids, group_id, values):
    if not isinstance(sequence, str) or not sequence or not sequence.isascii() or not sequence.isalpha() or sequence != sequence.upper():
        raise ValueError('full uppercase ASCII residue sequence required')
    if target is not None and not math.isfinite(target):
        raise ValueError('nonfinite target')
    row = dict(sample_id=sample_id, task=task, split=split, sequence=sequence,
               target=target, positive_ids=positive_ids, group_id=group_id, condition_values=values)
    condition_tensors_values(row)
    return row


def prepare_data(project_root, output_root):
    root, out = Path(project_root), Path(output_root)
    for relative, expected in SOURCE_PINS.items():
        if file_sha(root / relative) != expected:
            raise ValueError('frozen source changed: ' + relative)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('refusing to overwrite prepared data')
    out.mkdir(parents=True, exist_ok=True)
    tasks = {task: {split: [] for split in splits} for task, splits in SPLITS.items()}
    human = root / next(name for name in SOURCE_PINS if name.endswith('human.csv'))
    with human.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['sequence', 'target', 'set', 'validation']:
            raise ValueError('unexpected official human CSV schema')
        for index, raw in enumerate(reader):
            if raw['validation'] not in ('True', 'False') or raw['set'] not in ('train', 'test'):
                raise ValueError('unknown official split')
            if raw['validation'] == 'True' and raw['set'] == 'test':
                raise ValueError('test cannot also be validation')
            split = 'validation' if raw['validation'] == 'True' else raw['set']
            seq = raw['sequence']
            tasks['meltome'][split].append(_row(f'human:{index}:{sequence_sha(seq)[:16]}', 'meltome', split,
                seq, float(raw['target']), [], sequence_sha(seq), ['meltome', 'single_protein'] + [None] * 8))
    elm_root = root / 'data/processed/bench_v02/b1_homology_locked'
    catalog = read_jsonl(elm_root / 'b1_candidates_main.jsonl')
    candidate_ids = {item['candidate_id'] for item in catalog}
    if len(candidate_ids) != 31 or len(catalog) != 31 or any(set(item) != {'candidate_id', 'description'} for item in catalog):
        raise ValueError('fixed 31-description catalog required')
    for raw in read_jsonl(elm_root / 'b1_samples.jsonl'):
        split = raw['split']
        positives = raw['known_positive_candidate_ids']
        if not positives or not set(positives) <= candidate_ids:
            raise ValueError('invalid known positives')
        tasks['elm'][split].append(_row(raw['sample_id'], 'elm', split, raw['segment_sequence'], None,
            positives, raw['groups']['peptide_family'], ['elm', raw['molecule_scope']] + [None] * 8))
    rt_root = root / 'data/processed/foundation_20261016/rt_target_diagnostic'
    for split in SPLITS['rt']:
        for raw in read_jsonl(rt_root / (split + '.jsonl')):
            meta = raw['rawmetadata']
            condition = meta.get('condition_metadata', {})
            values = ['rt', 'single_peptidoform'] + [str(condition[k]) if condition.get(k) is not None else None
                for k in ('pH', 'phase', 'stationary_phase', 'mobile_phase_A', 'mobile_phase_B', 'sample')]
            values += [meta.get('stereochemistry'), json.dumps(raw.get('modifications', []), sort_keys=True, separators=(',', ':'))]
            tasks['rt'][split].append(_row(raw['sample_id'], 'rt', split, raw['peptide_sequence'], float(raw['rt_value']),
                [], raw['groups']['peptidoform'], values))
    # Preserve validation union solely as an audit view, never a checkpoint selection view.
    original_validation = read_jsonl(rt_root / 'validation.jsonl')
    validation_ids = [row['sample_id'] for row in original_validation]
    children = tasks['rt']['selection'] + tasks['rt']['calibration']
    if len(set(validation_ids)) != len(validation_ids) or set(validation_ids) != {row['sample_id'] for row in children}:
        raise ValueError('RT selection/calibration no longer partition original validation')
    lookup = {row['sample_id']: row for row in children}
    tasks['rt']['validation'] = [dict(lookup[sample_id], split='validation') for sample_id in validation_ids]
    expected = {'meltome': {'train': 7287, 'validation': 861, 'test': 1945},
                'elm': {'train': 378, 'validation': 134, 'test': 76},
                'rt': {'train': 4952, 'selection': 525, 'calibration': 530, 'test': 1063, 'validation': 1055}}
    files, task_audits = {}, {}
    for task, splits in tasks.items():
        if {split: len(rows) for split, rows in splits.items()} != expected[task]:
            raise ValueError('frozen split counts changed: ' + task)
        disjoint_ids, seq_splits = set(), collections.defaultdict(set)
        for split, rows in splits.items():
            if task == 'rt' and split == 'validation':
                continue
            ids = [row['sample_id'] for row in rows]
            if len(set(ids)) != len(ids) or disjoint_ids.intersection(ids):
                raise ValueError('duplicate/cross-split sample IDs')
            disjoint_ids.update(ids)
            for row in rows:
                seq_splits[row['sequence']].add(split)
        cross = sum(len(names) > 1 for names in seq_splits.values())
        if cross:
            raise ValueError('exact sequence split overlap: ' + task)
        task_audits[task] = {'counts': expected[task], 'unique_sequences': len(seq_splits),
            'raw_residues_unique': sum(map(len, seq_splits)), 'sequence_cross_split_overlap': cross,
            'noncanonical_residue_counts': dict(collections.Counter(c for s in seq_splits for c in s if c not in 'ACDEFGHIKLMNPQRSTVWY'))}
        for split, rows in splits.items():
            relative = f'{task}/{split}.jsonl'
            _write_rows(out / relative, rows)
            files[relative] = {'sha256': file_sha(out / relative), 'rows': len(rows)}
    _write_rows(out / 'elm/candidates.jsonl', catalog)
    files['elm/candidates.jsonl'] = {'sha256': file_sha(out / 'elm/candidates.jsonl'), 'rows': 31}
    sequences = {row['sequence'] for task in TASKS for split in SPLITS[task] for row in tasks[task][split]}
    manifest = {'schema': 'multitask-frozen-views.v1', 'source_pins': SOURCE_PINS, 'files': files,
        'tasks': task_audits, 'condition_fields': CONDITION_FIELDS, 'condition_buckets': 4096,
        'condition_policy': 'registered safe fields only; missing=0; field-aware SHA256, no IDs/classes/gold/publication metadata',
        'expected_sequences': len(sequences), 'expected_residues': sum(map(len, sequences)),
        'rt_validation_role': 'audit union; selection=checkpoint only; calibration=post-selection interval only',
        'test_labels_usage': 'schema/integrity only during preparation', 'source_code_sha256': file_sha(__file__)}
    write_json(out / 'manifest.json', manifest)
    return manifest


def load_manifest(data_root):
    manifest = json.loads((Path(data_root) / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'multitask-frozen-views.v1' or manifest.get('source_pins') != SOURCE_PINS:
        raise ValueError('unrecognized data closure')
    return manifest


def _load_registered(data_root, relative):
    manifest = load_manifest(data_root)
    path = Path(data_root) / relative
    expected = manifest['files'][relative]
    if file_sha(path) != expected['sha256']:
        raise ValueError('prepared view changed: ' + relative)
    rows = read_jsonl(path)
    if len(rows) != expected['rows']:
        raise ValueError('prepared view row count changed')
    return rows


def load_task(data_root, task, split):
    if task not in TASKS or split not in SPLITS[task] + (('validation',) if task == 'rt' else ()):
        raise ValueError('unknown task/split')
    return _load_registered(data_root, f'{task}/{split}.jsonl')


def load_catalog(data_root):
    return _load_registered(data_root, 'elm/candidates.jsonl')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_data(args.project_root, args.output_root), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

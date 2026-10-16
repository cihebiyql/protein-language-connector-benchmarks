"""Content-addressed, frozen ESM/Qwen features; sequences are never truncated."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / 'data/cache/bench_v02/panel_features_20261015'
ESM_REVISION = 'a695f6045e2e32885fa60af20c13cb35398ce30c'
QWEN_REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
AA = set('ACDEFGHIKLMNPQRSTVWY')


def content_key(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_local_models():
    """Rehash pinned local weights, tokenizer and config files before extraction."""
    source = ROOT / 'sources/model_manifest.json'
    receipts = json.loads(source.read_text(encoding='utf-8'))
    wanted = {'facebook/esm2_t30_150M_UR50D': ESM_REVISION,
              'Qwen/Qwen2.5-1.5B-Instruct': QWEN_REVISION}
    checked = []
    for row in receipts:
        if row['repo'] not in wanted:
            continue
        assert row['revision'] == wanted[row['repo']]
        path = ROOT / row['path'].replace('\\', '/')
        actual = file_sha(path)
        if actual != row['sha256']:
            raise ValueError(f'Local pinned model asset differs: {path}')
        if row['file'].endswith('.safetensors'):
            if row.get('official_sha256') != actual or not row.get('official_hash_verified'):
                raise ValueError('Weight file lacks verified official pinned-revision hash')
        checked.append({'repo': row['repo'], 'revision': row['revision'], 'file': row['file'],
                        'sha256': actual, 'official_sha256': row.get('official_sha256')})
    if {r['repo'] for r in checked if r['file'].endswith('.safetensors')} != set(wanted):
        raise ValueError('Both pinned foundation-model weights must be physically verified')
    return {'source_manifest_sha256': file_sha(source), 'verified_files': checked}


def chunk_spans(length, size=1022):
    """Zero-based half-open, disjoint contiguous chunks cover every residue."""
    if length < 1 or size < 1 or size > 1022:
        raise ValueError('Nonempty protein and chunk size 1..1022 required')
    return [[start, min(start + size, length)] for start in range(0, length, size)]


def masked_means(hidden, mask):
    weights = mask.to(hidden.dtype)
    if torch.any(weights.sum(-1) <= 0):
        raise ValueError('Empty masks cannot be pooled')
    return (hidden.float() * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True)


def protein_payload(sequence, residues, origin):
    if not sequence or not set(sequence) <= AA:
        raise ValueError('Canonical nonempty protein sequence required')
    if residues.shape != (len(sequence), 640) or not torch.isfinite(residues).all():
        raise ValueError('Residue features do not cover the full sequence')
    return {'schema_version': 1, 'sequence_sha256': content_key(sequence), 'sequence': sequence,
            'length': len(sequence), 'residues30': residues.to(torch.float16).contiguous(),
            'mean30': residues.float().mean(0), 'chunk_spans': chunk_spans(len(sequence)),
            'model_revision': ESM_REVISION, 'bos_eos_padding_removed': True,
            'chunk_rule': 'nonoverlapping contiguous <=1022 residues; concatenate in original order; no cross-chunk contextual attention',
            'origin': origin, 'frozen': True, 'training_labels_used': False}


def protein_file(cache, sequence):
    return Path(cache) / 'proteins' / f'{content_key(sequence)}.pt'


def text_file(cache, text):
    return Path(cache) / 'texts' / f'{content_key(text)}.pt'


def _save_atomic(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    torch.save(value, temporary)
    temporary.replace(path)


def _check_protein(path, sequence):
    value = torch.load(path, weights_only=True, map_location='cpu')
    assert value['sequence_sha256'] == content_key(sequence) and value['sequence'] == sequence
    assert value['model_revision'] == ESM_REVISION and value['frozen']
    assert value['residues30'].shape == (len(sequence), 640)
    assert value['chunk_spans'] == chunk_spans(len(sequence))
    assert torch.isfinite(value['residues30']).all()
    return {'path': str(path.name), 'length': len(sequence), 'sha256': file_sha(path),
            'chunk_spans': value['chunk_spans'], 'origin': value['origin']}


def reuse_raw_legacy(cache, sequences):
    """Reuse only frozen raw residues with the identical pinned ESM revision."""
    path = ROOT / 'data/cache/real_residues.pt'
    if not path.is_file():
        return 0
    old = torch.load(path, weights_only=True, map_location='cpu')
    if old.get('model_revision') != ESM_REVISION or not old.get('bos_eos_padding_removed'):
        return 0
    requested = set(sequences)
    count = 0
    for i, sequence in enumerate(old['sequences']):
        dest = protein_file(cache, sequence)
        if sequence not in requested or dest.exists():
            continue
        mask = old['mask'][i].bool()
        assert int(mask.sum()) == len(sequence)
        residues = old['features30'][i][mask].float()
        _save_atomic(protein_payload(sequence, residues, 'identical-revision frozen raw residue cache; no trained weights'), dest)
        count += 1
    return count


def encode_proteins(sequences, cache, device='cuda', token_budget=4096, max_batch=64):
    from transformers import AutoModel, AutoTokenizer
    pending = [seq for seq in sorted(set(sequences), key=lambda seq: (len(seq), seq))
               if not protein_file(cache, seq).exists()]
    if not pending:
        return
    model_path = ROOT / 'models/facebook--esm2_t30_150M_UR50D'
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True,
                                     dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
                                     add_pooling_layer=False).to(device).eval().requires_grad_(False)
    jobs = [(sequence, start, end) for sequence in pending for start, end in chunk_spans(len(sequence))]
    jobs.sort(key=lambda item: (item[2] - item[1], content_key(item[0]), item[1]))
    pieces = {}
    completed, cursor = 0, 0
    with torch.inference_mode():
        while cursor < len(jobs):
            batch = []
            while cursor < len(jobs):
                job = jobs[cursor]
                new_length = job[2] - job[1] + 2
                if batch and (len(batch) >= max_batch or (len(batch) + 1) * new_length > token_budget):
                    break
                batch.append(job)
                cursor += 1
            tokens = tokenizer([sequence[start:end] for sequence, start, end in batch],
                               padding=True, truncation=False, return_tensors='pt').to(device)
            hidden = model(**tokens).last_hidden_state
            for i, (sequence, start, end) in enumerate(batch):
                assert int(tokens['attention_mask'][i].sum()) == end - start + 2
                value = hidden[i, 1:end - start + 1].float().cpu()
                pieces.setdefault(sequence, {})[start] = value
                spans = chunk_spans(len(sequence))
                if len(pieces[sequence]) == len(spans):
                    residues = torch.cat([pieces[sequence][a] for a, _ in spans])
                    _save_atomic(protein_payload(sequence, residues, 'fresh frozen ESM2 forward only'),
                                 protein_file(cache, sequence))
                    del pieces[sequence]
                    completed += 1
            print(json.dumps({'protein_sequences_completed': completed, 'pending_total': len(pending),
                              'chunks_done': cursor, 'chunks_total': len(jobs)}), flush=True)
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()


def encode_texts(texts, cache, device='cuda', batch_size=8):
    from transformers import AutoModel, AutoTokenizer
    pending = [text for text in sorted(set(texts)) if not text_file(cache, text).exists()]
    if not pending:
        return
    model_path = ROOT / 'models/Qwen--Qwen2.5-1.5B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_path, local_files_only=True,
                                     dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
                                     attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    with torch.inference_mode():
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            tokens = tokenizer(batch, padding=True, truncation=False, return_tensors='pt').to(device)
            if tokens['input_ids'].shape[1] > model.config.max_position_embeddings:
                raise ValueError('Text exceeds model context; no silent truncation allowed')
            hidden = model(**tokens, use_cache=False).last_hidden_state
            means = masked_means(hidden, tokens['attention_mask']).cpu()
            for i, text in enumerate(batch):
                n = int(tokens['attention_mask'][i].sum())
                _save_atomic({'schema_version': 1, 'text_sha256': content_key(text), 'text': text,
                              'mean': means[i], 'last': hidden[i, n - 1].float().cpu(), 'n_tokens': n,
                              'model_revision': QWEN_REVISION, 'frozen': True,
                              'input_rule': 'plain anonymous rule-free function description only; no class id, sample, label, split, taxonomy or regex'},
                             text_file(cache, text))
            print(json.dumps({'descriptions_completed': min(start + batch_size, len(pending)),
                              'pending_total': len(pending)}), flush=True)
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--a1-source', type=Path, default=ROOT / 'data/processed/bench_v02/a1_broad_candidates.jsonl')
    parser.add_argument('--b1-source', type=Path, default=ROOT / 'data/processed/bench_v02/b1_model_eukaryotes/b1_samples.jsonl')
    parser.add_argument('--text-source', type=Path, default=ROOT / 'data/processed/bench_v02/b1_homology_locked/b1_candidates_main.jsonl')
    parser.add_argument('--mode', choices=['proteins', 'texts', 'all'], default='all')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.perf_counter()
    verified_models = verify_local_models()
    a1, b1 = read_jsonl(args.a1_source), read_jsonl(args.b1_source)
    sequences = {row[key] for row in a1 for key in ('peptide_sequence', 'target_sequence')}
    sequences.update(row['segment_sequence'] for row in b1)
    assert all(sequence and set(sequence) <= AA for sequence in sequences)
    args.cache.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'schema_version': 1, 'model_revisions': {'ESM2': ESM_REVISION, 'Qwen': QWEN_REVISION},
        'proteins': {}, 'texts': {}, 'training_labels_used': False,
        'protein_rule': 'all residues; disjoint contiguous chunks <=1022; concatenate after per-chunk BOS/EOS removal; no silent truncation',
        'text_rule': 'full plain rule-free description, frozen Qwen last-token and attention-mask mean'}
    if args.mode in ('proteins', 'all'):
        reused = reuse_raw_legacy(args.cache, sequences)
        print(json.dumps({'identical_revision_raw_residue_reuse': reused}), flush=True)
        encode_proteins(sequences, args.cache, args.device)
        manifest['proteins'].update({content_key(seq): _check_protein(protein_file(args.cache, seq), seq)
                                     for seq in sorted(sequences)})
        manifest['protein_sources'] = {str(path): file_sha(path) for path in (args.a1_source, args.b1_source)}
    if args.mode in ('texts', 'all'):
        texts = [row['description'] for row in read_jsonl(args.text_source)]
        assert texts and all(text.strip() for text in texts)
        encode_texts(texts, args.cache, args.device)
        for text in texts:
            path = text_file(args.cache, text)
            value = torch.load(path, weights_only=True, map_location='cpu')
            assert value['text'] == text and value['model_revision'] == QWEN_REVISION
            assert value['last'].shape == value['mean'].shape == (1536,)
            manifest['texts'][content_key(text)] = {'path': path.name, 'sha256': file_sha(path),
                                                   'n_tokens': value['n_tokens']}
        manifest['text_source'] = {'path': str(args.text_source), 'sha256': file_sha(args.text_source)}
    manifest['last_operation_seconds'] = time.perf_counter() - started
    manifest['verified_local_models'] = verified_models
    manifest['cache_script_sha256'] = file_sha(__file__)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'cache': str(args.cache), 'proteins': len(manifest['proteins']),
                      'texts': len(manifest['texts']), 'seconds': manifest['last_operation_seconds']}), flush=True)


if __name__ == '__main__':
    main()

"""Offline, label-free ESM pair extraction and frozen descriptor embeddings."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .mt_data import TASKS, SPLITS, file_sha, sequence_sha, load_task, load_catalog, load_manifest

ESM_REVISION = 'a695f6045e2e32885fa60af20c13cb35398ce30c'
ESM_PINS = {
 'README.md': '462a2f24724e19c6be0efab926315c294a863c9a9770e2c8b3d859b2d81a07de',
 'config.json': 'e512f68ec444d99477703a9806639ca83da3dbc19f6c5fe428d6e5b7460972dc',
 'model.safetensors': 'c3f1da8aea53bddd32c246c86168c23b9fd72341fb9db9a94436f855f5053566',
 'special_tokens_map.json': '3aedcd4211c0d43aec4e607ff60a63255f3174ead795e997350f09a5f8cd9ee1',
 'tokenizer_config.json': '7e9161ecdb548ec45a41cbc6b24aa4476fdd418461f491c4207baa99419a29ad',
 'vocab.txt': '0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03'}
SEMANTICS = {'p15': 'encoder.layer[14] output before final encoder normalization',
             'p30': 'EsmModel.last_hidden_state after final encoder normalization'}
QWEN25_PINS = {
 'config.json': '98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670',
 'generation_config.json': 'e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6',
 'merges.txt': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3',
 'model.safetensors': 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee',
 'tokenizer.json': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
 'tokenizer_config.json': '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583',
 'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(_canonical(value) + '\n', encoding='utf-8')
    temporary.replace(path)


def chunk_spans(length):
    if type(length) is not int or length < 1:
        raise ValueError('positive sequence length required')
    return [(start, min(start + 1022, length)) for start in range(0, length, 1022)]


def installed_semantics():
    from transformers.models.esm import modeling_esm
    source = Path(inspect.getfile(modeling_esm))
    encoder = inspect.getsource(modeling_esm.EsmEncoder.forward)
    model = inspect.getsource(modeling_esm.EsmModel.forward)
    a, b = encoder.find('hidden_states = layer_outputs[0]'), encoder.find('hidden_states = self.emb_layer_norm_after(hidden_states)')
    verified = a >= 0 and b > a and 'last_hidden_state=hidden_states' in encoder and 'sequence_output = encoder_outputs[0]' in model
    if not verified or importlib.metadata.version('transformers') != '4.51.3':
        raise ValueError('requires statically qualified Transformers4.51.3 ESM implementation')
    return {'transformers_version': '4.51.3', 'source_sha256': file_sha(source), 'verified': True,
            'encoder_forward': encoder, 'model_forward': model}


def extract_sequence(model, tokenizer, sequence, device='cpu'):
    """Full ordered disjoint chunks, one pass for both layers, float32 pool8."""
    import torch
    import torch.nn.functional as F
    if not sequence or not sequence.isascii() or not sequence.isalpha() or sequence != sequence.upper():
        raise ValueError('full uppercase sequence required; native U is retained')
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('frozen extractor required')
    model.eval()
    pieces = {'p15': [], 'p30': []}
    spans = chunk_spans(len(sequence))
    for start, stop in spans:
        captured = []
        def hook(_module, _args, output):
            captured.append(output[0] if isinstance(output, (tuple, list)) else output)
        handle = model.encoder.layer[14].register_forward_hook(hook)
        try:
            encoded = tokenizer(sequence[start:stop], return_tensors='pt', add_special_tokens=True, truncation=False)
            ids = encoded['input_ids']
            if tuple(ids.shape) != (1, stop - start + 2):
                raise ValueError('tokenizer is not one residue per token')
            # Length alone cannot detect a dropped U replaced with unknown; verify exact residue tokens.
            expected = tokenizer.convert_tokens_to_ids(list(sequence[start:stop]))
            if ids[0, 1:-1].tolist() != expected or tokenizer.unk_token_id in expected:
                raise ValueError('tokenizer changed a residue or emitted unknown token')
            with torch.no_grad():
                result = model(**{k: v.to(device) for k, v in encoded.items()}, return_dict=True, output_hidden_states=False)
            if len(captured) != 1:
                raise ValueError('block15 hook count mismatch')
            for key, value in (('p15', captured[0]), ('p30', result.last_hidden_state)):
                if value.shape[:2] != ids.shape or not torch.isfinite(value).all():
                    raise ValueError('invalid/nonfinite residue states')
                pieces[key].append(value[0, 1:-1].detach().float().cpu())
        finally:
            handle.remove()
    arrays = {}
    for key, chunks in pieces.items():
        raw = torch.cat(chunks, dim=0)
        if raw.shape[0] != len(sequence):
            raise ValueError('incomplete residue coverage')
        arrays[key] = F.adaptive_avg_pool1d(raw.T.unsqueeze(0), 8)[0].T.contiguous().numpy()
        if arrays[key].dtype != np.float32 or not np.isfinite(arrays[key]).all():
            raise ValueError('invalid pooled states')
    return arrays, {'sequence_sha256': sequence_sha(sequence), 'length': len(sequence),
        'raw_residues_covered': len(sequence), 'raw_full_coverage': True, 'chunk_spans': spans,
        'cross_chunk_attention': False, 'pooling': 'full ordered residue concatenation then FP32 adaptive_average8',
        'semantics': SEMANTICS}


class FeatureStore:
    def __init__(self, cache_root):
        self.root = Path(cache_root)
        self.spec = json.loads((self.root / 'cache_spec.json').read_text(encoding='utf-8'))
        self.spec_sha256 = file_sha(self.root / 'cache_spec.json')
        if (self.spec.get('schema') != 'multitask-esm-cache.v1' or self.spec.get('model_revision') != ESM_REVISION
                or self.spec.get('model_pins') != ESM_PINS or self.spec.get('semantics') != SEMANTICS
                or self.spec.get('source_semantics', {}).get('transformers_version') != '4.51.3'
                or not self.spec.get('source_semantics', {}).get('verified')):
            raise ValueError('unqualified/new-runtime paired cache closure')
        self.verified = {}

    def get(self, sequence):
        sha = sequence_sha(sequence)
        path = self.root / 'sequences' / (sha + '.npz')
        receipt = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if (receipt.get('cache_spec_sha256') != self.spec_sha256 or receipt.get('sequence_sha256') != sha
                or receipt.get('length') != len(sequence) or receipt.get('raw_residues_covered') != len(sequence)
                or receipt.get('raw_full_coverage') is not True
                or [tuple(x) for x in receipt.get('chunk_spans', [])] != chunk_spans(len(sequence))
                or file_sha(path) != receipt.get('sha256')):
            raise ValueError('feature identity/integrity/coverage mismatch')
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {'p15', 'p30'}:
                raise ValueError('paired payload required')
            arrays = {name: payload[name].copy() for name in payload.files}
        if any(array.dtype != np.float32 or array.shape != (8, 640) or not np.isfinite(array).all() for array in arrays.values()):
            raise ValueError('finite float32[8,640] feature pair required')
        self.verified[sha] = receipt['sha256']
        return arrays

    def verify_all(self, rows):
        sequences = {row['sequence'] for row in rows}
        for sequence in sorted(sequences, key=sequence_sha):
            self.get(sequence)
        return {'status': 'complete', 'sequences': len(sequences), 'residues': sum(map(len, sequences)),
                'cache_spec_sha256': self.spec_sha256, 'payloads': {sequence_sha(s): self.verified[sequence_sha(s)] for s in sorted(sequences)}}


def all_rows(data_root):
    return [row for task in TASKS for split in SPLITS[task] for row in load_task(data_root, task, split)]


def _select_gpu(gpu, allocator_gib):
    if gpu not in (3, 4, 5) or allocator_gib <= 0:
        raise ValueError('only registered physical GPU3/4/5 and positive allocator budget allowed')
    # Query only memory; utilization and existing compute processes are not gates.
    fields = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()
    if float(fields) < allocator_gib * 1024 + 512:
        raise RuntimeError('insufficient actual free VRAM for explicit allocator plus512MiB')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('exactly one visible registered GPU required')
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(allocator_gib * 1024**3 / total, 0)
    return torch


def extract(args):
    # Set device visibility before importing Torch/Transformers.
    torch = _select_gpu(args.gpu, args.allocator_gib)
    from transformers import AutoTokenizer, EsmModel
    if not 0 <= args.shard_index < args.shards or args.shards < 1:
        raise ValueError('invalid shard index/count')
    model_root, root = Path(args.esm_path), Path(args.cache_root)
    for relative, expected in ESM_PINS.items():
        if file_sha(model_root / relative) != expected:
            raise ValueError('pinned ESM file changed: ' + relative)
    semantics = installed_semantics()
    sequences = sorted({r['sequence'] for r in all_rows(args.data_root)}, key=sequence_sha)
    spec = {'schema': 'multitask-esm-cache.v1', 'model_revision': ESM_REVISION, 'model_pins': ESM_PINS,
        'source_semantics': semantics, 'semantics': SEMANTICS, 'extractor_sha256': file_sha(__file__),
        'torch_version': torch.__version__, 'parameter_dtype': 'float32', 'payload_dtype': 'float32',
        'data_manifest_sha256': file_sha(Path(args.data_root) / 'manifest.json'),
        'sequences_sha256': hashlib.sha256('\n'.join(map(sequence_sha, sequences)).encode()).hexdigest(),
        'expected_sequences': len(sequences), 'expected_residues': sum(map(len, sequences)), 'label_free': True,
        'numerical_policy': {'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
            'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cpu_threads': torch.get_num_threads(), 'attention_implementation': 'eager'},
        'pooling': 'full disjoint1022-residue chunks then FP32 adaptive_average8', 'cross_chunk_attention': False}
    root.mkdir(parents=True, exist_ok=True)
    (root / 'sequences').mkdir(exist_ok=True)
    spec_path = root / 'cache_spec.json'
    try:
        with spec_path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(_canonical(spec) + '\n')
    except FileExistsError:
        # Multiple independent shards may initialize simultaneously; allow the first write to finish.
        for attempt in range(20):
            try:
                old = json.loads(spec_path.read_text(encoding='utf-8'))
                break
            except json.JSONDecodeError:
                time.sleep(0.05)
        else:
            raise ValueError('cache spec initialization did not complete')
        if old != spec:
            raise ValueError('resume requires identical model, source, pooling and frozen views')
    store = FeatureStore(root)
    assigned = sequences[args.shard_index::args.shards]
    if args.max_sequences is not None:
        if args.max_sequences < 1:
            raise ValueError('max-sequences must be positive')
        assigned = assigned[:args.max_sequences]
    model = EsmModel.from_pretrained(model_root, local_files_only=True, torch_dtype=torch.float32,
                                     add_pooling_layer=False, attn_implementation='eager').to('cuda:0')
    if model.config.hidden_size != 640 or model.config.num_hidden_layers != 30:
        raise ValueError('pinned ESM2-150M30block640width required')
    model.requires_grad_(False).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
    start, completed = time.monotonic(), 0
    for sequence in assigned:
        sha = sequence_sha(sequence)
        path = root / 'sequences' / (sha + '.npz')
        if path.with_suffix('.json').exists():
            store.get(sequence)
        else:
            if path.exists():
                raise FileExistsError('orphan payload retained; inspect rather than overwrite: ' + str(path))
            arrays, receipt = extract_sequence(model, tokenizer, sequence, 'cuda:0')
            with path.open('xb') as stream:
                np.savez(stream, **arrays)
            receipt.update(sha256=file_sha(path), cache_spec_sha256=store.spec_sha256)
            with path.with_suffix('.json').open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(_canonical(receipt) + '\n')
        completed += 1
        if completed % 25 == 0 or completed == len(assigned):
            status = {'shard_index': args.shard_index, 'shards': args.shards, 'completed': completed,
                      'assigned': len(assigned), 'elapsed_seconds': time.monotonic() - start,
                      'status': 'complete' if completed == len(assigned) else 'running',
                      'cache_spec_sha256': store.spec_sha256, 'partial_limit': args.max_sequences,
                      'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}
            _atomic_json(root / f'shard_{args.shard_index}_of_{args.shards}.json', status)
            print(_canonical(status), flush=True)


def descriptions(args):
    torch = _select_gpu(args.gpu, args.allocator_gib)
    from transformers import AutoModel, AutoTokenizer
    if importlib.metadata.version('transformers') != '4.51.3':
        raise ValueError('descriptor cache requires pinned Transformers4.51.3')
    root = Path(args.cache_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / 'description_vectors.npz'
    if destination.exists():
        raise FileExistsError('descriptor cache already exists; refusing overwrite')
    catalog = load_catalog(args.data_root)
    model_root = Path(args.qwen_path)
    # Explicit offline folder; exact asset hashes become part of descriptor receipt.
    pins = {path.name: file_sha(path) for path in sorted(model_root.iterdir()) if path.is_file()}
    if any(pins.get(name) != expected for name, expected in QWEN25_PINS.items()):
        raise ValueError('descriptor encoder weights/tokenizer differ from pinned Qwen2.5 revision')
    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
    model = AutoModel.from_pretrained(model_root, local_files_only=True, torch_dtype=torch.bfloat16,
                                     attn_implementation='eager').to('cuda:0')
    if model.config.hidden_size != 1536 or model.config.num_hidden_layers != 28:
        raise ValueError('shared descriptor encoder must be Qwen2.5-1.5B')
    model.requires_grad_(False).eval()
    vectors = []
    for entry in catalog:
        encoded = tokenizer(entry['description'], return_tensors='pt', add_special_tokens=True, truncation=False)
        encoded = {k: v.to('cuda:0') for k, v in encoded.items()}
        with torch.no_grad():
            states = model(**encoded, return_dict=True).last_hidden_state[0].float()
        mask = encoded['attention_mask'][0].bool()
        vectors.append(states[mask].mean(dim=0).cpu().numpy())
    array = np.stack(vectors)
    if array.shape != (31, 1536) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError('invalid descriptor vectors')
    with destination.open('xb') as stream:
        np.savez(stream, candidate_ids=np.asarray([r['candidate_id'] for r in catalog], dtype='U'), vectors=array)
    _atomic_json(root / 'description_vectors.json', {'schema': 'frozen-descriptor-cache.v1',
        'encoder': 'Qwen/Qwen2.5-1.5B-Instruct', 'revision': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
        'model_pins': pins, 'catalog_sha256': file_sha(Path(args.data_root) / 'elm/candidates.jsonl'),
        'sha256': file_sha(destination), 'source_sha256': file_sha(__file__), 'transformers_version': '4.51.3',
        'pooling': 'FP32 mean of frozen final descriptor token states including tokenizer special tokens; attention mask only',
        'memory_injection': False, 'fitted_parameters': 0, 'shape': [31, 1536]})


def finalize(args):
    root = Path(args.cache_root)
    store = FeatureStore(root)
    receipt = store.verify_all(all_rows(args.data_root))
    manifest = load_manifest(args.data_root)
    if receipt['sequences'] != manifest['expected_sequences'] or receipt['residues'] != manifest['expected_residues']:
        raise ValueError('full sequence/residue coverage differs')
    receipt.update(schema='multitask-esm-complete.v1', data_manifest_sha256=file_sha(Path(args.data_root) / 'manifest.json'))
    with (root / 'manifest.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(_canonical(receipt) + '\n')
    print(_canonical({k: v for k, v in receipt.items() if k != 'payloads'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('extract', 'descriptions', 'finalize'):
        child = sub.add_parser(name)
        child.add_argument('--data-root', required=True)
        child.add_argument('--cache-root', required=True)
        if name != 'finalize':
            child.add_argument('--gpu', type=int, required=True)
            child.add_argument('--allocator-gib', type=float, default=2 if name == 'extract' else 8)
        if name == 'extract':
            child.add_argument('--esm-path', required=True)
            child.add_argument('--shard-index', type=int, default=0)
            child.add_argument('--shards', type=int, default=1)
            child.add_argument('--max-sequences', type=int)
        if name == 'descriptions':
            child.add_argument('--qwen-path', required=True)
    args = parser.parse_args()
    {'extract': extract, 'descriptions': descriptions, 'finalize': finalize}[args.command](args)


if __name__ == '__main__':
    main()

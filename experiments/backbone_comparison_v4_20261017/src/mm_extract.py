"""Offline paired ESM block15 / normalized layer30 extraction, no implicit download.

Importing this module does not import torch or load any model. CLI inspect only
reads installed implementation source. CLI extract is an explicit costly task.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import time
from datetime import datetime, timezone

from .mm_data import canonical, file_sha, read_input_view, registered_path, strict_loads, verify_sha, SCENARIOS

ESM_REVISION = 'a695f6045e2e32885fa60af20c13cb35398ce30c'
REMOTE_TF4481_SOURCE_SHA = '6f867571aeee0fb9f444cb23ac5e8ca189aa4b05424aad986c9c74b6ce412cb8'
FROZEN_EXTRACTOR_SHA = '7266a589c638530fb178df3c46997e8f3f27234fc99d143ace1300e6a9554e14'
FROZEN_CACHE_MANIFEST_SHA = '24ebf3118a9e00ff8a1c30ee289a5202577ae4f44d1e2d67092db7865e778d88'
MODEL_PINS = {
    'README.md': '462a2f24724e19c6be0efab926315c294a863c9a9770e2c8b3d859b2d81a07de',
    'config.json': 'e512f68ec444d99477703a9806639ca83da3dbc19f6c5fe428d6e5b7460972dc',
    'model.safetensors': 'c3f1da8aea53bddd32c246c86168c23b9fd72341fb9db9a94436f855f5053566',
    'special_tokens_map.json': '3aedcd4211c0d43aec4e607ff60a63255f3174ead795e997350f09a5f8cd9ee1',
    'tokenizer_config.json': '7e9161ecdb548ec45a41cbc6b24aa4476fdd418461f491c4207baa99419a29ad',
    'vocab.txt': '0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03'}
SEMANTICS = {'H15': 'encoder.layer[14] output; post block15, before final encoder emb_layer_norm_after',
             'H30': 'EsmModel.last_hidden_state; post block30 and final encoder emb_layer_norm_after'}


def sequence_sha(sequence):
    return hashlib.sha256(sequence.encode('ascii')).hexdigest()


def chunk_spans(length, max_residues=1022):
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError('positive residue length required')
    if not isinstance(max_residues, int) or not 1 <= max_residues <= 1022:
        raise ValueError('chunk residues must be 1..1022')
    return [(start, min(start + max_residues, length)) for start in range(0, length, max_residues)]


def inspect_esm_semantics():
    """Static installed-source receipt, never constructs a model or CUDA tensor."""
    from transformers.models.esm import modeling_esm
    encoder = inspect.getsource(modeling_esm.EsmEncoder.forward)
    model = inspect.getsource(modeling_esm.EsmModel.forward)
    source = Path(inspect.getfile(modeling_esm))
    # Fail closed on unfamiliar implementations; preserve exact source in receipt.
    block_at = encoder.find('hidden_states = layer_outputs[0]')
    norm_at = encoder.find('hidden_states = self.emb_layer_norm_after(hidden_states)')
    verified = (block_at >= 0 and norm_at > block_at and 'last_hidden_state=hidden_states' in encoder
                and 'sequence_output = encoder_outputs[0]' in model
                and 'last_hidden_state=sequence_output' in model)
    return {'transformers_version': importlib.metadata.version('transformers'),
            'source_path': str(source), 'source_sha256': file_sha(source),
            'encoder_forward': encoder, 'model_forward': model, 'verified': verified,
            'semantics': SEMANTICS, 'method': 'static_source_only_no_model_or_forward'}


def extract_sequence(model, tokenizer, sequence, device='cpu', max_residues=1022, storage_dtype='float32'):
    """Same pass per chunk captures both layers, then pools concatenated residues.

    CPU tiny fixtures may have arbitrary hidden width; production CLI separately
    requires the pinned 30-block, 640-wide config. Never retains all hidden states.
    """
    import torch
    from torch.nn import functional as F
    if storage_dtype not in ('float32', 'float16'):
        raise ValueError('unsupported storage dtype')
    if not isinstance(sequence, str) or not sequence or not set(sequence) <= set('ACDEFGHIKLMNPQRSTVWY'):
        raise ValueError('nonempty ASCII uppercase residue sequence required')
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('extractor requires explicitly frozen backbone')
    blocks = model.encoder.layer
    if len(blocks) < 15:
        raise ValueError('model must expose encoder.layer[14]')
    spans = chunk_spans(len(sequence), max_residues)
    collected = {'H15': [], 'H30': []}
    chunk_receipts = []
    for start, stop in spans:
        captured = []
        def hook(_module, _args, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            captured.append(value)
        handle = blocks[14].register_forward_hook(hook)
        try:
            encoded = tokenizer(sequence[start:stop], return_tensors='pt', add_special_tokens=True,
                                truncation=False)
            ids = encoded['input_ids']
            if tuple(ids.shape) != (1, stop - start + 2):
                raise ValueError('tokenizer must yield exactly BOS + one token/residue + EOS')
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                output = model(**encoded, output_hidden_states=False, return_dict=True)
            if len(captured) != 1:
                raise ValueError('block15 hook must fire exactly once per forward')
            paired = {'H15': captured[0], 'H30': output.last_hidden_state}
            for key, value in paired.items():
                if value.ndim != 3 or value.shape[:2] != ids.shape or not torch.isfinite(value).all():
                    raise ValueError('invalid/nonfinite layer output')
                collected[key].append(value[0, 1:-1].detach().float().cpu())
            chunk_receipts.append({'start': start, 'stop': stop, 'residues': stop - start,
                                   'tokens_with_bos_eos': stop - start + 2})
        finally:
            handle.remove()
    tensors, distortions = {}, {}
    for key, pieces in collected.items():
        raw = torch.cat(pieces, dim=0)
        if raw.shape[0] != len(sequence):
            raise ValueError('incomplete raw residue coverage')
        pooled = F.adaptive_avg_pool1d(raw.T.unsqueeze(0), 8).squeeze(0).T.contiguous()
        # Pool interval reconstruction is diagnostic, not a loss or fitted map.
        reconstructed = torch.zeros_like(raw)
        overlaps = torch.zeros(len(sequence), 1)
        intervals = []
        for slot in range(8):
            a, b = slot * len(sequence) // 8, ((slot + 1) * len(sequence) + 7) // 8
            reconstructed[a:b] += pooled[slot]
            overlaps[a:b] += 1
            intervals.append([a, b])
        reconstructed /= overlaps
        distortions[key] = {'reconstruction_mse': float((raw - reconstructed).square().mean()),
            'raw_mean_square': float(raw.square().mean()), 'adaptive_pool_intervals': intervals,
            'diagnostic': 'overlap-averaged slot broadcast; no inverse or lossless claim'}
        tensors[key] = pooled.to(getattr(torch, storage_dtype))
        if not torch.isfinite(tensors[key]).all():
            raise ValueError('nonfinite saved pooled tensor')
    return tensors, {'sequence_sha256': sequence_sha(sequence), 'length': len(sequence),
        'chunk_spans': chunk_receipts, 'raw_residues_covered': len(sequence),
        'raw_full_coverage': True, 'whole_sequence_joint_attention': len(spans) == 1,
        'cross_chunk_attention': False, 'raw_residues_saved': False, 'pooled_slots': 8,
        'pooling': 'float32 adaptive_avg_pool1d after full ordered chunk concatenation',
        'pooling_is_lossy': True, 'distortion': distortions, 'semantics': SEMANTICS}


class SequenceStore:
    """Verified paired pooled payloads; no old layer30 fallback or inferred layer15."""
    def __init__(self, cache_root, *, manifest_sha256=FROZEN_CACHE_MANIFEST_SHA):
        self.root = Path(cache_root).resolve(strict=True)
        self.manifest_path = registered_path(self.root, 'manifest.json')
        self.manifest_sha256 = file_sha(self.manifest_path)
        if self.manifest_sha256 != manifest_sha256:
            raise ValueError('paired cache manifest differs from immutable source closure')
        self.manifest = strict_loads(self.manifest_path.read_text(encoding='utf-8'))
        if self.manifest.get('schema') != 'mm_paired_esm_pooled_v1' or self.manifest.get('model_revision') != ESM_REVISION:
            raise ValueError('unrecognized paired cache schema/revision')
        if (self.manifest.get('cache_status') != 'complete' or self.manifest.get('phase') != 'full'
                or len(self.manifest.get('sequences', {})) != self.manifest.get('expected_sequences')):
            raise ValueError('incomplete/canary cache cannot supply formal training')
        if self.manifest.get('semantics') != SEMANTICS or not self.manifest.get('source_semantics', {}).get('verified'):
            raise ValueError('cache lacks verified layer semantics')
        if self.manifest.get('model_pins') != MODEL_PINS:
            raise ValueError('cache model pins differ from frozen six-file closure')
        if (self.manifest['source_semantics'].get('source_sha256') != REMOTE_TF4481_SOURCE_SHA
                or self.manifest['source_semantics'].get('transformers_version') != '4.48.1'):
            raise ValueError('cache installed source differs from qualified node1 implementation')
        if self.manifest.get('extractor_sha256') != FROZEN_EXTRACTOR_SHA:
            raise ValueError('cache extractor source differs from frozen producing code')
        # These are producer source semantics. Reading tensor-only frozen payloads
        # never consults or executes the consumer runtime installed ESM code.
        self.receipts = {}

    def pooled(self, sequence):
        import torch
        sha = sequence_sha(sequence)
        entry = self.manifest['sequences'][sha]
        path = registered_path(self.root, entry['path'])
        verify_sha(path, entry['sha256'])
        if entry['length'] != len(sequence) or entry['raw_residues_covered'] != len(sequence) or not entry['raw_full_coverage']:
            raise ValueError('sequence/coverage mismatch')
        cursor = 0
        for chunk in entry['chunk_spans']:
            if chunk['start'] != cursor or not cursor < chunk['stop'] <= min(cursor + 1022, len(sequence)):
                raise ValueError('noncontiguous/invalid raw chunk coverage')
            cursor = chunk['stop']
        if cursor != len(sequence):
            raise ValueError('incomplete raw chunk coverage')
        # weights_only prevents arbitrary pickle object loading; payload is tensors only.
        values = torch.load(path, map_location='cpu', weights_only=True)
        if set(values) != {'H15', 'H30'} or any(v.shape != (8, 640) or not torch.isfinite(v).all() for v in values.values()):
            raise ValueError('invalid paired payload')
        self.receipts[sha] = {'path': entry['path'], 'sha256': entry['sha256'], 'bytes': path.stat().st_size}
        return values

    def provenance(self):
        return {'manifest_sha256': self.manifest_sha256, 'model_revision': ESM_REVISION,
                'verified_payloads': dict(self.receipts), 'source_semantics': self.manifest['source_semantics']}

    def verify(self):
        """Hash every registered payload without tensor deserialization/forward."""
        for sha, entry in self.manifest['sequences'].items():
            length = entry.get('length')
            if (type(length) is not int or length < 1 or entry.get('raw_residues_covered') != length
                    or entry.get('raw_full_coverage') is not True):
                raise ValueError('cache raw residue coverage receipt differs')
            spans = entry.get('chunk_spans', [])
            if [(span.get('start'), span.get('stop')) for span in spans] != chunk_spans(length):
                raise ValueError('cache chunk coverage differs from frozen disjoint policy')
            if sha not in self.manifest.get('expected_sequence_shas', []):
                raise ValueError('unregistered sequence digest')
            path = registered_path(self.root, entry['path'])
            verify_sha(path, entry['sha256'])
            self.receipts[sha] = {'path': entry['path'], 'sha256': entry['sha256'], 'bytes': path.stat().st_size}
        if (set(self.manifest['sequences']) != set(self.manifest.get('expected_sequence_shas', []))
                or sum(entry['length'] for entry in self.manifest['sequences'].values()) != self.manifest.get('expected_raw_residues')
                or self.manifest.get('raw_residues_covered') != self.manifest.get('expected_raw_residues')):
            raise ValueError('cache complete sequence/residue coverage differs')
        return self.provenance()


def extract_cache(model, tokenizer, sequences, output_root, source_semantics, model_pins,
                  device='cpu', storage_dtype='float32', max_residues=1022, provenance=None,
                  phase='full', progress_every=25):
    import torch
    if not source_semantics.get('verified'):
        raise ValueError('installed ESM semantics must qualify before extraction')
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'sequences').mkdir()
    sequences = sorted(set(sequences), key=sequence_sha)
    if not sequences or phase not in ('full', 'train-canary') or progress_every < 1:
        raise ValueError('nonempty sequences, registered phase and positive progress interval required')
    started = time.monotonic()
    manifest = {'schema': 'mm_paired_esm_pooled_v1', 'model_revision': ESM_REVISION,
        'phase': phase, 'cache_status': 'in_progress', 'expected_sequences': len(sequences),
        'expected_sequence_shas': [sequence_sha(s) for s in sequences],
        'expected_raw_residues': sum(map(len, sequences)),
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'semantics': SEMANTICS, 'source_semantics': source_semantics, 'model_pins': model_pins,
        'extractor_sha256': file_sha(__file__), 'torch_version': torch.__version__,
        'device': str(device), 'parameter_dtype': str(next(model.parameters()).dtype),
        'storage_dtype': storage_dtype, 'label_free': True,
        'attention_implementation': getattr(model.config, '_attn_implementation', None) if hasattr(model, 'config') else None,
        'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'provenance': provenance or {}, 'sequences': {}}
    def checkpoint():
        manifest['elapsed_seconds'] = time.monotonic() - started
        manifest['updated_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['completed_sequences'] = len(manifest['sequences'])
        manifest['raw_residues_covered'] = sum(e['raw_residues_covered'] for e in manifest['sequences'].values())
        temporary = root / 'manifest.partial.json.tmp'
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(root / 'manifest.partial.json')
        print(json.dumps({k: manifest[k] for k in ('phase', 'cache_status', 'updated_utc', 'elapsed_seconds',
            'completed_sequences', 'expected_sequences', 'raw_residues_covered', 'expected_raw_residues')}), flush=True)
    checkpoint()
    try:
        for sequence in sequences:
            tensors, receipt = extract_sequence(model, tokenizer, sequence, device, max_residues, storage_dtype)
            relative = f'sequences/{receipt["sequence_sha256"]}.pt'
            path = root / relative
            torch.save(tensors, path)
            receipt.update(path=relative, sha256=file_sha(path), bytes=path.stat().st_size)
            manifest['sequences'][receipt['sequence_sha256']] = receipt
            if len(manifest['sequences']) % progress_every == 0:
                checkpoint()
        if set(manifest['sequences']) != set(manifest['expected_sequence_shas']):
            raise ValueError('expected sequence coverage incomplete')
        manifest['cache_status'] = 'complete'
        checkpoint()
        with (root / 'manifest.json').open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    except Exception as exc:
        manifest['cache_status'] = 'failed'
        manifest['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        checkpoint()
        raise
    return manifest


def select_sequences(view_root, phase='full', max_sequences=None, scenario=SCENARIOS[0]):
    """Canary is label-free but strictly TRAIN-only; shortest/longest fixed by length/SHA."""
    if phase not in ('full', 'train-canary') or (phase == 'full' and max_sequences is not None):
        raise ValueError('max-sequences is only legal for explicit train-canary')
    sequences = set()
    for current in (SCENARIOS if phase == 'full' else (scenario,)):
        for split in (('train', 'validation', 'test') if phase == 'full' else ('train',)):
            for row in read_input_view(view_root, current, split):
                sequences.update((row['peptide_sequence'], row['target_sequence']))
    if phase == 'train-canary':
        count = 4 if max_sequences is None else max_sequences
        if count not in (2, 3, 4) or len(sequences) < count:
            raise ValueError('TRAIN canary requires 2..4 available unique sequences')
        ordered = sorted(sequences, key=lambda s: (len(s), sequence_sha(s)))
        # Endpoints always included; no score/label-based choice.
        indices = [i * (len(ordered) - 1) // (count - 1) for i in range(count)]
        sequences = {ordered[i] for i in indices}
    return sorted(sequences, key=sequence_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inspect_parser = sub.add_parser('inspect')
    inspect_parser.add_argument('--output', required=True)
    extract = sub.add_parser('extract')
    extract.add_argument('--view-root', required=True)
    extract.add_argument('--model-root', required=True)
    extract.add_argument('--model-pins', help='Optional JSON {relative-model-file: sha256}; must equal pinned six-file closure')
    extract.add_argument('--output-root', required=True)
    extract.add_argument('--device', default='cpu')
    extract.add_argument('--compute-dtype', choices=('float32', 'bfloat16', 'float16'), default='float32')
    extract.add_argument('--storage-dtype', choices=('float32', 'float16'), default='float32')
    extract.add_argument('--max-residues', type=int, default=1022)
    extract.add_argument('--phase', choices=('full', 'train-canary'), default='full')
    extract.add_argument('--max-sequences', type=int, help='Only train-canary, fixed 2..4 length-stratified TRAIN sequences')
    extract.add_argument('--scenario', choices=SCENARIOS, default=SCENARIOS[0], help='TRAIN source for train-canary only')
    args = parser.parse_args()
    semantics = inspect_esm_semantics()
    if args.command == 'inspect':
        path = Path(args.output)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(semantics, stream, indent=2)
        if not semantics['verified']:
            raise SystemExit('unqualified installed ESM implementation')
        return
    if not semantics['verified']:
        raise ValueError('unqualified installed ESM implementation')
    if semantics['source_sha256'] != REMOTE_TF4481_SOURCE_SHA or semantics['transformers_version'] != '4.48.1':
        raise ValueError('production extraction is frozen to statically qualified node1 TF4.48.1 source')
    sequences = select_sequences(args.view_root, args.phase, args.max_sequences, args.scenario)
    import torch
    from transformers import AutoTokenizer, EsmModel
    pins = strict_loads(Path(args.model_pins).read_text(encoding='utf-8')) if args.model_pins else MODEL_PINS
    if pins != MODEL_PINS:
        raise ValueError('model pins differ from pinned six-file ESM closure')
    for relative, sha in pins.items():
        verify_sha(registered_path(args.model_root, relative), sha)
    # An explicit revision identifies the frozen asset; offline=True forbids remote fallback.
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, local_files_only=True)
    model = EsmModel.from_pretrained(args.model_root, local_files_only=True,
        torch_dtype=getattr(torch, args.compute_dtype), add_pooling_layer=False).to(args.device)
    if model.config.num_hidden_layers != 30 or model.config.hidden_size != 640:
        raise ValueError('expected pinned 30-block 640-wide ESM2')
    model.requires_grad_(False).eval()
    extract_cache(model, tokenizer, sequences, args.output_root, semantics, pins, args.device,
        args.storage_dtype, args.max_residues,
        {'view_manifest_sha256': file_sha(Path(args.view_root) / 'manifest.json'),
         'model_pins_sha256': file_sha(args.model_pins) if args.model_pins else hashlib.sha256(canonical(MODEL_PINS).encode()).hexdigest(),
         'canary_train_scenario': args.scenario if args.phase == 'train-canary' else None}, phase=args.phase)


if __name__ == '__main__':
    main()

"""Frozen ESM states and nested TRAIN-only molecular-attribute specialists.

Attributes are cross-fitted for TRAIN and fitted without validation/test labels.
The two targets are always ordered retention time, then CCS, in original units.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

# This stage owns two CPU threads; GPU visibility belongs to its launcher.
for _thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '2'

import numpy as np

SPLITS = ('train', 'validation', 'test')
ENDPOINTS = ('rt', 'ccs')
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
HYDROPATHY = dict(zip(AMINO_ACIDS, (1.8, 2.5, -3.5, -3.5, 2.8, -.4, -3.2, 4.5,
                                  -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -.8,
                                  -.7, 4.2, -.9, -1.3)))
PHYSICOCHEMICAL_DIM = 25
ESM_REVISION = 'a695f6045e2e32885fa60af20c13cb35398ce30c'
ESM_PINS = {
    'README.md': '462a2f24724e19c6be0efab926315c294a863c9a9770e2c8b3d859b2d81a07de',
    'config.json': 'e512f68ec444d99477703a9806639ca83da3dbc19f6c5fe428d6e5b7460972dc',
    'model.safetensors': 'c3f1da8aea53bddd32c246c86168c23b9fd72341fb9db9a94436f855f5053566',
    'special_tokens_map.json': '3aedcd4211c0d43aec4e607ff60a63255f3174ead795e997350f09a5f8cd9ee1',
    'tokenizer_config.json': '7e9161ecdb548ec45a41cbc6b24aa4476fdd418461f491c4207baa99419a29ad',
    'vocab.txt': '0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03',
}
SEMANTICS = {
    'p15': 'encoder.layer[14] output before final encoder normalization',
    'p30': 'EsmModel.last_hidden_state after final encoder normalization',
    'pooling': 'all sequence residues, excluding BOS/EOS/padding, FP32 ordered adaptive_average8',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def load_rows(scope):
    root = Path(scope) / 'data'
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        if sha(root / name) != expected:
            raise ValueError('Frozen data changed: ' + name)
    rows = []
    for split in SPLITS:
        values = [json.loads(line) for line in (root / (split + '.jsonl')).read_text(encoding='utf-8').splitlines() if line]
        if len(values) != manifest['counts'][split] or any(r['split'] != split for r in values):
            raise ValueError('Frozen split/count mismatch')
        rows.extend(values)
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)) or any(len(v) != 64 for v in ids):
        raise ValueError('Unique SHA256 identities required')
    return rows, sha(root / 'manifest.json')


def validate_features(path, rows):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'ids', 'p15', 'p30'}:
            raise ValueError('Feature arrays differ from contract')
        arrays = {k: data[k].copy() for k in data.files}
    if arrays['ids'].dtype != np.dtype('U64') or arrays['ids'].tolist() != [r['id'] for r in rows]:
        raise ValueError('Feature identities/order differ')
    for key in ('p15', 'p30'):
        value = arrays[key]
        if value.dtype != np.float32 or value.shape != (len(rows), 8, 640) or not np.isfinite(value).all():
            raise ValueError('Finite float32[N,8,640] states required')
    return arrays


def qualified_semantics():
    import importlib.metadata
    from transformers.models.esm import modeling_esm
    encoder = inspect.getsource(modeling_esm.EsmEncoder.forward)
    model = inspect.getsource(modeling_esm.EsmModel.forward)
    a = encoder.find('hidden_states = layer_outputs[0]')
    b = encoder.find('hidden_states = self.emb_layer_norm_after(hidden_states)')
    if (importlib.metadata.version('transformers') != '4.51.3' or a < 0 or b <= a
            or 'last_hidden_state=hidden_states' not in encoder
            or 'sequence_output = encoder_outputs[0]' not in model):
        raise ValueError('Requires qualified Transformers4.51.3 ESM semantics')
    return {'transformers_version': '4.51.3', 'source_sha256': sha(inspect.getfile(modeling_esm)), 'verified': True}


def extract_batch(model, tokenizer, sequences, device):
    """Same layer semantics as v5; short peptides need no sequence chunking."""
    import torch
    import torch.nn.functional as F
    if any(not 8 <= len(s) <= 30 or any(aa not in AMINO_ACIDS for aa in s) for s in sequences):
        raise ValueError('Canonical complete peptide sequences of length8-30 required')
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Frozen ESM required')
    encoded = tokenizer(sequences, padding=True, return_tensors='pt', add_special_tokens=True, truncation=False)
    for i, sequence in enumerate(sequences):
        size = int(encoded['attention_mask'][i].sum())
        expected = tokenizer.convert_tokens_to_ids(list(sequence))
        if size != len(sequence) + 2 or encoded['input_ids'][i, 1:size-1].tolist() != expected or tokenizer.unk_token_id in expected:
            raise ValueError('Tokenizer altered residues')
    captured = []
    def hook(_module, _args, output):
        captured.append(output[0] if isinstance(output, (tuple, list)) else output)
    handle = model.encoder.layer[14].register_forward_hook(hook)
    try:
        model.eval()
        with torch.inference_mode():
            output = model(**{k: v.to(device) for k, v in encoded.items()}, return_dict=True, output_hidden_states=False)
        if len(captured) != 1:
            raise ValueError('ESM block15 hook count mismatch')
        result = {}
        for key, states in (('p15', captured[0]), ('p30', output.last_hidden_state)):
            if states.shape[:2] != encoded['input_ids'].shape or not torch.isfinite(states).all():
                raise ValueError('Invalid ESM residue states')
            result[key] = np.stack([
                F.adaptive_avg_pool1d(states[i, 1:len(sequence)+1].detach().float().T[None], 8)[0].T.cpu().numpy()
                for i, sequence in enumerate(sequences)
            ]).astype(np.float32, copy=False)
        return result
    finally:
        handle.remove()


def extract(scope, model_path):
    rows, data_sha = load_rows(scope)
    root = Path(scope) / 'cache'
    root.mkdir(parents=True, exist_ok=True)
    payload, receipt_path = root / 'features.npz', root / 'features_receipt.json'
    model_path = Path(model_path)
    for name, expected in ESM_PINS.items():
        if sha(model_path / name) != expected:
            raise ValueError('Pinned ESM model changed: ' + name)
    contract = {'schema': 'paired-peptide-esm.v1', 'data_manifest_sha256': data_sha,
                'extractor_source_sha256': sha(__file__), 'model_revision': ESM_REVISION,
                'model_pins': ESM_PINS, 'semantics': SEMANTICS, 'expected_rows': len(rows)}
    if payload.exists() or receipt_path.exists():
        if not (payload.exists() and receipt_path.exists()):
            raise ValueError('Incomplete cache: preserve it for inspection; no partial cache reuse')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('contract') != contract or receipt.get('status') != 'complete' or receipt.get('sha256') != sha(payload):
            raise ValueError('Feature cache receipt mismatch')
        validate_features(payload, rows)
        return receipt
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from transformers import AutoTokenizer, EsmModel
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Launcher must expose exactly one GPU; extractor never chooses physical GPU')
    semantics = qualified_semantics()
    model = EsmModel.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32,
                                    add_pooling_layer=False, attn_implementation='eager').to('cuda:0')
    if model.config.hidden_size != 640 or model.config.num_hidden_layers != 30:
        raise ValueError('ESM150M30-layer640-width required')
    model.requires_grad_(False).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    arrays = {k: np.empty((len(rows), 8, 640), dtype=np.float32) for k in ('p15', 'p30')}
    started = time.monotonic()
    for start in range(0, len(rows), 32):
        values = extract_batch(model, tokenizer, [r['sequence'] for r in rows[start:start+32]], 'cuda:0')
        for key in arrays:
            arrays[key][start:start+32] = values[key]
        print(json.dumps({'stage': 'extract', 'completed': min(start+32, len(rows)), 'total': len(rows)}), flush=True)
    atomic_npz(payload, ids=np.asarray([r['id'] for r in rows], dtype='U64'), **arrays)
    validate_features(payload, rows)
    receipt = {'contract': contract, 'status': 'complete', 'sha256': sha(payload),
               'source_semantics': semantics, 'elapsed_seconds': time.monotonic()-started,
               'batch_size': 32, 'parameter_dtype': 'float32', 'cpu_threads': 2,
               'torch_version': torch.__version__, 'all_residues_covered': True,
               'labels_used_for_extraction': False}
    write_json(receipt_path, receipt)
    return receipt


def physicochemical_features(rows, experiments=None):
    """Explicit allowlist: sequence composition, length/GRAVY and acquisition cohort."""
    experiments = tuple(experiments or sorted({str(r['experiment']) for r in rows}))
    if len(experiments) != 3:
        raise ValueError('Frozen three-cohort condition vocabulary required')
    result = []
    for row in rows:
        sequence = row['sequence']
        if not sequence or any(aa not in AMINO_ACIDS for aa in sequence):
            raise ValueError('Canonical sequence required')
        if row['experiment'] not in experiments:
            raise ValueError('Unknown acquisition condition')
        result.append([sequence.count(aa)/len(sequence) for aa in AMINO_ACIDS]
                      + [len(sequence), sum(HYDROPATHY[aa] for aa in sequence)/len(sequence)]
                      + [float(row['experiment'] == value) for value in experiments])
    return np.asarray(result, dtype=np.float64)


def build_specialist_covariates(rows, p15, p30):
    if any(np.shape(a) != (len(rows), 8, 640) or not np.isfinite(a).all() for a in (p15, p30)):
        raise ValueError('Complete finite ESM features required')
    # First25 features define the physicochemical-only candidate family.
    return np.column_stack((physicochemical_features(rows), p15.mean(axis=1), p30.mean(axis=1))).astype(np.float64)


def grouped_fold_ids(groups):
    return np.asarray([int(hashlib.sha256(('oof-v1\0'+str(g)).encode()).hexdigest(), 16) % 5 for g in groups], dtype=np.int64)


def inner_holdout(groups):
    return np.asarray([int(hashlib.sha256(('inner-v1\0'+str(g)).encode()).hexdigest(), 16) % 5 == 0 for g in groups])


def _rbf(a, b, gamma):
    distance = np.maximum(np.sum(a*a, axis=1)[:, None] + np.sum(b*b, axis=1)[None] - 2*a@b.T, 0.)
    return np.exp(-gamma*distance)


def _prepare(x, y):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    y_mean, y_scale = y.mean(axis=0), y.std(axis=0)
    y_scale = np.where(y_scale > 1e-12, y_scale, 1.)
    return (x-mean)/scale, (y-y_mean)/y_scale, mean, scale, y_mean, y_scale


def _fit(x, y, family, alpha, physics_dim):
    from scipy.linalg import solve
    columns = physics_dim if family == 'physics_ridge' else x.shape[1]
    z, target, mean, scale, y_mean, y_scale = _prepare(x[:, :columns], y)
    gamma = 1/z.shape[1] if family == 'esm_rbf' else None
    dual = family == 'esm_rbf' or z.shape[0] < z.shape[1]
    matrix = _rbf(z, z, gamma) if family == 'esm_rbf' else z@z.T if dual else z.T@z
    matrix.flat[::matrix.shape[0]+1] += alpha
    rhs = target if dual else z.T@target
    coefficient = solve(matrix, rhs, assume_a='pos', check_finite=False)
    return {'family': family, 'alpha': float(alpha), 'gamma': gamma, 'columns': columns,
            'x_mean': mean, 'x_scale': scale, 'y_mean': y_mean, 'y_scale': y_scale,
            'coefficient': coefficient, 'basis': z if dual else np.empty((0, columns)), 'dual': dual}


def _predict(model, x):
    z = (x[:, :model['columns']]-model['x_mean'])/model['x_scale']
    if model['family'] == 'esm_rbf':
        z = _rbf(z, model['basis'], model['gamma'])
    elif model['dual']:
        z = z@model['basis'].T
    return (z@model['coefficient'])*model['y_scale'] + model['y_mean']


def _select(x, y, groups, physics_dim):
    """One deterministic grouped20% inner TRAIN holdout, no external labels."""
    from scipy.linalg import eigh
    holdout = inner_holdout(groups)
    if not holdout.any() or holdout.all():
        raise ValueError('Fixed inner grouped split has no support')
    candidates = []
    for family in ('physics_ridge', 'esm_ridge', 'esm_rbf'):
        columns = physics_dim if family == 'physics_ridge' else x.shape[1]
        z, target, mean, scale, y_mean, y_scale = _prepare(x[~holdout, :columns], y[~holdout])
        query = (x[holdout, :columns]-mean)/scale
        gamma = 1/columns if family == 'esm_rbf' else None
        dual = family == 'esm_rbf' or z.shape[0] < columns
        matrix = _rbf(z, z, gamma) if family == 'esm_rbf' else z@z.T if dual else z.T@z
        projection = _rbf(query, z, gamma) if family == 'esm_rbf' else query@z.T if dual else query
        rhs = target if dual else z.T@target
        eigenvalues, eigenvectors = eigh(matrix, check_finite=False)
        left, right = projection@eigenvectors, eigenvectors.T@rhs
        normalized_gold = (y[holdout]-y_mean)/y_scale
        for alpha in (.1, 1., 10.):
            prediction = (left*(1/(np.maximum(eigenvalues, 0.)+alpha)))@right
            endpoint_mse = np.mean((prediction-normalized_gold)**2, axis=0)
            candidates.append({'family': family, 'alpha': alpha, 'gamma': gamma,
                               'inner_standardized_mse': float(endpoint_mse.mean()),
                               'endpoint_standardized_mse': endpoint_mse.tolist()})
    chosen = min(candidates, key=lambda c:c['inner_standardized_mse'])
    model = _fit(x, y, chosen['family'], chosen['alpha'], physics_dim)
    return model, {'selected': chosen, 'candidates': candidates,
                   'inner_fit_rows': int((~holdout).sum()), 'inner_holdout_rows': int(holdout.sum()),
                   'inner_fit_groups': len(set(groups[~holdout])), 'inner_holdout_groups': len(set(groups[holdout])),
                   'inner_group_overlap': len(set(groups[~holdout]) & set(groups[holdout])),
                   'coefficient_count': int(model['coefficient'].size),
                   'basis_value_count': int(model['basis'].size), 'feature_columns': model['columns']}


def fit_predict_specialists(train_x, train_y, train_groups, predict_x, physicochemical_dim=PHYSICOCHEMICAL_DIM):
    """Only TRAIN targets enter this API. Other splits supply covariates only."""
    train_x, train_y = np.asarray(train_x, dtype=np.float64), np.asarray(train_y, dtype=np.float64)
    train_groups = np.asarray(train_groups, dtype='U64')
    predict_x = np.asarray(predict_x, dtype=np.float64)
    if (train_y.shape != (len(train_x), 2) or len(train_groups) != len(train_x)
            or train_x.ndim != 2 or predict_x.ndim != 2 or predict_x.shape[1] != train_x.shape[1]
            or not 1 <= physicochemical_dim <= train_x.shape[1]
            or any(not np.isfinite(a).all() for a in (train_x, train_y, predict_x))):
        raise ValueError('Finite TRAIN two-endpoint arrays and compatible prediction covariates required')
    folds = grouped_fold_ids(train_groups)
    oof = np.full(train_y.shape, np.nan)
    fold_receipts, models = [], {}
    started = time.monotonic()
    for fold in range(5):
        held = folds == fold
        if not held.any() or held.all():
            raise ValueError('Fixed five-fold grouped TRAIN split has insufficient support')
        model, receipt = _select(train_x[~held], train_y[~held], train_groups[~held], physicochemical_dim)
        oof[held] = _predict(model, train_x[held])
        overlap = set(train_groups[held]) & set(train_groups[~held])
        if overlap:
            raise ValueError('OOF group leakage')
        receipt.update(fold=fold, fit_rows=int((~held).sum()), heldout_rows=int(held.sum()),
                       fit_groups=len(set(train_groups[~held])), excluded_groups=len(set(train_groups[held])),
                       outer_group_overlap=len(overlap))
        fold_receipts.append(receipt)
        models['fold'+str(fold)] = model
    final_model, final_receipt = _select(train_x, train_y, train_groups, physicochemical_dim)
    models['final'] = final_model
    prediction = _predict(final_model, predict_x)
    if not np.isfinite(oof).all() or not np.isfinite(prediction).all():
        raise ValueError('Incomplete specialist predictions')
    receipt = {'schema': 'nested-grouped-train-only-specialists.v1', 'endpoint_order': list(ENDPOINTS),
               'train_rows': len(train_x), 'prediction_rows': len(predict_x), 'outer_folds': fold_receipts,
               'final': final_receipt, 'fold_rule': 'SHA256(oof-v1 NUL group) modulo5',
               'inner_rule': 'SHA256(inner-v1 NUL group) modulo5 equals0; grouped20percent',
               'selection_criterion': 'mean of two endpoint population-standardized inner TRAIN MSE',
               'standardizers_fit_on_inner_fit_only': True, 'external_labels_used': False,
               'final_train_target_mean': train_y.mean(axis=0).tolist(),
               'final_train_target_population_std': train_y.std(axis=0).tolist(),
               'elapsed_seconds': time.monotonic()-started}
    return oof, prediction, folds, receipt, models


def _model_arrays(models):
    result = {}
    metadata = {}
    for prefix, model in models.items():
        metadata[prefix] = {key: value for key, value in model.items() if not isinstance(value, np.ndarray)}
        for key, value in model.items():
            if isinstance(value, np.ndarray):
                result[prefix+'_'+key] = value
    result['metadata_json'] = np.asarray(json.dumps(metadata, sort_keys=True))
    return result


def fit_panel_specialists(rows, p15, p30):
    """Panel adapter exposes validation/test covariates but reads TRAIN labels only."""
    x = build_specialist_covariates(rows, p15, p30)
    training = np.asarray([r['split'] == 'train' for r in rows])
    y = np.asarray([[r[key] for key in ENDPOINTS] for r in rows if r['split'] == 'train'], dtype=np.float64)
    groups = np.asarray([r['group_id'] for r in rows if r['split'] == 'train'], dtype='U64')
    oof, prediction, folds, receipt, models = fit_predict_specialists(x[training], y, groups, x[~training])
    values = np.empty((len(rows), 2), dtype=np.float64)
    values[training], values[~training] = oof, prediction
    all_folds = np.full(len(rows), -1, dtype=np.int64)
    all_folds[training] = folds
    receipt['experiment_vocabulary'] = sorted({str(r['experiment']) for r in rows})
    return values, all_folds, receipt, models


def specialists(scope):
    rows, data_sha = load_rows(scope)
    root = Path(scope) / 'cache'
    features_path, receipt_path = root / 'features.npz', root / 'features_receipt.json'
    features = validate_features(features_path, rows)
    source_receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if (source_receipt.get('status') != 'complete' or source_receipt.get('sha256') != sha(features_path)
            or source_receipt['contract']['data_manifest_sha256'] != data_sha):
        raise ValueError('Feature provenance differs from frozen data')
    output, result_path = root / 'attributes.npz', root / 'attributes_receipt.json'
    contract = {'data_manifest_sha256': data_sha, 'feature_sha256': sha(features_path),
                'specialist_source_sha256': sha(__file__), 'endpoint_order': list(ENDPOINTS)}
    if output.exists() or result_path.exists():
        if not (output.exists() and result_path.exists()):
            raise ValueError('Incomplete attribute artifacts; no silent partial reuse')
        receipt = json.loads(result_path.read_text(encoding='utf-8'))
        if receipt.get('contract') != contract or receipt.get('status') != 'complete' or receipt.get('sha256') != sha(output):
            raise ValueError('Attribute receipt mismatch')
        with np.load(output, allow_pickle=False) as saved:
            if (saved['ids'].tolist() != [r['id'] for r in rows] or saved['predictions'].shape != (len(rows), 2)
                    or not np.isfinite(saved['predictions']).all()):
                raise ValueError('Invalid saved attributes')
        if sha(Path(scope)/'private/specialists.npz') != receipt['model_sha256']:
            raise ValueError('Specialist model state changed')
        return receipt
    # Construct target arrays only from TRAIN rows; validation/test labels never
    # enter model selection, standardization, fitting, diagnostic scores or API.
    values, all_folds, receipt, models = fit_panel_specialists(rows, features['p15'], features['p30'])
    model_path = Path(scope)/'private/specialists.npz'
    atomic_npz(model_path, **_model_arrays(models))
    atomic_npz(output, ids=features['ids'], predictions=values,
               split=np.asarray([r['split'] for r in rows], dtype='U10'), oof_fold=all_folds)
    receipt.update(contract=contract, status='complete', sha256=sha(output), model_sha256=sha(model_path),
                   feature_spec={'physics_columns': ['20_amino_acid_fractions', 'length', 'gravy', 'three_experiment_indicators'],
                                 'esm_columns': 'mean of eight ordered p15 and p30 slots;1280 values',
                                 'excluded': ['measured Mass', 'Intensity', 'Score', 'measured RT', 'measured CCS', 'identity', 'group_id']})
    write_json(result_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', required=True)
    parser.add_argument('--stage', choices=('extract', 'specialists'), required=True)
    parser.add_argument('--model-path')
    args = parser.parse_args()
    if args.stage == 'extract' and not args.model_path:
        parser.error('--model-path is required for extraction')
    receipt = extract(args.scope, args.model_path) if args.stage == 'extract' else specialists(args.scope)
    print(json.dumps({'status': receipt['status'], 'stage': args.stage, 'sha256': receipt['sha256']}), flush=True)


if __name__ == '__main__':
    main()

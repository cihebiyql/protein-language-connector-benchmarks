"""Recompute locked panel results; never mix deprecated data revisions."""
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_v02.contracts import ROOT, jsonl, sha256
from bench_v02.metrics import known_positive_retrieval, regression
from bench_v02.panel_execution import validate_panel_data

OLD = ROOT / 'results/bench_v02/panel_baselines_20261015'
CURRENT = ROOT / 'results/bench_v02/panel_baselines_20261015_v2'
SEEDS = {17, 42, 2026}


def paired_group_gain(reference_error, candidate_error, groups):
    """Positive is improvement; bootstrap locked independent groups, not rows."""
    delta = np.asarray(reference_error) - np.asarray(candidate_error)
    grouped = defaultdict(list)
    for index, group in enumerate(groups):
        grouped[group].append(index)
    positions = list(grouped.values())
    sums = np.array([delta[p].sum() for p in positions])
    sizes = np.array([len(p) for p in positions])
    rng = np.random.default_rng(2026)
    draws = rng.integers(len(positions), size=(2000, len(positions)))
    effects = sums[draws].sum(1) / sizes[draws].sum(1)
    return {'gain_in_pKD_MAE': float(delta.mean()), 'ci95': np.quantile(effects, [.025, .975]).tolist(),
            'n_independent_groups': len(positions), 'bootstrap_draws': 2000, 'bootstrap_seed': 2026,
            'scope': 'fixed three-seed mean absolute errors; locked group sampling uncertainty; seed variability separate'}


def _same(actual, expected, label):
    if not math.isclose(float(actual), float(expected), abs_tol=1e-8, rel_tol=1e-8):
        raise ValueError(f'{label} differs from independently recomputed metrics')


def verify_run(path, rows, candidates=None):
    path=Path(path).resolve()
    result = json.loads(path.read_text(encoding='utf-8'))
    panel = result['panel_id']
    allowed={'A1':{'BASE01','MEANPOOL'},'B1':{'BASE02','MEANALIGN'}}
    if (result.get('status')!='complete' or result.get('role')!='controlled_predictor'
            or result.get('channel')!='predictor' or result['architecture'] not in allowed[panel]
            or result.get('formal_multitask_connector_result') is not False):
        raise ValueError('run role, architecture or completed scope is invalid')
    for name, expected in result['artifact_sha256'].items():
        if sha256(path.parent/name) != expected:
            raise ValueError(f'changed run artifact {path.parent/name}')
    if sha256(path.parent/'best.pt') != result['checkpoint_sha256']:
        raise ValueError('checkpoint fingerprint differs')
    if result['seed'] not in SEEDS or result['actual_epochs'] != 30:
        raise ValueError('unregistered seed or epoch budget')
    history = json.loads((path.parent/'history.json').read_text(encoding='utf-8'))
    if [row['epoch'] for row in history] != list(range(1, 31)):
        raise ValueError('training history is incomplete')
    metric_key = 'mae' if panel == 'A1' else 'class_macro_mrr'
    key = (lambda h: h['validation'][metric_key]) if panel == 'A1' else (lambda h: -h['validation'][metric_key])
    best = min(history, key=key)
    if best['epoch'] != result['best_epoch']:
        raise ValueError('checkpoint was not the earliest best validation selection')
    n_train = result['split_counts_after_unit_aggregation']['train']
    if result['optimizer_steps'] != 30 * math.ceil(n_train/64):
        raise ValueError('optimizer budget differs')
    gold = {row['sample_id']: row for row in rows if row['split'] == 'test'}
    predictions = list(jsonl(path.parent/'test_predictions.jsonl'))
    if len({p['sample_id'] for p in predictions}) != len(predictions):
        raise ValueError('duplicate prediction sample')
    raw = np.load(path.parent/'test_scores.npz', allow_pickle=False)
    ids = raw['sample_ids'].tolist()
    scores = raw['scores']
    if ids != [p['sample_id'] for p in predictions] or not np.isfinite(scores).all():
        raise ValueError('prediction identities or numerical validity differ')
    if panel == 'A1':
        if set(ids) != set(gold):
            raise ValueError('incomplete A1 locked test coverage')
        targets = [gold[i]['groups']['target_sequence_sha256'] for i in ids]
        y = np.array([gold[i]['pkd'] for i in ids])
        np.testing.assert_array_equal(raw['labels'], y)
        np.testing.assert_allclose(scores, [p['prediction']['pKD'] for p in predictions], atol=0, rtol=0)
        metrics = regression(y, scores, targets)
        _same(metrics['mae'], result['test_metrics']['mae'], 'A1 MAE')
        _same(metrics['group_macro_mae'], result['test_metrics']['group_macro_mae'], 'A1 exact-target macro MAE')
        evidence = {'sample_ids': ids, 'errors': np.abs(y-scores).tolist(),
                    'groups': [gold[i]['group_id'] for i in ids]}
    else:
        candidate_ids = raw['candidate_ids'].tolist()
        candidate_classes = raw['candidate_class_ids'].tolist()
        if candidate_ids != [c['candidate_id'] for c in candidates]:
            raise ValueError('B1 ranking universe/order differs from locked descriptions')
        mapping = {c['candidate_id']: c['class_id'] for c in candidates}
        if candidate_classes != [mapping[i] for i in candidate_ids]:
            raise ValueError('candidate/class mapping differs')
        positives_by_sequence = defaultdict(set)
        for row in gold.values():
            positives_by_sequence[row['segment_sequence']].update(row['known_positive_class_ids'])
        members=[m for p in predictions for m in p['member_sample_ids']]
        if len(members)!=len(gold) or set(members) != set(gold):
            raise ValueError('B1 aggregated queries do not cover all locked annotated rows')
        rankings, positives = [], []
        for index, prediction in enumerate(predictions):
            ranking = [candidate_ids[j] for j in np.argsort(-scores[index], kind='stable')]
            if ranking != prediction['prediction']['ranked_candidate_ids']:
                raise ValueError('B1 ranking differs from raw saved logits')
            rankings.append([mapping[c] for c in ranking])
            positives.append(positives_by_sequence[gold[ids[index]]['segment_sequence']])
        metrics = known_positive_retrieval(rankings, positives)
        for field in ('mrr','recall_at_1','recall_at_3','class_macro_mrr','class_macro_recall_at_1','class_macro_recall_at_3'):
            _same(metrics[field], result['test_metrics'][field], f'B1 {field}')
        evidence = {'sample_ids': ids}
    return {'run_id': path.parent.name, 'panel_id': panel, 'architecture': result['architecture'],
            'seed': result['seed'], 'scenario': result['scenario'], 'result_path': str(path.relative_to(ROOT)),
            'result_sha256': sha256(path), 'checkpoint_sha256': result['checkpoint_sha256'],
            'best_epoch': result['best_epoch'], 'optimizer_steps': result['optimizer_steps'],
            'trainable_parameters': result['trainable_parameters'], 'training_seconds': result['training_seconds'],
            'gpu_peak_allocated_bytes': result['gpu_peak_allocated_bytes'], 'metrics': metrics,
            'evidence': evidence, 'data_gate': result['data_gate']}


def main():
    data = {}
    locks = {}
    for scenario in ('new_peptide_known_target','new_peptide_new_target'):
        directory = ROOT/'data/processed/bench_v02/a1_locked_v2'/scenario
        data[scenario], locks[scenario] = validate_panel_data(directory/'samples.jsonl', directory/'benchmark_lock.json', 'A1')
    directory = ROOT/'data/processed/bench_v02/b1_homology_locked'
    data['unseen_classes'], locks['unseen_classes'] = validate_panel_data(directory/'b1_samples.jsonl', directory/'benchmark_lock.json', 'B1')
    mapping = {row['candidate_id']: row for row in jsonl(directory/'b1_classes_main.jsonl')}
    candidates = [{**row, 'class_id': mapping[row['candidate_id']]['class_id']} for row in jsonl(directory/'b1_candidates_main.jsonl')]
    paths = sorted(OLD.glob('B1_*/result.json')) + sorted(CURRENT.glob('A1_*/result.json'))
    runs = []
    for path in paths:
        result = json.loads(path.read_text(encoding='utf-8'))
        scenario = result['scenario']
        if result['data_gate']['benchmark_lock_sha256'] != locks[scenario]['benchmark_lock_sha256']:
            raise ValueError('run uses an obsolete data revision')
        runs.append(verify_run(path, data[scenario], candidates if result['panel_id']=='B1' else None))
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run['scenario'],run['architecture'])].append(run)
    tables = {}
    for (scenario, architecture), values in grouped.items():
        if {run['seed'] for run in values} != SEEDS or len(values) != 3:
            raise ValueError('an architecture is missing a registered seed')
        fields = ('mae','group_macro_mae') if values[0]['panel_id']=='A1' else ('class_macro_mrr','class_macro_recall_at_1','class_macro_recall_at_3')
        tables.setdefault(scenario,{})[architecture] = {
            'seed_metrics': {str(run['seed']): {f:run['metrics'][f] for f in fields} for run in values},
            'mean': {f:float(np.mean([run['metrics'][f] for run in values])) for f in fields},
            'sample_sd_between_seeds': {f:float(np.std([run['metrics'][f] for run in values],ddof=1)) for f in fields}}
    comparisons = {}
    control_sources=[]
    for scenario, short in [('new_peptide_known_target','known'),('new_peptide_new_target','double_new')]:
        control_path=CURRENT/f'A1_{short}_BASE00/results.json'
        prediction_path=control_path.parent/'test_predictions.jsonl'
        base = json.loads(control_path.read_text(encoding='utf-8'))
        if base['execution_receipt']['benchmark_lock_sha256']!=locks[scenario]['benchmark_lock_sha256']:
            raise ValueError('BASE00 uses an obsolete data revision')
        for name,expected in base['artifact_sha256'].items():
            if sha256(control_path.parent/name)!=expected:
                raise ValueError('BASE00 physical artifact changed')
        preds = list(jsonl(prediction_path))
        rows = {row['sample_id']:row for row in data[scenario] if row['split']=='test'}
        if len(preds)!=len(rows) or {p['sample_id'] for p in preds}!=set(rows):
            raise ValueError('BASE00 test coverage is incomplete or duplicated')
        if any(p['pkd']!=rows[p['sample_id']]['pkd'] for p in preds):
            raise ValueError('BASE00 gold labels differ from locked data')
        control_sources.extend([control_path,prediction_path])
        controls = {}
        for model in base['models']:
            controls[model] = regression([rows[p['sample_id']]['pkd'] for p in preds],
                                        [p['predictions'][model] for p in preds],
                                        [rows[p['sample_id']]['groups']['target_sequence_sha256'] for p in preds])
            _same(controls[model]['mae'],base['models'][model]['test']['mae'],'BASE00 MAE')
        tables.setdefault(scenario,{})['BASE00_recomputed_exact_target_groups'] = controls
        if (scenario,'BASE01') in grouped and (scenario,'MEANPOOL') in grouped:
            attention = grouped[(scenario,'BASE01')]
            meanpool = grouped[(scenario,'MEANPOOL')]
            all_ids = attention[0]['evidence']['sample_ids']
            if any(run['evidence']['sample_ids'] != all_ids for run in attention+meanpool):
                raise ValueError('paired comparison sample ordering differs')
            errors = lambda values: np.mean([run['evidence']['errors'] for run in values],axis=0)
            comparisons[scenario] = paired_group_gain(errors(meanpool),errors(attention),attention[0]['evidence']['groups'])
    b1_expected = {'uniform_random_expected_mrr':sum(1/i for i in range(1,len(candidates)+1))/len(candidates),
                   'uniform_random_expected_recall_at_1':1/len(candidates),
                   'uniform_random_expected_recall_at_3':min(3,len(candidates))/len(candidates),
                   'interpretation':'analytical reference for a uniform random complete ranking, not an additional trained run',
                   'candidate_count':len(candidates),'unseen_test_classes':2}
    result = {'benchmark_version':'0.2','scope':'panel_specific_predictors',
              'status':'complete' if len(runs)==18 else 'partial','verified_neural_runs':len(runs),
              'tables':tables,'paired_A1_attention_gain':comparisons,'B1_analytical_reference':b1_expected,
              'A1_macro_group_definition':'exact target reference sequence SHA256 for all methods; BASE00 raw family-macro preserved but recomputed here for comparison',
              'condition_comparable_pair_count':0,'unified_connector_claim':False,
              'runs':[{k:v for k,v in run.items() if k!='evidence'} for run in runs],
              'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in paths+control_sources},
              'summary_script_sha256':sha256(__file__)}
    destination=CURRENT/'comparison.json'
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'verified_neural_runs':len(runs),'output':str(destination)},ensure_ascii=False))


if __name__=='__main__':
    main()

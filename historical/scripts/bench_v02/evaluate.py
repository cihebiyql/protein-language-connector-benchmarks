"""Score v0.2 outputs with explicit channels; fixtures never become real results."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bench_v02.contracts import ROOT,ContractError,finite_number,jsonl,json_generation,numeric_generation,record_reasons,sha256,validate_run_manifest
from bench_v02.metrics import regression,grouped_regression,known_positive_retrieval

def evaluate(gold,predictions,manifest,panel,*,diagnostic=False,protocol_path=None,gold_path=None,
             candidate_to_class=None,candidate_map_path=None,benchmark_lock_path=None):
    if not gold:
        raise ContractError('empty gold cannot produce a completed benchmark score')
    channel=validate_run_manifest(manifest,panel,diagnostic=diagnostic,protocol_path=protocol_path,
                                 gold_path=gold_path,benchmark_lock_path=benchmark_lock_path)
    if not diagnostic:
        if list(jsonl(gold_path))!=gold:
            raise ContractError('in-memory gold differs from the fingerprinted physical data')
        for row in gold:
            reasons=record_reasons(row,panel)
            if row.get('split') not in ('test','external'):
                reasons.append('gold split is not locked test/external')
            if reasons:
                raise ContractError(f'ineligible formal gold {row.get("sample_id")}: {reasons}')
        if panel=='B1':
            if candidate_map_path is None or benchmark_lock_path is None:
                raise ContractError('formal semantic catalog requires a physical frozen map and benchmark lock')
            lock=json.loads(Path(benchmark_lock_path).read_text(encoding='utf-8'))
            if lock.get('candidate_map_sha256')!=sha256(candidate_map_path):
                raise ContractError('semantic catalog fingerprint mismatch')
            catalog=[r for r in jsonl(candidate_map_path) if r.get('candidate_id') and r.get('eligible')]
            actual={r['candidate_id']:r['class_id'] for r in catalog}
            if not actual or actual!=candidate_to_class or len(set(actual.values()))!=len(actual):
                raise ContractError('semantic map differs from the frozen catalog')
            if set(lock.get('candidate_ids',[]))!=set(actual):
                raise ContractError('candidate universe is not frozen')
            heldout={r['class_id'] for r in catalog if r.get('split')=='test'}
            if any(not set(r.get('known_positive_class_ids',[r.get('class_id')]))<=heldout for r in gold):
                raise ContractError('main B1 positives must be held-out classes in the frozen catalog')
    ids=[r['sample_id'] for r in gold]
    if len(ids)!=len(set(ids)):
        raise ContractError('gold sample IDs must be unique')
    by_id={}
    for row in predictions:
        if row.get('sample_id') in by_id:
            raise ContractError('duplicate prediction sample IDs')
        if row.get('panel_id')!=panel or row.get('channel')!=channel:
            raise ContractError('mixed panel/output channel cannot enter one result table')
        if row.get('checkpoint_sha256')!=manifest.get('checkpoint_sha256'):
            raise ContractError('per-row checkpoint differs from the declared run')
        by_id[row['sample_id']]=row
    if set(by_id)-set(ids):
        raise ContractError('predictions contain IDs outside locked gold')
    values=[]
    predicted=[]
    retained=[]
    errors=[]
    for row in gold:
        try:
            output=by_id[row['sample_id']]
            if panel in ('A1','A2'):
                key='pKD' if panel=='A1' else 'retention_time'
                if channel=='generation':
                    value=numeric_generation(output.get('raw_text'),key,'min' if panel=='A2' else None)
                else:
                    value=output.get('prediction',{}).get(key)
                    if not finite_number(value):
                        raise ContractError('readout/predictor numerical output must be finite')
                    if panel=='A2' and output.get('prediction',{}).get('unit')!='min':
                        raise ContractError('readout RT must be inverse-transformed to declared minutes')
                truth=row['pkd'] if panel=='A1' else row['rt_value']
                if not finite_number(truth):
                    raise ContractError('gold has no verified numeric target')
                if panel=='A1' and (row.get('relation')!='=' or not row.get('label_exact')):
                    raise ContractError('censored values cannot enter exact MAE')
                if panel=='A2' and row.get('rt_unit')!='min':
                    raise ContractError('unknown RT coordinates cannot be called minutes')
                values.append(float(truth));predicted.append(float(value))
            elif panel=='B1':
                if channel=='generation':
                    answer=json_generation(output.get('raw_text'))
                else:
                    answer=output.get('prediction',{})
                ranking=answer.get('ranked_candidate_ids')
                if not isinstance(ranking,list) or not ranking or not all(isinstance(x,str) for x in ranking):
                    raise ContractError('function answer requires anonymous ranked candidate IDs')
                if not candidate_to_class or any(c not in candidate_to_class for c in ranking):
                    raise ContractError('candidate identity is not in the frozen semantic catalog')
                if len(ranking)!=len(set(ranking)):
                    raise ContractError('ranked candidate IDs must not repeat')
                if not diagnostic and set(ranking)!=set(candidate_to_class):
                    raise ContractError('main Macro-MRR requires the complete frozen candidate ranking')
                positives=row.get('known_positive_class_ids')
                if positives is None:
                    positives=[row['class_id']]
                if not positives or not set(positives)<=set(candidate_to_class.values()):
                    raise ContractError('known positives must belong to the frozen class catalog')
                predicted.append([candidate_to_class[c] for c in ranking])
                values.append(positives)
            else:
                raise ContractError('B2/C panels require adjudicated fact/intervention evidence; use registered metrics after review')
            retained.append(row)
        except (KeyError,ValueError,TypeError) as error:
            errors.append({'sample_id':row['sample_id'],'reason':str(error)})
    result={'benchmark_version':'0.2','panel_id':panel,'channel':channel,'model_id':manifest.get('model_id'),
            'seed':manifest.get('seed'),'checkpoint_sha256':manifest.get('checkpoint_sha256'),
            'role':manifest['role'],
            'comparison_panel':('published_systems' if manifest['role']=='published_system' else
                                 'diagnostic_oracle' if manifest['role']=='diagnostic_oracle' else 'controlled_information'),
            'exposure_disclosure':manifest.get('exposure_disclosure','not_recorded'),
            'score_scope':'diagnostic_not_formal' if diagnostic else 'locked_formal',
            'status':'complete' if not errors else 'incomplete_invalid_outputs',
            'n_gold':len(gold),'n_valid':len(retained),'output_validity':len(retained)/len(gold) if gold else 0,
            'invalid_outputs':errors,'main_table_eligible':not errors and not diagnostic and channel!='oracle_pipeline','metrics':None}
    if retained:
        if panel=='A1':
            groups=[str(r.get('groups',{}).get('target_sequence_sha256') or
                        (hashlib.sha256(r['target_sequence'].encode()).hexdigest() if r.get('target_sequence') else None) or
                        r.get('groups',{}).get('target_uniprot') or r.get('target_id','unknown')) for r in retained]
            result['metrics']=regression(values,predicted,groups)
            comparable=[i for i,r in enumerate(retained) if r.get('experimental_context',{}).get('comparable_condition_group')]
            result['metrics']['within_target_condition']=(grouped_regression(
                [values[i] for i in comparable],[predicted[i] for i in comparable],
                [retained[i]['experimental_context']['comparable_condition_group'] for i in comparable],
                target_ids=[groups[i] for i in comparable]) if comparable else {'n':0,'status':'no_comparable_conditions','pooled':False})
        elif panel=='A2':
            result['metrics']=grouped_regression(values,predicted,[r['condition_id'] for r in retained])
        else:
            result['metrics']=known_positive_retrieval(predicted,values)
    if errors:
        result['metric_scope']='conditional_on_valid_outputs_only_not_primary_score'
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',choices=['A1','A2','B1'],required=True)
    p.add_argument('--gold',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--run-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--diagnostic',action='store_true')
    config=json.loads((ROOT/'configs/bench_v02.json').read_text(encoding='utf-8'))
    p.add_argument('--candidate-map',type=Path,default=ROOT/config['panels']['B1']['candidate_class_map'])
    p.add_argument('--benchmark-lock',type=Path,default=ROOT/'data/processed/bench_v02/benchmark_lock.json')
    args=p.parse_args()
    manifest=json.loads(args.run_manifest.read_text(encoding='utf-8'))
    try:
        mapping={r['candidate_id']:r['class_id'] for r in jsonl(args.candidate_map) if r.get('candidate_id') and r.get('eligible')} if args.panel=='B1' else None
        report=evaluate(list(jsonl(args.gold)),list(jsonl(args.predictions)),manifest,args.panel,
                        diagnostic=args.diagnostic,protocol_path=ROOT/'configs/bench_v02.json',gold_path=args.gold,
                        candidate_to_class=mapping,candidate_map_path=args.candidate_map,benchmark_lock_path=args.benchmark_lock)
    except ContractError as error:
        p.exit(2,f'evaluation refused: {error}\n')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'scope':report['score_scope'],'output':str(args.output)}))
    if report['invalid_outputs']:
        raise SystemExit(2)

if __name__=='__main__':
    main()

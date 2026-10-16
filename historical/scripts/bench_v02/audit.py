"""Build a versioned readiness report and refuse unverified formal training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bench_v02.contracts import ROOT,jsonl,record_reasons,sha256
from bench_v02.panel_execution import validate_panel_data

def load(path):
    p=ROOT/path
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}

def check_rows(path,panel):
    p=ROOT/path
    if not p.exists():
        return {'path':path,'status':'missing','rows':0,'row_contract_valid':0}
    errors=Counter()
    seen=set()
    n=valid=0
    for row in jsonl(p):
        n+=1
        reasons=record_reasons(row,panel)
        if row['sample_id'] in seen:
            reasons.append('duplicate_sample_id')
        seen.add(row['sample_id'])
        errors.update(reasons)
        valid+=not reasons
    return {'path':path,'sha256':sha256(p),'rows':n,'row_contract_valid':valid,
            'status':'row_audited_not_globally_locked','row_issues':dict(errors)}


def panel_execution_status(panel):
    """Recompute each panel's physical gate without granting a unified lock."""
    directory=ROOT/'data/processed/bench_v02'
    affinity_root=ROOT/load('configs/bench_v02.json')['panels']['A1'].get('locked_panels_root','data/processed/bench_v02/a1_locked')
    locks=(sorted(affinity_root.glob('*/benchmark_lock.json')) if panel=='A1' else
           [directory/'b1_homology_locked/benchmark_lock.json'] if panel=='B1' else [])
    scenarios=[]
    for lock_path in locks:
        if not lock_path.exists():
            continue
        samples=lock_path.parent/('samples.jsonl' if panel=='A1' else 'b1_samples.jsonl')
        try:
            _,receipt=validate_panel_data(samples,lock_path,panel)
            scenarios.append({'lock_path':str(lock_path.relative_to(ROOT)),
                              'baseline_training_ready':True,**receipt})
        except (ValueError,KeyError,OSError) as error:
            scenarios.append({'lock_path':str(lock_path.relative_to(ROOT)),
                              'baseline_training_ready':False,'blocker':str(error)})
    return {'scope':'panel_specific_predictor','wait_for_other_panels':False,
            'baseline_training_ready':bool(scenarios) and any(s.get('baseline_training_ready') for s in scenarios),
            'scenarios':scenarios,'unified_multitask_connector_claim':False}


def completed_baseline_artifacts():
    """Only report completed fits with their physical output artifacts intact."""
    directory=ROOT/'results/bench_v02/panel_baselines_20261015'
    revision=ROOT/'results/bench_v02/panel_baselines_20261015_v2'
    completed=[]
    paths=sorted(revision.glob('*/results.json')) if revision.exists() else sorted(directory.glob('*/results.json'))
    for path in paths:
        result=json.loads(path.read_text(encoding='utf-8'))
        hashes=result.get('artifact_sha256',{})
        if result.get('baseline_id')!='BASE-00' or not hashes:
            continue
        if not all((path.parent/name).is_file() and sha256(path.parent/name)==expected for name,expected in hashes.items()):
            raise ValueError(f'completed baseline artifact changed: {path}')
        completed.append({'path':str(path.relative_to(ROOT)),'sha256':sha256(path),
                          'panel_id':result['panel_id'],'baseline_id':'BASE-00',
                          'fitted_models':len(result.get('models',{})),
                          'scope':result['execution_receipt']['status'],'stochastic_training':False})
    neural_paths=([p for p in directory.glob('*/result.json') if p.parent.name.startswith('B1_')]
                  +list(revision.glob('*/result.json')) if revision.exists() else list(directory.glob('*/result.json')))
    for path in sorted(neural_paths):
        result=json.loads(path.read_text(encoding='utf-8'))
        required=['best.pt','history.json','test_scores.npz','test_predictions.jsonl','run_preregistration.json']
        if not all((path.parent/name).is_file() for name in required):
            continue
        if result.get('checkpoint_sha256')!=sha256(path.parent/'best.pt'):
            raise ValueError(f'completed neural checkpoint changed: {path}')
        if not all((path.parent/name).is_file() and sha256(path.parent/name)==expected
                   for name,expected in result.get('artifact_sha256',{}).items()):
            raise ValueError(f'completed neural artifact changed: {path}')
        history=json.loads((path.parent/'history.json').read_text(encoding='utf-8'))
        if len(history)!=30:
            continue
        completed.append({'path':str(path.relative_to(ROOT)),'sha256':sha256(path),
                          'panel_id':result.get('panel_id'),'baseline_id':result.get('architecture'),
                          'seed':result.get('seed'),'fitted_models':1,'stochastic_training':True,
                          'checkpoint_sha256':sha256(path.parent/'best.pt')})
    return completed


def coverage_supplement_status(path):
    source=ROOT/path
    if not source.exists():
        return {'status':'missing','path':path}
    value=json.loads(source.read_text(encoding='utf-8'))
    if 'parent_benchmark_lock_sha256' in value:
        lock=ROOT/'data/processed/bench_v02/b1_homology_locked/benchmark_lock.json'
        if sha256(lock)!=value['parent_benchmark_lock_sha256']:
            raise ValueError('B1 external supplement ancestry changed')
        for relative,expected in value.get('pins',{}).items():
            if sha256(ROOT/relative)!=expected:
                raise ValueError(f'B1 external supplement evidence changed: {relative}')
        return {'status':'verified','path':path,'sha256':sha256(source),
                'external_isolation_verified_for_declared_sources':value['external_isolation_verified_for_declared_sources'],
                'coverage':value['coverage'],'original_training_receipt_retained':True}
    return {'status':'evidence_registered','path':path,'sha256':sha256(source),
            'scenarios':{scenario:{key:item.get(key) for key in
                ('pepbi_rows','unique_peptides','unique_actual_targets','unique_original_itc_dois',
                 'new_external_excluded_rows','split_counts','checks')}
                for scenario,item in value.items()}}

def build():
    cfg=load('configs/bench_v02.json')
    p='data/processed/bench_v02/'
    a1=check_rows(p+'a1_broad_candidates.jsonl','A1')
    a1['strict_comparable_pairs']=sum(1 for _ in jsonl(ROOT/(p+'a1_strict_pairs.jsonl'))) if (ROOT/(p+'a1_strict_pairs.jsonl')).exists() else 0
    a1['source_audit']=p+'a1_preparation_summary.json'
    a1['formal_training_ready']=False
    a1['blockers']=['external_original_experiment_and_PEPBI_reference_lock_pending',
                    'peptide_and_target_family_split_not_frozen','target_construct_and_condition_comparability_incomplete']
    if cfg['panels']['A1'].get('external_coverage_supplement'):
        a1['external_coverage_supplement']=coverage_supplement_status(cfg['panels']['A1']['external_coverage_supplement'])
        a1['blockers']=['unified_multitask_cross_panel_supervision_audit_not_frozen',
                        'target_construct_and_condition_comparability_incomplete',
                        'PepInter_original_experiment_and_peptide_parent_lineage_provenance_incomplete']
    rt=load(p+'a2_audit.json')
    rt_reconciled=load(p+'a2_source_irt_reconciliation.json')
    a2=check_rows(p+'a2_target_march.jsonl','A2')
    a2['source_candidate_rows']=rt.get('source_domain_rows_local',0)
    a2['source_verified_measurement_rows']=0
    a2['source_audit']=p+'a2_audit.json'
    a2['formal_training_ready']=False
    a2['blockers']=['PXD005573_RT_coordinate_and_unit_unverified','source_target_PTM_and_fixed_modification_semantics_unreconciled',
                    'global_parent_near_sequence_split_not_frozen']
    a2['budgets']=p+'a2_target_budgets.json'
    if rt_reconciled:
        a2['source_scale_reconciliation']={key:rt_reconciled.get(key) for key in
            ('status','original_iRT_labels_exactly_equal','label_value_changes','rt_unit',
             'missing_N_acetyl_restored_rows','native_D_E_or_upstream_deamidation_uncertain_rows',
             'chemical_identity_resolved_rows','exact_2026_source_version_verified','cross_condition_training_ready','min_shift_performed')}
        a2['source_verified_measurement_rows']=rt_reconciled.get('original_iRT_labels_exactly_equal',0)
        a2['source_chemically_resolved_rows']=rt_reconciled.get('chemical_identity_resolved_rows',0)
        a2['blockers']=['exact_2026_source_snapshot_unverified',
                        'upstream_lossy_deamidation_identity_unresolved',
                        'global_source_target_new_sequence_split_not_frozen']
    b1_config=cfg['panels']['B1']
    b1=check_rows(b1_config['source_records'],'B1')
    elm=load(b1_config['class_manifest'])
    human=load(b1_config['human_sensitivity_manifest'])
    b1.update({'classes':elm.get('eligible_classes',0),'split_rows':elm.get('eligible_split_counts',elm.get('split_records',{})),
               'split_classes':elm.get('eligible_split_class_counts',elm.get('split_classes',{})),'snapshot':elm.get('snapshot_date'),
               'formal_training_ready':False,'blockers':['unified_multitask_cross_panel_supervision_audit_not_frozen'],
               'scope_limits':['two_unseen_test_classes']})
    b1['scope']=b1_config['biological_scope']
    if b1_config.get('external_coverage_supplement'):
        b1['external_coverage_supplement']=coverage_supplement_status(b1_config['external_coverage_supplement'])
    b1['human_sensitivity']={'rows':human.get('eligible_samples',0),'classes':human.get('eligible_classes',0),
                             'split_rows':human.get('eligible_split_counts',{})}
    b2={'evidence_cards':human.get('b2_evidence_cards',0),'reviewed_qa':human.get('b2_reviewed_qa',0),
        'formal_training_ready':False,'blockers':['reviewed_experimental_answers_and_heldout_relation_combinations_missing']}
    panels={'A1':a1,'A2':a2,'B1':b1,'B2':b2,
            'C1':{'formal_training_ready':False,'blockers':['requires_locked_A_B_test_sets','independently_trained_constant_control_not_available']},
            'C2':{'formal_training_ready':False,'blockers':['matched_pre_answer_probe_and_causal_extraction_not_implemented',
                  'experimental_contacts_or_perturbation_gold_not_reviewed']},
            'C3':{'formal_training_ready':False,'review_target':200,'reviewed_cases':0,
                  'blockers':['two_blinded_reviewers_and_adjudicated_atomic_fact_gold_pending']}}
    for panel in ('A1','B1'):
        panels[panel]['independent_baseline_execution']=panel_execution_status(panel)
    receipt=load('results/real_run_20261015/final_verification.json')
    legacy=[]
    for relative,expected in receipt.get('source_sha256',{}).items():
        if relative.startswith('results/'):
            path=ROOT/relative
            legacy.append({'path':relative,'unchanged':path.exists() and sha256(path)==expected})
    sources={}
    for name in ('affinity_sources.json','rt_sources.json','elm_sources.json'):
        path=ROOT/'sources/bench_v02'/name
        if path.exists():
            sources[name]=sha256(path)
    completed=completed_baseline_artifacts()
    return {'benchmark':'PepBridge-Bench','version':'0.2','protocol_sha256':sha256(ROOT/'configs/bench_v02.json'),
            'status':'independent_panel_baselines_enabled_unified_training_not_ready',
            'panels':panels,'source_manifests':sources,
            'baseline_implementation':cfg.get('controlled_baselines',{}),
            'phase0_lock':{'status':'not_frozen','external_experiments_excluded':False,
                           'homology_split_verified':False,'class_holdout_verified':False,'ready_panels':[]},
            'legacy':{'trained_runs':18,'status':'preserved_exploratory_v01',
                       'checked_artifacts':len(legacy),'all_unchanged':bool(legacy) and all(r['unchanged'] for r in legacy),
                       'artifacts':legacy},
            'scores':{'completed_independent_baselines':completed,'unified_multitask_connector':None},
            'new_model_training_performed':bool(completed),
            'unified_connector_training_performed':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/bench_v02/readiness.json')
    parser.add_argument('--panel',choices=['A1','A2','B1','B2','C1','C2','C3'])
    parser.add_argument('--require-ready',action='store_true')
    parser.add_argument('--scope',choices=['unified','panel-baseline'],default='unified')
    args=parser.parse_args()
    report=build()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'output':str(args.output),
                      'rows':{p:r.get('rows') for p,r in report['panels'].items() if 'rows' in r},
                      'legacy_unchanged':report['legacy']['all_unchanged']},ensure_ascii=False))
    if not report['legacy']['all_unchanged']:
        raise SystemExit('Legacy result/checkpoint fingerprint changed')
    if args.require_ready:
        selected=[args.panel] if args.panel else list(report['panels'])
        ready=lambda p:(report['panels'][p].get('independent_baseline_execution',{}).get('baseline_training_ready',False)
                        if args.scope=='panel-baseline' else report['panels'][p]['formal_training_ready'])
        if any(not ready(p) for p in selected):
            print(json.dumps({'training_refused':selected,'reason':'phase0_and_panel_prerequisites_not_verified'}))
            raise SystemExit(2)

if __name__=='__main__':
    main()

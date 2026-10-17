"""Synthetic analysis fixtures only; no real score, canonical acceptance or GPU run."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

SCOPE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(SCOPE))
from src import mm_evaluate as ev
from src.mm_queue import fixed_cases
spec=importlib.util.spec_from_file_location('v4_analysis_fixture',SCOPE/'resources/analyze_v4.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf8');return path


def fake_result(case,error=1.,lm_valid=True):
    group={'n':3,'mae':error,'rmse':error,'spearman':1.,'target_macro_mae':error}
    return {'scope_id':a.SCOPE_ID,'protocol_sha256':'a'*64,'evaluation_protocol_sha256':'b'*64,
        'backbone':case['backbone'],'scenario':case['scenario'],'arm':case['variant'],'seed':case['seed'],
        'status':'complete','phase':'formal','formal_eligible':True,
        'LM':{'n_total':3,'n_valid':3 if lm_valid else 2,'n_failed':0 if lm_valid else 1,
              'full_sample_mae':error if lm_valid else None,'parse_valid_rate':1. if lm_valid else 2/3,'regression_valid_only':group},
        'numeric':{'coverage':1.,'full_population':group},'C1':{'G_off':.1,'numeric_off_minus_on_MAE':.2}}


def complete_registry():
    rows=[]
    for c in fixed_cases():
        error=2. if c['variant']=='M0' else 1.5 if c['backbone']==ev.BACKBONES[0] else 1.
        r=fake_result(c,error)
        rows.append({**c,'queue_status':'succeeded','analysis_status':'accepted','reason':None,'metrics':a.metric_values(r),'capacity':None,'result':r})
    return {'n_expected':24,'n_canonically_accepted_and_recomputed':24,'queue_status_counts':{'not_started':0,'running':0,'succeeded':24,'failed':0},
        'queue_terminal_receipt_observed':True,'cases':rows}


class AnalysisTests(unittest.TestCase):
    def test_invalid_LM_never_inherits_valid_subset_or_numeric_error(self):
        c=fixed_cases()[0];v=a.metric_values(fake_result(c,lm_valid=False))
        self.assertIsNone(v['LM_full_sample_mae']);self.assertIsNone(v['LM_rmse']);self.assertIsNone(v['LM_spearman'])
        self.assertIsNone(v['LM_target_macro_mae']);self.assertEqual(v['numeric_mae'],1.)
        self.assertEqual(v['LM_parse_valid_rate'],2/3)

    def test_three_seed_within_backbone_and_four_cell_interaction(self):
        extended=a.extended_summary(complete_registry(),ev)
        within=next(x for x in extended['within_backbone_contrasts'] if x['backbone']==ev.BACKBONES[0] and x['scenario']==ev.SCENARIOS[0] and x['metric']=='numeric_mae')
        interaction=next(x for x in extended['four_cell_interactions'] if x['scenario']==ev.SCENARIOS[0] and x['metric']=='numeric_mae')
        self.assertEqual((within['mean'],within['coverage']),(.5,'3/3'))
        self.assertEqual((interaction['mean'],interaction['coverage']),(.5,'3/3'))
        self.assertTrue(all(x['parameter_or_FLOP_matched'] is False for x in extended['backbone_main_effects']))

    def test_partial_seed_never_produces_mean_or_stable_gain_claim(self):
        registry=complete_registry();registry['cases'][0]['metrics']=a.metric_values(None)
        extended=a.extended_summary(registry,ev)
        within=next(x for x in extended['within_backbone_contrasts'] if x['backbone']==ev.BACKBONES[0] and x['scenario']==ev.SCENARIOS[0] and x['metric']=='numeric_mae')
        interaction=next(x for x in extended['four_cell_interactions'] if x['scenario']==ev.SCENARIOS[0] and x['metric']=='numeric_mae')
        self.assertIsNone(within['mean']);self.assertIsNone(within['sample_SD']);self.assertEqual(interaction['coverage'],'2/3')
        self.assertIn('不能判断',a.stable_statement(within))
        within.update(coverage='3/3',per_seed={'17':1.,'42':-1.,'2026':1.})
        self.assertIn('方向不一致',a.stable_statement(within))
        within['per_seed']={'17':0.,'42':-1.,'2026':0.}
        self.assertIn('均未降低',a.stable_statement(within))

    def test_outputs_are_new_version_and_prior_summary_exactly_compared(self):
        registry=complete_registry();extended=a.extended_summary(registry,ev)
        summary={'scope_id':a.SCOPE_ID,'n_expected':24,'cases':[]}
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);out=root/'v1'
            receipt=a.write_outputs(out,registry,summary,extended,{'deadline':'fixed_fixture_deadline'})
            self.assertEqual(set(receipt['files']),{'REPORT.md','execution_registry.json','formal_summary.json','metrics.csv','contrasts.csv'})
            report=(out/'REPORT.md').read_text(encoding='utf8')
            self.assertIn('24/24组',report);self.assertIn('历史v3结果未填入',report)
            self.assertIn('numeric_spearman',(out/'metrics.csv').read_text(encoding='utf-8-sig'))
            second=a.write_outputs(root/'v2',registry,summary,extended,{'deadline':'fixed_fixture_deadline'},expected_summary=out/'formal_summary.json')
            self.assertIs(second['prior_summary_match'],True)
            changed=put(root/'wrong_summary.json',{'wrong':True})
            bad=a.write_outputs(root/'v3',registry,summary,extended,{'deadline':'fixed_fixture_deadline'},expected_summary=changed)
            self.assertIs(bad['prior_summary_match'],False)
            self.assertIn('不一致',(root/'v3/REPORT.md').read_text(encoding='utf8'))
            with self.assertRaises(FileExistsError):a.write_outputs(out,registry,summary,extended,{'deadline':'fixture'})

    def test_fresh_canonical_validation_requires_actual_file_checks_and_no_skip(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);standard=root/'standard';standard.mkdir()
            validator=standard/'validate_reproduction.py'
            validator.write_text("def validate(contract,record,**kwargs):\n    assert kwargs['check_files'] is True\n    assert kwargs['contract_bytes']\n    return {'valid':True,'valid_for_acceptance':True,'files_skipped':[]}\n",encoding='utf8')
            attempt=root/'attempt';put(attempt/'contract/contract.json',{'fixture':True});put(attempt/'archive/result-record.json',{'fixture':True})
            plan={'paths':{'standard_root':str(standard)}}
            self.assertTrue(a.canonical_recheck(plan,attempt)['valid_for_acceptance'])
            validator.write_text("def validate(*args,**kwargs):\n    return {'valid':True,'valid_for_acceptance':True,'files_skipped':['fixture_unchecked']}\n",encoding='utf8')
            with self.assertRaisesRegex(ValueError,'check_files'):a.canonical_recheck(plan,attempt)

    def test_duplicate_or_nonfinite_input_json_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'bad.json'
            for value in ('{"x":1,"x":2}','{"x":NaN}','{"x":1e309}'):
                p.write_text(value,encoding='utf8')
                with self.subTest(value=value),self.assertRaises(ValueError):a.read(p)

    def test_queue_authoritative_and_bad_revalidation_excluded_without_erasing24(self):
        with tempfile.TemporaryDirectory() as t:
            scope=Path(t)/'scope';root=scope/'runs/queue';root.mkdir(parents=True)
            config=put(scope/'configs/frozen.json',{'qualified_interface_parameter_counts':{b:10 for b in ev.BACKBONES}})
            protocol=put(scope/'docs/eval.json',{'fixture':True})
            plan={'paths':{'scope_root':str(scope),'config':str(config),'evaluation_protocol':str(protocol),'data_root':str(scope/'fixture_data')},
                'asset_pins':{str(config):'a'*64},'timeout_seconds_per_case':100}
            put(root/'plan.json',plan);(root/'queue_events.jsonl').write_text('SYNTHETIC_FIXTURE_ONLY')
            states={};results={};calls=[]
            cases=fixed_cases()
            for i,c in enumerate(cases):
                state={'status':'succeeded' if i<3 else 'failed' if i==3 else 'not_started','sequence':i,'reason':'fixture_failure' if i==3 else None}
                if i<3:
                    state.update(attempt=f'cases/{c["case_id"]}/attempt_0001',receipt_sha256='f'*64)
                    attempt=root/state['attempt'];manifest={'model_policy':{'trainable_parameters':99 if i==2 else 10},'elapsed_seconds':1.}
                    put(attempt/'case/evaluate/run_manifest.json',manifest)
                    put(attempt/'case/fit/run_manifest.json',{'runtime':{'training_performance':{'elapsed_seconds':1.}},'elapsed_seconds':2.})
                    put(attempt/'supervision/receipt.json',{'elapsed_seconds':3.,'peak_group_rss_bytes':100})
                    for phase in ('fit','evaluate'):put(attempt/'case'/f'{phase}_gpu_limit.json',{'peak_allocated_bytes':100,'peak_reserved_bytes':200,'allocator_budget_bytes':20*1024**3,'visible_devices':'4'})
                    result=fake_result(c);put(attempt/'case/result.json',result)
                    if i==1:result={**result,'LM':{**result['LM'],'full_sample_mae':999.}}
                    results[str(attempt/'case/evaluate/run_manifest.json')]=result
                states[c['case_id']]=state
            class Ledger:
                def __init__(self,*_):
                    self.states=states;self.events=[{'sequence':0,'event':'controller_start'},{'sequence':99,'event':'controller_terminal'}]
            queue=SimpleNamespace(validate_plan=lambda p:True,verify_pins=lambda p:True,Ledger=Ledger,fixed_cases=fixed_cases,
                accepted_attempt=lambda *_:{'receipt_sha256':'f'*64})
            def evaluate(path,*_):calls.append(str(path));return results[str(path)]
            evaluator=SimpleNamespace(evaluate_run=evaluate,summarize_runs=ev.summarize_runs)
            registry,summary,_=a.collect(scope,root,queue_module=queue,evaluator=evaluator,recheck=lambda *_:{'valid':True,'valid_for_acceptance':True,'files_skipped':[]})
            self.assertEqual(len(registry['cases']),24);self.assertEqual(len(calls),3)
            self.assertEqual(registry['analysis_status_counts'],{'accepted':1,'not_accepted':21,'revalidation_rejected':2})
            self.assertEqual(summary['n_formal_complete'],1)
            self.assertEqual(registry['queue_status_counts']['succeeded'],3)
            self.assertTrue(all(v is None for v in registry['cases'][1]['metrics'].values()))
            self.assertEqual(registry['cases'][3]['queue_status'],'failed')
            self.assertEqual(registry['cases'][4]['queue_status'],'not_started')
            self.assertIn('differs',registry['cases'][1]['reason'])
            self.assertIn('interface count',registry['cases'][2]['reason'])


if __name__=='__main__':unittest.main()

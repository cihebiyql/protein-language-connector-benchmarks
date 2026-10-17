"""Model-free v4 terminal audit and Chinese report from the authoritative queue.

Execute where original canonical paths resolve. No model/Torch import, fitting,
plot dependencies, source mutation, deadline extension, Git operation or GPU use.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

SCOPE_ID='backbone_comparison_v4_20261017'
METRICS=('LM_full_sample_mae','LM_rmse','LM_spearman','LM_target_macro_mae','LM_parse_valid_rate',
         'numeric_mae','numeric_rmse','numeric_spearman','numeric_target_macro_mae','numeric_coverage',
         'C1_G_off','C1_numeric_off_minus_on_MAE')
ERROR_METRICS=('LM_full_sample_mae','LM_rmse','LM_target_macro_mae','numeric_mae','numeric_rmse','numeric_target_macro_mae')


def read(path):
    def pairs(values):
        require(len({k for k,_ in values})==len(values),'duplicate JSON key in analysis input')
        return dict(values)
    value=json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=pairs,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite analysis input')))
    pending=[value]
    while pending:
        item=pending.pop()
        if isinstance(item,dict):pending.extend(item.values())
        elif isinstance(item,list):pending.extend(item)
        elif type(item) is float:require(math.isfinite(item),'nonfinite analysis input')
    return value


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def require(ok,reason):
    if not ok:raise ValueError(reason)


def runtime(scope):
    sys.path.insert(0,str(scope))
    from src import mm_queue,mm_evaluate
    require(all(Path(m.__file__).resolve().parent==(Path(scope).resolve()/'src') for m in (mm_queue,mm_evaluate)),'analysis imported code from a different scope')
    return mm_queue,mm_evaluate


def canonical_recheck(plan,attempt):
    validator=Path(plan['paths']['standard_root'])/'validate_reproduction.py'
    spec=importlib.util.spec_from_file_location('v4_analysis_canonical_validator',validator)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    contract=attempt/'contract/contract.json';record=attempt/'archive/result-record.json'
    result=module.validate(read(contract),read(record),contract_bytes=contract.read_bytes(),check_files=True,
        contract_directory=contract.parent,result_directory=record.parent)
    require(result.get('valid') is True and result.get('valid_for_acceptance') is True and not result.get('files_skipped'),
        'fresh canonical check_files terminal validation rejected this attempt')
    return result


def metric_values(result):
    if result is None:return {key:None for key in METRICS}
    lm=result.get('LM') or {};numeric=result.get('numeric') or {};c1=result.get('C1') or {}
    complete=lm.get('n_total',0)>0 and lm.get('n_valid')==lm.get('n_total') and lm.get('n_failed')==0
    # Never present valid-only regression as full-population LM results.
    full_lm=lm.get('regression_valid_only',{}) if complete else {}
    full_num=numeric.get('full_population') or {}
    values={'LM_full_sample_mae':lm.get('full_sample_mae'),'LM_rmse':full_lm.get('rmse'),
        'LM_spearman':full_lm.get('spearman'),'LM_target_macro_mae':full_lm.get('target_macro_mae'),
        'LM_parse_valid_rate':lm.get('parse_valid_rate'),'numeric_coverage':numeric.get('coverage'),
        **{'numeric_'+k:full_num.get(k) for k in ('mae','rmse','spearman','target_macro_mae')},
        'C1_G_off':c1.get('G_off'),'C1_numeric_off_minus_on_MAE':c1.get('numeric_off_minus_on_MAE')}
    require(all(v is None or type(v) in (int,float) and math.isfinite(v) for v in values.values()),'nonfinite analysis metric')
    return values


def capacity(attempt,manifest,case_timeout=None):
    fit=read(attempt/'case/fit/run_manifest.json');supervision=read(attempt/'supervision/receipt.json')
    caps={phase:read(attempt/'case'/f'{phase}_gpu_limit.json') for phase in ('fit','evaluate')}
    require(caps['fit'].get('visible_devices')==caps['evaluate'].get('visible_devices')
        and all(cap.get('allocator_budget_bytes')==20*1024**3
            and all(type(cap.get(k)) is int and 0<cap[k]<=20*1024**3 for k in ('peak_allocated_bytes','peak_reserved_bytes')) for cap in caps.values()),
        'actual fit/evaluate GPU assignment or budget differs')
    return {'trainable_interface_parameters':manifest.get('model_policy',{}).get('trainable_parameters'),
        'fit_training_seconds':fit.get('runtime',{}).get('training_performance',{}).get('elapsed_seconds'),
        'fit_total_seconds':fit.get('elapsed_seconds'),'evaluation_total_seconds':manifest.get('elapsed_seconds'),
        'case_elapsed_seconds':supervision.get('elapsed_seconds'),'case_timeout_seconds':case_timeout,
        'peak_group_rss_bytes':supervision.get('peak_group_rss_bytes'),
        'peak_allocated_bytes':max(caps[p]['peak_allocated_bytes'] for p in caps),
        'peak_reserved_bytes':max(caps[p]['peak_reserved_bytes'] for p in caps),
        'allocator_budget_bytes':caps['fit']['allocator_budget_bytes'],
        'physical_gpu':caps['fit']['visible_devices'],
        'allocator_budget_includes_CUDA_context':False,
        'communication_slots':58,'communication_width':128,'communication_dtype':'float32','logical_communication_bytes':29696}


def collect(scope,queue_root,*,queue_module=None,evaluator=None,recheck=None):
    scope,root=Path(scope).resolve(strict=True),Path(queue_root).resolve(strict=True)
    queue_module,evaluator=(queue_module,evaluator) if queue_module and evaluator else runtime(scope)
    plan_path=root/'plan.json';plan=read(plan_path)
    queue_module.validate_plan(plan);queue_module.verify_pins(plan)
    require(Path(plan['paths']['scope_root']).resolve()==scope,'analysis must use original frozen execution scope')
    ledger=queue_module.Ledger(root/'queue_events.jsonl',sha(plan_path))
    fixed=queue_module.fixed_cases()
    require(len(fixed)==24 and set(ledger.states)=={x['case_id'] for x in fixed},'authoritative queue does not register exact24 cases')
    protocol_path=Path(plan['paths']['evaluation_protocol']);data_root=Path(plan['paths']['data_root'])
    config=read(plan['paths']['config'])
    results,registry=[],[]
    for case in fixed:
        event=ledger.states[case['case_id']];status=event['status']
        entry={**case,'queue_status':status,'analysis_status':'not_accepted','reason':event.get('reason'),
            'queue_error':event.get('error'),'attempt':event.get('attempt'),'queue_event_sequence':event['sequence'],'result':None,'metrics':metric_values(None),'capacity':None}
        if status=='succeeded':
            try:
                attempt=(root/event['attempt']).resolve(strict=True)
                require(attempt.is_relative_to(root),'queue attempt escapes frozen queue root')
                binding=queue_module.accepted_attempt(plan,attempt,case)
                for key,value in binding.items():require(event.get(key)==value,'queue success event SHA differs from current terminal attempt')
                validation=(recheck or canonical_recheck)(plan,attempt)
                manifest_path=attempt/'case/evaluate/run_manifest.json'
                result=evaluator.evaluate_run(manifest_path,protocol_path,data_root)
                saved=read(attempt/'case/result.json')
                require(canonical(result)==canonical(saved),'saved scientific result differs from fresh model-free evaluation')
                require(result.get('formal_eligible') is True and result.get('status')=='complete','recomputed result is not complete formal output')
                manifest=read(manifest_path)
                cap=capacity(attempt,manifest,plan.get('timeout_seconds_per_case'))
                expected=config.get('qualified_interface_parameter_counts',{}).get(case['backbone'])
                require(type(expected) is int and cap['trainable_interface_parameters']==expected,'actual trainable interface count differs from qualified backbone budget')
                results.append(result)
                entry.update(analysis_status='accepted',reason=None,result=result,metrics=metric_values(result),capacity=cap,
                    binding=binding,run_manifest_sha256=sha(manifest_path),fresh_canonical_validation=validation)
            except (OSError,ValueError,KeyError,TypeError,OverflowError) as exc:
                entry.update(analysis_status='revalidation_rejected',reason=str(exc),error_type=type(exc).__name__)
        registry.append(entry)
    summary=evaluator.summarize_runs(results,plan['asset_pins'][plan['paths']['config']])
    # The evaluator's numerical registry receives only accepted scores. Report queue states alongside it.
    indexed={(x['backbone'],x['scenario'],x['variant'],x['seed']):x for x in registry}
    for row in summary['cases']:
        original=indexed[(row['backbone'],row['scenario'],row['arm'],row['seed'])]
        row.update(queue_status=original['queue_status'],analysis_status=original['analysis_status'],reason=original['reason'])
    starts=[e['sequence'] for e in ledger.events if e.get('event')=='controller_start']
    ends=[e['sequence'] for e in ledger.events if e.get('event')=='controller_terminal']
    terminal=bool(ends and (not starts or ends[-1]>starts[-1]))
    execution={'schema_version':'v4.execution_registry.v1','scope_id':SCOPE_ID,'created_utc':datetime.now(timezone.utc).isoformat(),
        'queue_plan_sha256':sha(plan_path),'queue_ledger_sha256':sha(root/'queue_events.jsonl'),
        'evaluation_protocol_sha256':sha(protocol_path),'config_sha256':sha(plan['paths']['config']),
        'analysis_source_sha256':sha(__file__),'queue_terminal_receipt_observed':terminal,
        'n_expected':24,'n_canonically_accepted_and_recomputed':len(results),'cases':registry,
        'queue_status_counts':{s:sum(x['queue_status']==s for x in registry) for s in ('not_started','running','succeeded','failed')},
        'analysis_status_counts':{s:sum(x['analysis_status']==s for x in registry) for s in ('accepted','not_accepted','revalidation_rejected')},
        'historical_results_counted':False,'source_artifacts_modified':False,'models_or_GPU_executed':False}
    return execution,summary,plan


def extended_summary(registry,evaluator):
    """All scoring channels, complete seeds only; comparisons have explicit directions."""
    values={(r['backbone'],r['scenario'],r['variant'],r['seed']):r['metrics'] for r in registry['cases']}
    def get(b,s,a,seed,m):return values[(b,s,a,seed)][m]
    groups=[];contrasts=[];interactions=[];main=[]
    for b in evaluator.BACKBONES:
        for s in evaluator.SCENARIOS:
            for a in evaluator.ARMS:
                groups.append({'backbone':b,'scenario':s,'arm':a,'metrics':{m:evaluator.seed_statistics({str(seed):get(b,s,a,seed,m) for seed in evaluator.SEEDS}) for m in METRICS}})
            for m in METRICS:
                per={}
                for seed in evaluator.SEEDS:
                    simple,full=(get(b,s,a,seed,m) for a in evaluator.ARMS)
                    per[str(seed)]=simple-full if simple is not None and full is not None else None
                contrasts.append({'backbone':b,'scenario':s,'metric':m,'formula':'M0-M3',
                    'positive_means':'lower_error_with_M3' if m in ERROR_METRICS else 'M3_value_is_lower; inspect metric direction',**evaluator.seed_statistics(per)})
    for s in evaluator.SCENARIOS:
        for m in METRICS:
            per={}
            for seed in evaluator.SEEDS:
                v=[get(b,s,a,seed,m) for b in evaluator.BACKBONES for a in evaluator.ARMS]
                per[str(seed)]=(v[2]-v[3])-(v[0]-v[1]) if all(x is not None for x in v) else None
            interactions.append({'scenario':s,'metric':m,'formula':'(M0-M3)_4B-(M0-M3)_1.5B',
                'positive_means':'larger_M3_error_reduction_on_4B' if m in ERROR_METRICS else 'larger_M0_minus_M3_value_difference_on_4B',**evaluator.seed_statistics(per)})
            for a in evaluator.ARMS:
                per={}
                for seed in evaluator.SEEDS:
                    old,new=(get(b,s,a,seed,m) for b in evaluator.BACKBONES)
                    per[str(seed)]=old-new if old is not None and new is not None else None
                main.append({'scenario':s,'arm':a,'metric':m,'formula':'1.5B-4B',
                    'parameter_or_FLOP_matched':False,**evaluator.seed_statistics(per)})
    return {'configurations':groups,'within_backbone_contrasts':contrasts,'four_cell_interactions':interactions,'backbone_main_effects':main}


def fmt(value):return '未定义' if value is None else f'{value:.6f}' if type(value) in (int,float) else str(value)


def stable_statement(contrast):
    vals=list(contrast['per_seed'].values())
    if contrast['coverage']!='3/3':return '种子不足，不能判断稳定提升'
    if all(v>0 for v in vals):return '三个种子均降低误差；仍限于当前固定测试'
    if all(v<=0 for v in vals):return '三个种子均未降低误差'
    return '种子方向不一致，未见稳定提升'


def report_text(registry,summary,extended,plan,prior_match):
    count=registry['n_canonically_accepted_and_recomputed']
    lines=['# Qwen底座与连接器对照：核验结果','',
        f'已核验并重新计算 **{count}/24组**；队列状态：'+canonical(registry['queue_status_counts'])+'。',
        '队列终态收据：'+('已观察到。' if registry['queue_terminal_receipt_observed'] else '未观察到；这是进行中的快照。'),
        '所有分数来自规范终态已接受、当前文件SHA核验和模型无关重算一致的任务。失败／未启动／运行中／重验拒绝均保留，未作为模型负分。',
        '固定划分、严格JSON解析、完整测试分母和验证NLL选点不变。数值头不回填语言输出；格式改善本身不能证明预测误差下降。',
        '历史v3结果未填入本轮。初期负分或无提升不构成技术失败；本工具不运行或恢复训练。','',
        '## 主要比较','',
        '| 底座 | 场景 | 指标 | 同种子M0−M3均值±样本SD | 完整种子 | 判断 |',
        '|---|---|---|---|---|---|']
    for row in extended['within_backbone_contrasts']:
        if row['metric'] not in ('LM_full_sample_mae','numeric_mae'):continue
        lines.append('| '+' | '.join([row['backbone'],row['scenario'],row['metric'],fmt(row['mean'])+' ± '+fmt(row['sample_SD']),row['coverage'],stable_statement(row)])+' |')
    lines+=['','正的M0−M3表示完整连接器误差更低。仅3/3完整种子报告均值和样本SD；三个种子同向仍不能代替新数据确认或统计功效分析。','',
        '## 底座与连接器交互','',
        '| 场景 | 指标 | (M0−M3)4B−(M0−M3)1.5B | 完整四格种子 |','|---|---|---|---|']
    for row in extended['four_cell_interactions']:
        if row['metric'] in ('LM_full_sample_mae','numeric_mae'):
            lines.append('| '+' | '.join([row['scenario'],row['metric'],fmt(row['mean'])+' ± '+fmt(row['sample_SD']),row['coverage']])+' |')
    lines+=['','该值为正表示4B上的完整连接器误差收益更大。只有同一个种子的四格全部通过核验才计算该种子。底座主效应另列于JSON/CSV；跨底座LM参数、接口总参数和FLOPs不同。','',
        '## 全部24位置','',
        '| 底座 | 场景 | 接口 | 种子 | 队列／分析状态 | LM MAE | JSON有效率 | 数值MAE | 原因 |','|---|---|---|---|---|---|---|---|---|']
    for r in registry['cases']:
        lines.append('| '+' | '.join([r['backbone'],r['scenario'],r['variant'],str(r['seed']),r['queue_status']+'/'+r['analysis_status'],
            fmt(r['metrics']['LM_full_sample_mae']),fmt(r['metrics']['LM_parse_valid_rate']),fmt(r['metrics']['numeric_mae']),str(r['reason'] or '').replace('|','/').replace('\n',' ')])+' |')
    lines+=['','## 多指标与计算预算','',
        'metrics.csv和formal_summary.json同时保留RMSE、含并列秩处理的Spearman、按靶点宏平均误差、JSON有效率、数值覆盖率及同权重C1 on/off。LM仅当整个人口全部有效时才报告完整人口RMSE／Spearman／宏平均；有效子集指标不会冒充完整测试指标。',
        '每个通过任务保存真实训练／评估／进程组耗时、实际接口参数、显存allocated/reserved峰值、RSS和GPU分配。相同通信预算为58×128×float32＝29,696逻辑字节；CUDA上下文／库开销不包含在PyTorch allocator上限内。',
        'C1 G_off为关闭连接器后每答案token NLL的平均增量；numeric_off_minus_on_MAE为关闭后的数值误差增量。它们衡量同权重分支依赖，不构成生物学因果／实验可比性证明。当前无独立生物donor或同条件配对金标准，P／H保持未定义。',
        'temperature／pH／buffer／实际construct缺失时，只报告缺失数量，不虚构条件分层或可比配对。',
        '绝对截止：'+plan['deadline']+'。读取与分析不会延长或重开训练窗口。']
    lines+=['','| 底座 | 已接受组数 | 接口参数 | 已接受任务累计秒 | 最大reserved GiB | 最大RSS GiB |','|---|---|---|---|---|---|']
    for b in ('qwen25_1p5b','qwen3_4b_2507'):
        observed=[r['capacity'] for r in registry['cases'] if r['backbone']==b and r.get('analysis_status')=='accepted' and r.get('capacity')]
        parameters=sorted({r['trainable_interface_parameters'] for r in observed})
        elapsed=[r['case_elapsed_seconds'] for r in observed if type(r.get('case_elapsed_seconds')) in (int,float)]
        reserved=[r['peak_reserved_bytes']/1024**3 for r in observed]
        rss=[r['peak_group_rss_bytes']/1024**3 for r in observed if type(r.get('peak_group_rss_bytes')) in (int,float)]
        lines.append('| '+' | '.join([b,str(len(observed)),','.join(str(x) for x in parameters) or '未记录',
            fmt(sum(elapsed) if elapsed else None),fmt(max(reserved) if reserved else None),fmt(max(rss) if rss else None)])+' |')
    lines+=['','计算资源表仅描述已接受任务，不把未完成任务估算为已运行，也不把并行任务累计秒当作自然耗时。']
    if prior_match is not None:lines+=['','提供的旧summary与本次规范重算结果：'+('一致。' if prior_match else '**不一致，已保留原文件并生成独立新版本，不能声称旧汇总已核验通过。**')]
    return '\n'.join(lines)+'\n'


def write_outputs(output,registry,summary,extended,plan,*,expected_summary=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    prior=None
    if expected_summary is not None:
        previous=read(expected_summary)
        previous.pop('extended_metrics',None)
        prior=canonical(previous)==canonical(summary)
    summary={**summary,'extended_metrics':extended}
    for name,value in [('execution_registry.json',registry),('formal_summary.json',summary)]:
        with (output/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    fields=('case_id','backbone','scenario','variant','seed','queue_status','analysis_status','reason','queue_error',*METRICS,
        'trainable_interface_parameters','fit_training_seconds','fit_total_seconds','evaluation_total_seconds','case_elapsed_seconds',
        'peak_group_rss_bytes','peak_allocated_bytes','peak_reserved_bytes','allocator_budget_bytes','physical_gpu','logical_communication_bytes')
    with (output/'metrics.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader()
        for r in registry['cases']:
            writer.writerow({k:v for k,v in {**r,**r['metrics'],**(r['capacity'] or {})}.items() if k in fields})
    with (output/'contrasts.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        fields=('kind','backbone','scenario','arm','metric','formula','coverage','mean','sample_SD','seed17','seed42','seed2026')
        writer=csv.DictWriter(stream,fields);writer.writeheader()
        for kind,items in [('within_backbone',extended['within_backbone_contrasts']),('four_cell_interaction',extended['four_cell_interactions']),('backbone_main_effect',extended['backbone_main_effects'])]:
            for item in items:
                writer.writerow({k:v for k,v in {**item,'kind':kind,**{'seed'+k:v for k,v in item['per_seed'].items()}}.items() if k in fields})
    (output/'REPORT.md').write_text(report_text(registry,summary,extended,plan,prior),encoding='utf-8')
    receipt={'scope_id':SCOPE_ID,'analysis_source_sha256':sha(__file__),'prior_summary_match':prior,
        'source_artifacts_modified':False,'models_or_GPU_executed':False,
        'files':{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}}
    if expected_summary is not None:receipt['expected_summary']={'path':str(Path(expected_summary).resolve()),'sha256':sha(expected_summary)}
    with (output/'analysis_receipt.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('scope-root','queue-root','output'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--expected-summary',type=Path)
    args=parser.parse_args(argv)
    registry,summary,plan=collect(args.scope_root,args.queue_root)
    _,ev=runtime(args.scope_root)
    receipt=write_outputs(args.output,registry,summary,extended_summary(registry,ev),plan,expected_summary=args.expected_summary)
    print(json.dumps({'status':'analyzed','accepted':registry['n_canonically_accepted_and_recomputed'],'expected':24,'output':str(args.output),'prior_summary_match':receipt['prior_summary_match']}))
    return 2 if receipt['prior_summary_match'] is False or registry['analysis_status_counts']['revalidation_rejected'] else 0


if __name__=='__main__':raise SystemExit(main())

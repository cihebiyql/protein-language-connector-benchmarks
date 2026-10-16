"""Summarize all predeclared cases, including missing seeds and parse failures.

Positive contrast means the named treatment reduced full-population MAE.
Group bootstrap describes a fixed fitted model on the frozen test. Seed sample
SD is reported separately and only when all three seeds have valid full MAE.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from matched_budget_20261016.execute_suite import ROOT, case_id, load, sha, trial_matrix, write

# This frozen module is the same parser/metric implementation used by the
# original benchmark. New summaries verify its source before replaying scores.
LEGACY_PATH=Path(__file__).resolve().parents[1]/'foundation_20261016/run_connectors.py'
LEGACY_SHA256='88da80f9c75116625c2ec8601d4d5685882c4d899f693371338179f36e905da5'


def frozen_scoring():
    if sha(LEGACY_PATH)!=LEGACY_SHA256:
        raise ValueError('Frozen original parser/metric implementation changed')
    from foundation_20261016 import run_connectors
    return run_connectors


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def metrics_summary(metrics,expected):
    counts=[metrics.get(k) for k in ('n_total','n_valid','n_failed')]
    if any(not isinstance(x,int) or isinstance(x,bool) or x<0 for x in counts):
        raise ValueError('Missing/invalid generation denominators')
    total,valid,failed=counts
    if total!=expected or valid+failed!=total:
        raise ValueError('Full population mismatch')
    coverage=metrics.get('parse_valid_rate')
    if not finite(coverage) or abs(coverage-valid/total)>1e-12:
        raise ValueError('Coverage not consistent with literal counts')
    mae=metrics.get('full_sample_mae')
    if failed and mae is not None: raise ValueError('Full MAE defined despite generation failure')
    if not failed and (not finite(mae) or mae<0): raise ValueError('Missing/nonfinite full MAE')
    conditional=metrics.get('regression_valid_only',{}).get('mae')
    if valid and (not finite(conditional) or conditional<0): raise ValueError('Invalid conditional MAE')
    return dict(n_total=total,n_valid=valid,n_failed=failed,parse_valid_rate=coverage,
                full_sample_mae=mae,valid_only_mae=conditional,valid_only_denominator=valid,
                failure_counts=metrics.get('failure_counts',{}),target_macro=metrics.get('regression_valid_only',{}))


def seed_aggregate(values,seeds=(17,42,2026)):
    valid={str(seed):values.get(seed) for seed in seeds if finite(values.get(seed))}
    complete=len(valid)==len(seeds)
    return dict(k=len(valid),expected_k=len(seeds),coverage=f'{len(valid)}/{len(seeds)}',
                per_seed={str(seed):values.get(seed) for seed in seeds},
                mean=statistics.mean(valid.values()) if complete else None,
                sample_SD=statistics.stdev(valid.values()) if complete else None,
                missing_invalid_seeds=[seed for seed in seeds if str(seed) not in valid])


def fixed_contrasts():
    # key=(placement, message_input, sender_mode, gate_init), weights sum to zero.
    er=('early','real','balanced','zero'); en=('early','neutral','balanced','zero')
    mr=('mid','real','balanced','zero'); mn=('mid','neutral','balanced','zero')
    return {
        'depth_real':[(er,1),(mr,-1)],
        'depth_neutral':[(en,1),(mn,-1)],
        'content_early':[(en,1),(er,-1)],
        'content_mid':[(mn,1),(mr,-1)],
        'depth_content_difference_in_differences':[(er,1),(mr,-1),(en,-1),(mn,1)],
        'gate_zero_initialization':[(('mid','real','balanced','near_unit'),1),(mr,-1)],
        'balanced_pool_allocation':[(('mid','real','global16','zero'),1),(mr,-1)],
    }


def quantile(sorted_values,p):
    pos=(len(sorted_values)-1)*p
    low=int(pos);high=min(low+1,len(sorted_values)-1)
    return sorted_values[low]+(pos-low)*(sorted_values[high]-sorted_values[low])


def group_bootstrap(rows,per_sample_contrast,draws=2000,seed=2026):
    if len(rows)!=len(per_sample_contrast) or not rows or draws<1:
        raise ValueError('Empty/misaligned group bootstrap inputs')
    groups=defaultdict(list)
    for row,value in zip(rows,per_sample_contrast):
        if not finite(value) or not row.get('group_id'): raise ValueError('Invalid value/group')
        groups[row['group_id']].append(value)
    totals=[(sum(v),len(v)) for v in groups.values()]
    rng=random.Random(seed);samples=[]
    for _ in range(draws):
        selected=rng.choices(totals,k=len(totals))
        samples.append(sum(x[0] for x in selected)/sum(x[1] for x in selected))
    samples.sort()
    return dict(estimate=sum(per_sample_contrast)/len(rows),lower=quantile(samples,.025),upper=quantile(samples,.975),
                confidence=.95,bootstrap_draws=draws,bootstrap_seed=seed,n_measurements=len(rows),
                n_source_family_components=len(groups),unit='pKD',
                interpretation='Fixed-checkpoint frozen-test source/family-component uncertainty; not seed variability or new independent experiment count')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def compare_recomputed(saved,recomputed,path='generation_metrics'):
    """Require all original metric fields; never trust stored counts or MAE."""
    if isinstance(recomputed,dict):
        if not isinstance(saved,dict): raise ValueError(path+' is not an object')
        for key,value in recomputed.items():
            if key not in saved: raise ValueError(path+'.'+key+' missing from stored metrics')
            compare_recomputed(saved[key],value,path+'.'+key)
    elif isinstance(recomputed,list):
        if not isinstance(saved,list) or len(saved)!=len(recomputed): raise ValueError(path+' length mismatch')
        for index,(left,right) in enumerate(zip(saved,recomputed)):
            compare_recomputed(left,right,path+f'[{index}]')
    elif isinstance(recomputed,bool):
        if saved is not recomputed: raise ValueError(path+' disagrees with raw reconstruction')
    elif isinstance(recomputed,int):
        if not isinstance(saved,int) or isinstance(saved,bool) or saved!=recomputed:
            raise ValueError(path+' disagrees with raw reconstruction')
    elif isinstance(recomputed,(int,float)):
        if not finite(saved) or abs(saved-recomputed)>1e-10:
            raise ValueError(path+' disagrees with raw reconstruction')
    elif saved!=recomputed:
        raise ValueError(path+' disagrees with raw reconstruction')


def audit_raw_predictions(job,rows,saved_metrics):
    """Check every row before deciding whether a full-population contrast exists.

    Parse failures remain ordinary prediction records. Missing failure records,
    duplicate IDs, fabricated parse fields and forged aggregate metrics reject
    the whole case rather than silently shrink its denominator.
    """
    legacy=frozen_scoring()
    predictions=read_jsonl(job/'artifacts/test_predictions_correct.jsonl')
    expected_ids=[r['sample_id'] for r in rows]
    if len(expected_ids)!=len(set(expected_ids)):
        raise ValueError('Frozen test contains duplicate sample IDs')
    if len(predictions)!=len(rows) or any(not isinstance(p,dict) for p in predictions):
        raise ValueError('Raw predictions must cover the complete frozen test denominator')
    actual_ids=[p.get('sample_id') for p in predictions]
    if len(actual_ids)!=len(set(actual_ids)):
        raise ValueError('Raw predictions contain duplicate sample IDs')
    if actual_ids!=expected_ids:
        raise ValueError('Raw predictions must cover identical frozen order/population')
    reconstructed=[];errors=[]
    for row,prediction in zip(rows,predictions):
        if prediction.get('split')!='test' or row.get('split')!='test':
            raise ValueError('Prediction is not from the immutable test split')
        if not isinstance(prediction.get('raw_text'),str):
            raise ValueError('Every prediction requires the literal raw_text string')
        parsed=legacy.parse_answer(prediction['raw_text'])
        stored=prediction.get('parse')
        if not isinstance(stored,dict) or set(stored)!=set(parsed):
            raise ValueError('Stored parsed fields differ from frozen raw_text parser')
        if (stored['valid'] is not parsed['valid'] or stored['reason']!=parsed['reason']
                or (parsed['valid'] and (not finite(stored['value']) or stored['value']!=parsed['value']))
                or (not parsed['valid'] and stored['value'] is not None)):
            raise ValueError('Stored parsed fields disagree with frozen raw_text parser')
        if not finite(prediction.get('gold_pkd')) or prediction['gold_pkd']!=row['pkd']:
            raise ValueError('Prediction gold differs from immutable source')
        reconstructed.append({**prediction,'parse':parsed})
        errors.append(abs(parsed['value']-float(row['pkd'])) if parsed['valid'] else None)
    recomputed=legacy.generation_metrics(rows,reconstructed)
    compare_recomputed(saved_metrics,recomputed)
    return recomputed,errors if recomputed['n_failed']==0 else None


def mechanism_scalars(actual,history):
    """Keep diagnostic estimands separate from the generation endpoint."""
    values={};units={}
    def put(key,value,unit):
        if value is None or finite(value):
            values[key]=value;units[key]=unit
        else:
            raise ValueError('Nonfinite or nonnumeric mechanism scalar: '+key)
    put('best_validation_answer_token_nll',actual.get('best_validation_answer_token_nll'),'nats/answer token; original token-weighted checkpoint criterion')
    c1=actual.get('C1') or {}
    put('C1.G_all_test',c1.get('G_all_test'),'nats/answer token; mean of per-sample off-minus-correct NLL')
    for name,value in c1.get('means_all_test',{}).items():
        put('C1.means_all_test.'+name,value,'nats/answer token; mean of per-sample NLL')
    for channel in ('correct','off'):
        for category in ('full','format','numeric','mixed'):
            bucket=c1.get('token_breakdown',{}).get(channel,{}).get(category,{})
            for metric,unit in (('nll_nats_per_token','nats/category target token; global token weighting'),
                                ('contribution_nats_per_full_token','nats/full-answer target token; global token weighting')):
                put(f'C1.token_breakdown.{channel}.{category}.{metric}',bucket.get(metric),unit)
    for name,value in c1.get('G_decomposition',{}).items():
        put('C1.G_decomposition.'+name,value,'nats/full-answer token; mean per-sample contribution; original_G uses original fp32 reduction')
    for item in history:
        epoch=item['epoch']
        for key in ('train_answer_token_nll','validation_answer_token_nll'):
            put(f'epoch{epoch}.{key}',item.get(key),'nats/answer token; original token-weighted criterion')
        for category in ('full','format','numeric','mixed'):
            bucket=item.get('validation_token_nll_breakdown',{}).get(category,{})
            for metric,unit in (('nll_nats_per_token','nats/category target token; global validation weighting'),
                                ('contribution_nats_per_full_token','nats/full-answer target token; global validation weighting')):
                put(f'epoch{epoch}.validation_token_breakdown.{category}.{metric}',bucket.get(metric),unit)
        for phase in ('train','validation'):
            diagnostic=item.get(phase+'_reader_diagnostics',{})
            for key in ('alpha','gate_tanh_alpha','raw_delta_over_hidden_l2','gated_delta_over_hidden_l2'):
                put(f'epoch{epoch}.{phase}.{key}',diagnostic.get(key),'dimensionless scalar/ratio')
    return values,units


def summarize(suite,protocol_path):
    protocol=load(protocol_path)
    cases=trial_matrix(protocol);entries=[];lookup={};error_vectors={}
    rows_by_scenario={}
    for scenario,counts in protocol['scenarios'].items():
        samples=ROOT/protocol['data_root']/scenario/'samples.jsonl'
        if sha(samples)!=counts['samples_sha256']:
            raise ValueError('Frozen scenario samples SHA differs from protocol')
        rows=[row for row in read_jsonl(samples) if row['split']=='test']
        if len(rows)!=counts['test']: raise ValueError('Frozen test denominator differs from protocol')
        rows_by_scenario[scenario]=rows
    for case in cases:
        job=suite/case_id(case)
        entry=dict(case=case,case_id=case_id(case),status='missing',metrics=None)
        artifact=job/'artifacts/result.json'
        if artifact.exists():
            try:
                actual=load(artifact);receipt=load(job/'execution-receipt.json');validation=load(job/'result-validation.json')
                if (actual.get('status')!='complete' or actual.get('optimizer_steps')!=1548 or actual.get('actual_epochs')!=3
                        or receipt.get('exit_code')!=0 or not receipt.get('process_group_absent')
                        or validation.get('valid_for_acceptance') is not True):
                    raise ValueError('Not a completed, terminal, accepted full-budget trial')
                reconstructed,errors=audit_raw_predictions(job,rows_by_scenario[case['scenario']],actual['test_generation']['correct'])
                history=load(job/'artifacts/history.json') if (job/'artifacts/history.json').exists() else []
                scalar_values,scalar_units=mechanism_scalars(actual,history)
                entry.update(status='complete',metrics=metrics_summary(reconstructed,protocol['scenarios'][case['scenario']]['test']),
                    raw_prediction_audit=dict(all_test_rows_checked=True,all_raw_text_reparsed=True,source_parser_sha256=LEGACY_SHA256,
                        prediction_file_sha256=sha(job/'artifacts/test_predictions_correct.jsonl'),stored_parse_and_metrics_match_reconstruction=True),
                    actual_epochs=actual['actual_epochs'],optimizer_steps=actual['optimizer_steps'],best_epoch=actual['best_epoch'],
                    best_validation_answer_token_nll=actual.get('best_validation_answer_token_nll'),
                    train_validation_reader_diagnostics=actual.get('train_validation_reader_diagnostics'),
                    test_reader_diagnostics=actual.get('test_reader_diagnostics'),actual_generated_token_count=actual.get('actual_generated_token_count'),
                    elapsed_seconds=receipt.get('elapsed_seconds'),peak_host_process_group_rss_bytes=receipt.get('peak_host_process_group_rss_bytes'),
                    gpu_peak_allocated_bytes=actual.get('gpu_peak_allocated_bytes'),result_sha256=sha(artifact))
                entry['mechanism']=dict(C1=actual.get('C1'),history=history,numerical_policy=actual.get('numerical_policy'),
                    scalar_values=scalar_values,scalar_units=scalar_units,
                    optimizer_gate_trajectory_evidence=str(job/'artifacts/run.jsonl'),
                    role='Mechanism diagnosis only; never replaces full-population generation MAE or parse coverage')
                if errors is not None:
                    error_vectors[case_id(case)]=errors
            except (ValueError,KeyError,OSError,TypeError) as exc:
                entry.update(status='invalid_or_incomplete',metrics=None,reason=str(exc))
        elif (job/'execution-receipt.json').exists() or (job/'artifacts/failure.json').exists():
            entry['status']='failed_or_incomplete'
        entries.append(entry)
        key=(case['scenario'],case['seed'],case['placement'],case['message_input'],case['sender_mode'],case['gate_init'])
        lookup[key]=entry
    summaries=[]
    for scenario in protocol['scenarios']:
        configurations=list(dict.fromkeys((c['placement'],c['message_input'],c['sender_mode'],c['gate_init']) for c in cases))
        for config in configurations:
            values={seed:(lookup[(scenario,seed,*config)]['metrics'] or {}).get('full_sample_mae') for seed in protocol['seeds']}
            summaries.append(dict(scenario=scenario,configuration=config,**seed_aggregate(values,protocol['seeds'])))
    mechanism_summaries=[]
    for scenario in protocol['scenarios']:
        for config in configurations:
            selected={seed:lookup[(scenario,seed,*config)] for seed in protocol['seeds']}
            keys=sorted({key for entry in selected.values() for key in entry.get('mechanism',{}).get('scalar_values',{})})
            for key in keys:
                values={seed:entry.get('mechanism',{}).get('scalar_values',{}).get(key) for seed,entry in selected.items()}
                unit=next(entry['mechanism']['scalar_units'][key] for entry in selected.values()
                          if key in entry.get('mechanism',{}).get('scalar_units',{}))
                mechanism_summaries.append(dict(scenario=scenario,configuration=config,diagnostic=key,unit=unit,
                    **seed_aggregate(values,protocol['seeds']),role='diagnostic only; no substitution for generation endpoint'))
    contrasts=[]
    declared={x['name'] for x in protocol['paired_contrasts_per_scenario']}
    if declared!=set(fixed_contrasts()): raise ValueError('Declared contrasts changed; no implicit adaptation')
    for scenario,rows in rows_by_scenario.items():
        for name,terms in fixed_contrasts().items():
            values={};intervals={};missing={}
            for seed in protocol['seeds']:
                selected=[(lookup[(scenario,seed,*config)],weight) for config,weight in terms]
                if any(entry['case_id'] not in error_vectors for entry,_ in selected):
                    values[seed]=None
                    missing[str(seed)]=[entry['case_id'] for entry,_ in selected if entry['case_id'] not in error_vectors]
                    continue
                sample=[sum(weight*error_vectors[entry['case_id']][i] for entry,weight in selected) for i in range(len(rows))]
                values[seed]=statistics.mean(sample)
                intervals[str(seed)]=group_bootstrap(rows,sample)
            contrasts.append(dict(scenario=scenario,name=name,orientation='positive favors named treatment',
                **seed_aggregate(values,protocol['seeds']),fixed_model_group_intervals_by_seed=intervals,missing_or_invalid_cases=missing,
                multiplicity='seven preregistered contrasts per scenario; descriptive intervals, no multiplicity-adjusted superiority claim'))
    result=dict(schema_version=1,protocol_sha256=sha(protocol_path),n_expected_training_trials=36,
        n_complete=sum(e['status']=='complete' for e in entries),cases=entries,seed_summaries=summaries,paired_contrasts=contrasts,
        mechanism_seed_summaries=mechanism_summaries,
        seed_uncertainty_and_group_uncertainty_separate=True,old_predictor_status='reference only; not parameter/communication-budget matched',
        historical_pure_Qwen_OFF='Prior frozen OFF reference may be reused explicitly; no new OFF inference counted in these trials')
    write(suite/'matched_summary.json',result)
    lines=['# 同参数与通信预算的固定36项对照','','全部种子、缺失和生成失败均保留。仅3/3个完整有效种子才报告三种子均值和样本SD；固定模型的来源／家族组bootstrap另列。',
           '',f'完整终态训练：{result["n_complete"]}/36。canary不计入训练结果。','',
           '| 场景 | seed | 位置 | 消息 | 池化 | 门控 | 状态 | 有效/总数 | full MAE | valid-only MAE |',
           '|---|---:|---|---|---|---|---|---:|---:|---:|']
    def fmt(v): return f'{v:.5f}' if finite(v) else 'null'
    for e in entries:
        c=e['case'];m=e['metrics'] or {}
        lines.append(f'| {c["scenario"]} | {c["seed"]} | {c["placement"]} | {c["message_input"]} | {c["sender_mode"]} | {c["gate_init"]} | {e["status"]} | {m.get("n_valid","—")}/{m.get("n_total","—")} | {fmt(m.get("full_sample_mae"))} | {fmt(m.get("valid_only_mae"))} |')
    lines+=['','| 场景 | 固定配对差值（正值有利） | 完整种子 | 三种子均值 | 种子样本SD |','|---|---|---:|---:|---:|']
    for c in contrasts:
        lines.append(f'| {c["scenario"]} | {c["name"]} | {c["coverage"]} | {fmt(c["mean"])} | {fmt(c["sample_SD"])} |')
    lines+=['','C1、token分解和门控诊断独立报告；即使full MAE未定义也保留，不能替代主评价。G是逐样本完整答案off−correct NLL的均值；类别NLL使用各类别token分母，贡献项使用完整答案token分母。P/H独立donor仍为0。',
            '','| 场景 | seed | 位置/内容/池化/门控 | 原验证最佳NLL | C1 G | 末轮验证tanh门控 | 末轮验证残差/hidden |',
            '|---|---:|---|---:|---:|---:|---:|']
    for e in entries:
        c=e['case'];values=e.get('mechanism',{}).get('scalar_values',{})
        lines.append(f'| {c["scenario"]} | {c["seed"]} | {c["placement"]}/{c["message_input"]}/{c["sender_mode"]}/{c["gate_init"]} | '+
            ' | '.join(fmt(values.get(k)) for k in ('best_validation_answer_token_nll','C1.G_all_test',
                'epoch3.validation.gate_tanh_alpha','epoch3.validation.gated_delta_over_hidden_l2'))+' |')
    lines+=['','逐epoch原验证NLL、完整format/numeric/mixed分解、同配置逐seed均值和SD及逐步gate轨迹来源，见matched_summary.json。任何机制均值同样要求3/3；缺失或无该类别token时为null。']
    lines+=['','组bootstrap保持旧group_id定义，2000次、seed2026；不能视为新的独立实验。通信预算是16×128 float32的逻辑载荷，每次8192字节；实际EOS调用数和padding开销从原始诊断读取，不冒称总FLOPs或物理传输相等。',
            '旧数值预测器只作参考。中性消息仍保留原允许提示上下文。旧test已被观察，本轮属于同冻结基准上的预先规定后续对照。']
    (suite/'matched_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/matched_budget_20261016.protocol.json')
    args=p.parse_args();print(json.dumps({'complete':summarize(args.suite,args.protocol)['n_complete'],'expected':36}))


if __name__=='__main__': main()

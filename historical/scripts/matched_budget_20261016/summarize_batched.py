"""Uniform V2 report using the immutable V1 raw-output audit/statistics.

Job directory names retain case_id; only standard contract run IDs are prefixed
batched_v2. The runner's epochs/steps are explicitly cumulative training-budget
fields. Reused training has zero NEW optimizer steps and is never counted twice.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from matched_budget_20261016 import summarize as original
from matched_budget_20261016.execute_suite_batched import (
    ROOT, case_id, check_reuse_pins, load, sha, trial_matrix, write)


def budget_view(actual,case,contract,record):
    """Validate new execution separately from the inherited 3-epoch budget."""
    reuse=case.get('reuse_training')
    if actual.get('generation_batch_size')!=4:
        raise ValueError('V2 uniformly requires generation batch4')
    if actual.get('execution_revision',{}).get('id')!='fixed_order_generation_batch4_v2':
        raise ValueError('Result is not the uniform V2 engineering implementation')
    if actual.get('actual_epochs')!=3 or actual.get('optimizer_steps')!=1548:
        raise ValueError('Cumulative/source training budget must remain 3 epochs and 1548 steps')
    if actual.get('training_budget_original')!={'epochs':3,'optimizer_steps':1548}:
        raise ValueError('Original training budget provenance missing')
    expected_level='independent_inference' if reuse else 'training'
    if (contract.get('execution_scope',{}).get('intended_level')!=expected_level
            or record.get('reproduction_level')!=expected_level or record.get('lifecycle_status')!='succeeded'):
        raise ValueError('Standard execution level does not distinguish reused inference from new training')
    if reuse:
        if (actual.get('new_optimizer_steps')!=0 or actual.get('training_performed_in_this_run') is not False
                or actual.get('training_reused_from_v1') is not True):
            raise ValueError('Reused reader must prove zero new optimizer steps and no new training')
        if reuse.get('source_training_epochs')!=3 or reuse.get('source_training_optimizer_steps')!=1548:
            raise ValueError('Original source training not complete')
    elif (actual.get('new_optimizer_steps')!=1548 or actual.get('training_performed_in_this_run') is not True
          or actual.get('training_reused_from_v1') is not False):
        raise ValueError('Fresh V2 training must prove 1548 new optimizer steps')
    return dict(evaluation_version='uniform_batch4_V2',new_training_count=0 if reuse else 1,
        new_training_epochs=0 if reuse else 3,new_optimizer_steps=0 if reuse else 1548,
        reused_training=bool(reuse),source_V1_job=reuse.get('source_job') if reuse else None,
        cumulative_or_source_training_budget={'epochs':3,'optimizer_steps':1548},
        budget_field_interpretation='raw actual_epochs/optimizer_steps describe the original or current trained model; new execution is counted separately')


def summarize(suite,protocol_path):
    protocol=load(protocol_path);views={};invalid={}
    for base_case in trial_matrix(protocol):
        job=suite/case_id(base_case);result_path=job/'artifacts/result.json'
        if not result_path.is_file(): continue
        actual=load(result_path)
        if actual.get('status')!='complete': continue
        try:
            case=load(job/'case.json');contract=load(job/'task-contract.json');record=load(job/'result-record.json')
            if {key:case[key] for key in base_case}!=base_case:
                raise ValueError('V2 case identity differs from fixed scientific matrix')
            if record.get('contract_sha256')!=sha(job/'task-contract.json'):
                raise ValueError('V2 terminal record does not bind exact frozen contract')
            view=budget_view(actual,case,contract,record)
            if case.get('reuse_training'):
                check_reuse_pins(case['reuse_training'])
                source=load(Path(case['reuse_training']['source_artifacts'])/'result.json')
                if (source.get('actual_epochs')!=3 or source.get('optimizer_steps')!=1548
                        or source.get('checkpoint_sha256')!=actual.get('checkpoint_sha256')):
                    raise ValueError('V2 inference does not use exact complete V1 selected checkpoint')
            views[case_id(base_case)]=view
        except (ValueError,KeyError,TypeError,OSError) as exc:
            invalid[str(result_path.resolve())]=str(exc)
    # The old function remains byte-for-byte unchanged. Only an invalid V2
    # provenance is made ineligible in the transient read view, never on disk.
    original_load=original.load
    def audited_load(path):
        value=original_load(path)
        reason=invalid.get(str(Path(path).resolve()))
        if reason and isinstance(value,dict):
            value=copy.deepcopy(value);value['status']='invalid_V2_training_or_engineering_provenance'
        return value
    with patch.object(original,'load',audited_load):
        result=original.summarize(suite,protocol_path)
    for entry in result['cases']:
        job=suite/entry['case_id']
        reason=invalid.get(str((job/'artifacts/result.json').resolve()))
        if reason: entry['V2_provenance_failure']=reason
        if entry['status']=='complete':
            view=views[entry['case_id']]
            entry['training_budget_epochs']=entry.pop('actual_epochs')
            entry['training_budget_optimizer_steps']=entry.pop('optimizer_steps')
            entry.update(view)
    complete=[entry for entry in result['cases'] if entry['status']=='complete']
    plan=load(suite/'reuse_plan.json') if (suite/'reuse_plan.json').exists() else {}
    result.pop('n_expected_training_trials',None)
    result.update(evaluation_version='uniform_batch4_V2',n_expected_configuration_evaluations=36,
        n_expected_reused_training_inference_cases=len(plan),n_expected_new_training_trials=36-len(plan),
        n_completed_reused_training_inference_cases=sum(entry['reused_training'] for entry in complete),
        n_completed_new_training_trials=sum(entry['new_training_count'] for entry in complete),
        interpretation='All 36 configurations use V2 batch4 evaluation. Original V1 evaluations remain separate, never selected or substituted. Source training counted once; reused inference has zero new updates.')
    write(suite/'matched_summary.json',result)
    path=suite/'matched_summary.md'
    text=path.read_text(encoding='utf-8').replace('完整终态训练：','统一 V2 评估配置：')
    text+='\n新完成训练：'+str(result['n_completed_new_training_trials'])+'；复用完整V1训练、仅新做V2推理：'+str(result['n_completed_reused_training_inference_cases'])+'。复用任务本次0轮、0个optimizer更新；来源模型仍具有原3轮/1548步预算。\n'
    path.write_text(text,encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/matched_budget_20261016.frozen.json')
    args=p.parse_args();result=summarize(args.suite,args.protocol)
    print(json.dumps({key:result[key] for key in ('n_complete','n_completed_new_training_trials','n_completed_reused_training_inference_cases')}))


if __name__=='__main__': main()

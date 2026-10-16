"""All 36 ORIGINAL V1 evaluations: fixed first two sources plus new remaining34.

No V2/V3 answers or source-score selection is supported. Temporary read-only
directory links let the frozen raw-text/parser/metric audit run unchanged.
Source PID/receipts remain source evidence and are never copied as new runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from matched_budget_20261016 import summarize as original
from matched_budget_20261016.execute_remaining_v1 import ROOT,case_id,load,sha,trial_matrix,write,verify_reused_evidence


def rewrite_paths(value,mapping):
    if isinstance(value,dict): return {key:rewrite_paths(child,mapping) for key,child in value.items()}
    if isinstance(value,list): return [rewrite_paths(child,mapping) for child in value]
    if isinstance(value,str):
        for alias,target in mapping.items():
            if value==alias or value.startswith(alias+'/'): return target+value[len(alias):]
    return value


def summarize(suite,source_suite,protocol_path):
    suite=Path(suite).resolve();source_suite=Path(source_suite).resolve()
    protocol=load(protocol_path);cases=trial_matrix(protocol)
    first={case_id(case) for case in cases[:2]}
    source_receipts=load(suite/'original_source_receipts.json')
    if set(source_receipts)!=first: raise ValueError('Only fixed first two original completed sources may fill source positions')
    verify_reused_evidence(source_receipts,{})
    for name,item in source_receipts.items():
        if Path(item['source_job']).resolve()!=(source_suite/name).resolve():
            raise ValueError('Original source lookup differs from frozen admission')
    with tempfile.TemporaryDirectory(prefix='original_V1_36_read_view_') as raw:
        view=Path(raw);mapping={}
        for case in cases:
            name=case_id(case);target=(source_suite/name if name in first else suite/name).resolve()
            if target.is_dir():
                alias=view/name;alias.symlink_to(target,target_is_directory=True);mapping[str(alias)]=str(target)
        mapping[str(view)]=str(suite)
        result=rewrite_paths(original.summarize(view,protocol_path),mapping)
        text=(view/'matched_summary.md').read_text(encoding='utf8')
    for entry in result['cases']:
        source=entry['case_id'] in first
        entry['execution_origin']='original_completed_V1_source' if source else 'new_original_V1_continuation'
        entry['physical_job_directory']=str((source_suite if source else suite)/entry['case_id'])
        entry['new_training_count']=int(entry['status']=='complete' and not source)
        entry['new_optimizer_steps']=0 if source else entry.get('optimizer_steps')
        if source:
            entry['source_training_epochs']=entry.pop('actual_epochs',None)
            entry['source_training_optimizer_steps']=entry.pop('optimizer_steps',None)
            entry['new_execution_performed']=False
            entry['source_evidence_pins']=source_receipts[entry['case_id']]['evidence_pins']
    result.update(evaluation_version='original_V1_unchanged_batch1_all36',original_source_suite=str(source_suite),
        n_original_completed_source_cases=sum(entry['status']=='complete' and entry['case_id'] in first for entry in result['cases']),
        n_new_completed_training_cases=sum(entry['new_training_count'] for entry in result['cases']),
        n_expected_new_training_cases=34,n_expected_original_source_cases=2,
        source_PID_policy='Original source receipts retain real source PID/PGID; no source process is represented as new execution',
        candidate_outputs_excluded=['V2','V3'],source_case_selection='fixed matrix first two; no score selection')
    write(suite/'matched_summary.json',result)
    text+='\n全部36项采用原V1逐行batch1解码。固定前2项读取原始完整训练和原始评估；其余34项来自本次原V1续跑，绝不读取V2/V3答案。\n'
    text+=f'原始已完成来源：{result["n_original_completed_source_cases"]}/2；新完成训练：{result["n_new_completed_training_cases"]}/34。原PID仅属于原来源，不伪造本次执行。\n'
    (suite/'matched_summary.md').write_text(text,encoding='utf8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True);p.add_argument('--source-suite',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/matched_budget_20261016.frozen.json')
    args=p.parse_args();result=summarize(args.suite,args.source_suite,args.protocol)
    print(json.dumps({key:result[key] for key in ('n_complete','n_original_completed_source_cases','n_new_completed_training_cases')}))


if __name__=='__main__':main()

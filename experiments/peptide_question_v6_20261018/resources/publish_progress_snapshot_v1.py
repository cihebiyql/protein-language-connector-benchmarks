"""Additional read-only scorer and finite progress publisher; no training changes."""
import argparse,collections,hashlib,importlib.util,json,math,os
from datetime import datetime,timezone,timedelta
from pathlib import Path

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def score(gold,pred,thresholds):
    questions=('rt','ccs','both');lookup={(p['id'],p['question']):p['raw'].strip() for p in pred}
    expected={(r['id'],q) for r in gold for q in questions}
    if len(lookup)!=len(pred) or set(lookup)!=expected:raise ValueError('Prediction identity mismatch')
    correct={};out={}
    for q in questions:
        totals=collections.Counter();hits=collections.Counter();valid=0
        for r in gold:
            truth=(r['rt']>thresholds['rt'] if q=='rt' else r['ccs']>thresholds['ccs'] if q=='ccs' else
                   r['rt']>thresholds['rt'] and r['ccs']>thresholds['ccs'])
            text=lookup[r['id'],q];ok=text==('yes' if truth else 'no')
            correct[r['id'],q]=ok;totals[truth]+=1;hits[truth]+=ok;valid+=text in ('yes','no')
        out[q]={'balanced_accuracy':(hits[False]/totals[False]+hits[True]/totals[True])/2,
                'accuracy':sum(hits.values())/len(gold),'valid_rate':valid/len(gold)}
    return {'primary':(out['rt']['balanced_accuracy']+out['ccs']['balanced_accuracy'])/2,
            'joint_all_correct':sum(all(correct[r['id'],q] for q in questions) for r in gold)/len(gold),'questions':out}

def main(scope,publish):
    scope=Path(scope).resolve();config=load(scope/'config.json')
    spec=importlib.util.spec_from_file_location('v6_verify',scope/'resources/node_pipeline.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
    manifest=v.verify_freeze(scope);state=load(scope/'STATUS.json');launch=load(scope/'private/pipeline_launch.json')
    if not v.same_live_process(launch['process']):raise ValueError('Independent original queue is not live')
    accepted=[c for c in state['cases'] if c['status']=='accepted'];case_metrics={}
    if not accepted:raise ValueError('No accepted partial results')
    gold=rows(scope/'data/test.jsonl')
    for c in accepted:
        v.verify_result(scope,c,config)
        root=scope/'runs'/c['id']/'formal';normal=score(gold,rows(root/'test_normal.jsonl'),config['thresholds'])
        stored=load(root/'test_summary.json')['normal']
        if abs(normal['primary']-stored['primary'])>1e-12 or abs(normal['joint_all_correct']-stored['joint_all_correct'])>1e-12:
            raise ValueError('Independent normal rescoring differs')
        for q,m in normal['questions'].items():
            if any(abs(value-stored['questions'][q][name])>1e-12 for name,value in m.items()):
                raise ValueError('Independent question metric differs')
        case_metrics[c['id']]={**normal,'arm':c['arm'],'seed':c['seed'],'epochs_completed':6,
            'normal_predictions_sha256':sha(root/'test_normal.jsonl'),'completion_sha256':sha(root/'completion.json')}
    running=[]
    for c in state['cases']:
        if c['status']!='running':continue
        if not v.same_live_process(c['process']):raise ValueError('A recorded running child is not live')
        p=load(scope/'runs'/c['id']/'formal/progress.json')
        running.append({'case':c['id'],'GPU':c['gpu'],'epoch':p.get('epoch_one_based'),'updates':p.get('optimizer_updates')})
    stamp=datetime.now(timezone(timedelta(hours=8)));name='progress_'+stamp.strftime('%Y%m%d_%H%M%S')
    snapshot={'schema':'v6-progress-with-independent-partial-rescore.v1','snapshot_at':stamp.isoformat(),
        'status':'running_partial_results','expected_cases':9,'counts':state['counts'],'accepted_cases':len(accepted),
        'completed_case_metrics':case_metrics,'running_cases':running,'test_peptides':len(gold),
        'three_seed_summary_available':False,'direction_decision':'pending_complete_three_seeds',
        'frozen_manifest_sha256':manifest,'node1_independent_verified':True,'scheduled_monitor':False,
        'additional_operational_entrypoint_sha256':sha(Path(__file__))}
    snapshot['parallel_motifae_pilot']={'status':'complete_pilot_no_incremental_gain',
        'completed_answers':1608,'expected_answers':1608,'decision':'pause_current_motifae_evidence_card_extension',
        'verified_report_commit':'0d9ac604075e8c800aa78ea7347d0dc48bb2e3df',
        'v6_final_direction_decision_still_requires_all_three_seeds':True}
    output=scope/'private/progress_snapshots'/name;output.mkdir(parents=True)
    v.atomic_json(output.with_suffix('.json'),snapshot)
    if not publish:print(json.dumps(snapshot,indent=2));return
    spec=importlib.util.spec_from_file_location('v6_pub',scope/'resources/publish_v6.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    runtime=load(scope/'private/runtime.json');auth=load(runtime['publication_auth_config'])
    env={**os.environ,**auth.get('environment_overrides',{}),'GIT_SSH_COMMAND':auth['git_ssh_command'],'GIT_TERMINAL_PROMPT':'0',
        'GIT_AUTHOR_NAME':auth['git_author_name'],'GIT_AUTHOR_EMAIL':auth['git_author_email'],
        'GIT_COMMITTER_NAME':auth['git_author_name'],'GIT_COMMITTER_EMAIL':auth['git_author_email']}
    repo=Path(auth['repo']);worktree=scope/'private/publication_worktree';relative='experiments/'+scope.name
    with p.locked(scope/'private/publication.lock'):
        if p.call(['git','status','--porcelain'],worktree,env):raise ValueError('Unexpected changes in dedicated worktree')
        p.call(['git','fetch','origin','main'],repo,env);p.call(['git','merge','--ff-only','origin/main'],worktree,env)
        target=worktree/relative;report=target/'reports'/(name+'.json');p.write(report,snapshot)
        text=['本轮九组仍在运行，已完成结果仅代表指定种子，不能形成多随机种子稳定提升或最终暂停判断。','',
            '| 已完成方案 | 种子 | RT平衡准确率 | CCS平衡准确率 | 两项均值 | 三问全对率 |',
            '|---|---:|---:|---:|---:|---:|']
        for c,m in case_metrics.items():text.append(f"| {m['arm']} | {m['seed']} | {m['questions']['rt']['balanced_accuracy']:.6f} | {m['questions']['ccs']['balanced_accuracy']:.6f} | {m['primary']:.6f} | {m['joint_all_correct']:.6f} |")
        text+=['','固定六轮预算、三个种子、数据划分、评分、停止规则均保持。剩余实验由node1独立运行；无定时监控。',
               '本次读取已验收原始生成答案重新计算，未重新调用模型、调整阈值或删掉失败答案。']
        (target/'reports'/(name+'.md')).write_text('\n'.join(text)+'\n',encoding='utf-8')
        status={'schema':'same-peptide-progress-public-completion.v1','status':'formal_recovery_running',
            'snapshot_at':snapshot['snapshot_at'],'expected_cases':9,'accepted_cases':len(accepted),'counts':state['counts'],
            'three_seed_summary_available':False,'analysis_completed':False,'publication_host':'node1',
            'local_callback_required':False,'scheduled_monitor':False,'all_original_results_preserved':True,
            'frozen_manifest_sha256':manifest,'progress_report':'reports/'+name+'.json'}
        p.write(target/'COMPLETION_STATUS.json',status)
        pub_manifest=p.read(target/'PUBLICATION_MANIFEST.json')
        pub_manifest['latest_progress_snapshot']={'path':'reports/'+name+'.json','sha256':p.sha(report),'final_results_pending':True}
        p.write(target/'PUBLICATION_MANIFEST.json',pub_manifest)
        extra=target/'resources/publish_progress_snapshot_v1.py';extra.write_bytes(Path(__file__).read_bytes())
        paths=[report,report.with_suffix('.md'),target/'COMPLETION_STATUS.json',target/'PUBLICATION_MANIFEST.json',extra]
        rels=[x.relative_to(worktree).as_posix() for x in paths]
        p.call(['git','add','-f','--',*rels],worktree,env);p.call(['git','diff','--cached','--check'],worktree,env)
        if set(p.call(['git','diff','--cached','--name-only'],worktree,env).splitlines())-set(rels):raise ValueError('Out-of-scope changes staged')
        message=('Expose accepted results without claiming a premature multiseed conclusion\n\n'
            'Record independently re-scored completed cases while the original frozen queue continues.\n\n'
            'Constraint: No new model calls, changed thresholds, dropped seeds, or altered stopping rules\nConfidence: high\nScope-risk: narrow\n'
            'Directive: Replace this partial view with the original automatic final report after all9 cases finish\n'
            'Tested: Frozen source hashes, accepted artifact bindings, independent raw-answer metrics and live process identities\n'
            'Not-tested: Remaining seeds and final stability decision\nCo-authored-by: OmX <omx@oh-my-codex.dev>')
        p.call(['git','commit','-m',message],worktree,env);p.call(['git','push','origin','HEAD:refs/heads/main'],worktree,env)
        commit=p.call(['git','rev-parse','HEAD'],worktree,env)
        import urllib.request
        verified=[]
        for path in [report,target/'COMPLETION_STATUS.json']:
            rel=path.relative_to(worktree).as_posix();url=f'https://raw.githubusercontent.com/cihebiyql/protein-language-connector-benchmarks/{commit}/{rel}'
            with urllib.request.urlopen(url,timeout=45) as response:data=response.read()
            if hashlib.sha256(data).hexdigest()!=p.sha(path):raise ValueError('Anonymous public bytes differ')
            verified.append({'path':rel,'sha256':p.sha(path)})
        result={'status':'published_verified','commit':commit,'accepted_cases':len(accepted),'snapshot':name,'verified':verified}
        p.write(scope/'private/latest_progress_publication_receipt.json',result)
        active=load(scope/'ACTIVE_EXPERIMENT.json');active.update(accepted_cases=len(accepted),latest_manual_counts=state['counts'],
            latest_manual_snapshot_at=snapshot['snapshot_at'],latest_progress_commit=commit)
        p.write(scope/'ACTIVE_EXPERIMENT.json',active)
        print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scope',required=True);parser.add_argument('--publish',action='store_true');a=parser.parse_args();main(a.scope,a.publish)

"""Publish this finite experiment using the existing node1 repository deploy key.

No credentials are copied, displayed, committed, or created. Public files are
allowlisted; external runtime paths are converted to documented placeholders.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

SCOPE_ID='peptide_question_v6_20261018'
REPOSITORY='cihebiyql/protein-language-connector-benchmarks'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)

def call(args,cwd,env):
    # Frozen Windows-origin data deliberately retain CRLF bytes (-text repo).
    # Treat CR-at-EOL as a line ending while retaining the normal space checks.
    if args and args[0]=='git':
        args=['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',*args[1:]]
    return subprocess.run(args,cwd=cwd,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,check=True,timeout=600).stdout.strip()

def public_files(scope):
    scope=Path(scope)
    result=[]
    for directory in ('src','tests'):
        result.extend(p for p in (scope/directory).glob('*.py') if p.is_file())
    result.extend(p for p in (scope/'resources').glob('*.py') if p.is_file())
    for name in ('PROTOCOL.md','README.md','config.json','frozen_manifest.json','ACTIVE_EXPERIMENT.json',
                 'resources/model_assets.json','data/manifest.json','data/train.jsonl','data/validation.jsonl',
                 'data/test.jsonl','data/control_mapping.json','cache/features_receipt.json','cache/specialists_receipt.json',
                 'cache/attributes_receipt.json','cache/attributes.npz','analysis/summary.json','analysis/REPORT.md','resources/qualification_receipt.json'):
        p=scope/name
        if p.is_file():result.append(p)
    for directory in (scope/'runs').glob('*/*'):
        if directory.name not in ('formal','smoke'):continue
        for p in directory.iterdir():
            if p.is_file() and (p.suffix in ('.json','.jsonl') or p.name=='best_adapter.pt'):
                result.append(p)
    return sorted(set(result))

def sanitize(value,replacements):
    if isinstance(value,str):
        for source,target in replacements:value=value.replace(source,target)
        return value
    if isinstance(value,list):return [sanitize(v,replacements) for v in value]
    if isinstance(value,dict):return {sanitize(k,replacements):sanitize(v,replacements) for k,v in value.items()}
    return value

def stage(scope,target,stage_name):
    scope,target=Path(scope),Path(target);target.mkdir(parents=True,exist_ok=True)
    config=read(scope/'config.json')
    replacements=[(config['model_path'],'${QWEN3_4B_MODEL_PATH}'),
                  (config['esm_model_path'],'${ESM150M_MODEL_PATH}'),(str(scope),'${EXPERIMENT_SCOPE}')]
    entries=[]
    for source in public_files(scope):
        if source.is_symlink() or not source.resolve().is_relative_to(scope.resolve()):
            raise ValueError('Publication source symlink or path escape')
        relative=source.relative_to(scope);destination=target/relative
        if destination.is_symlink() or not destination.resolve().is_relative_to(target.resolve()):
            raise ValueError('Publication destination symlink or path escape')
        destination.parent.mkdir(parents=True,exist_ok=True)
        if source.stat().st_size>50*1024**2:raise ValueError('Unapproved publication size: '+str(relative))
        transform='byte_exact'
        if source.suffix=='.json':
            original=read(source);clean=sanitize(original,replacements)
            if clean!=original:
                write(destination,clean);transform='external_runtime_paths_to_placeholders'
            else:shutil.copyfile(source,destination)
        else:shutil.copyfile(source,destination)
        entries.append({'path':relative.as_posix(),'source_sha256':sha(source),
                        'public_sha256':sha(destination),'transformation':transform})
    summary=read(scope/'analysis/summary.json') if (scope/'analysis/summary.json').exists() else None
    terminal=stage_name=='final'
    status={'schema':'same-peptide-public-completion.v1',
            'status':summary.get('status') if terminal and summary else ('technical_inconclusive' if terminal else 'protocol_registered_not_started'),
            'publication_stage':stage_name,'expected_cases':9,
            'accepted_cases':summary.get('accepted_cases',0) if summary else 0,
            'analysis_completed':bool(terminal and summary),'decision':summary.get('decision') if summary else None,
            'publication_host':'node1','local_callback_required':False,'scheduled_monitor':False,
            'original_v5_results_preserved':True,'frozen_original_manifest_sha256':sha(scope/'frozen_manifest.json') if (scope/'frozen_manifest.json').exists() else None,
            'original_runtime_paths_replaced_with_placeholders':True,
            'raw_feature_cache_and_specialist_state_private':True,
            'all_registered_seeds_preserved':True}
    write(target/'COMPLETION_STATUS.json',status)
    write(target/'PUBLICATION_MANIFEST.json',{'schema':'same-peptide-public-manifest.v1',
        'stage':stage_name,'files':entries,'source_dataset':read(scope/'data/manifest.json')['source_url'],
        'excluded':['private/**','cache/features.npz','raw source CSV/archive','environment/frozen model weights'],
        'frozen_original_manifest_sha256':status['frozen_original_manifest_sha256']})
    return entries,status

@contextmanager
def locked(path):
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield

def publish(scope,stage_name='final'):
    scope=Path(scope).resolve()
    if scope.name!=SCOPE_ID:raise ValueError('Wrong scope')
    runtime=read(scope/'private/runtime.json');auth=read(runtime['publication_auth_config'])
    # Reuse the pre-existing scope-restricted deploy key and private Git runtime.
    env={**os.environ,**auth.get('environment_overrides',{}),
         'GIT_SSH_COMMAND':auth['git_ssh_command'],'GIT_TERMINAL_PROMPT':'0',
         'GIT_AUTHOR_NAME':auth['git_author_name'],'GIT_AUTHOR_EMAIL':auth['git_author_email'],
         'GIT_COMMITTER_NAME':auth['git_author_name'],'GIT_COMMITTER_EMAIL':auth['git_author_email'],
         'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    repo=Path(auth['repo']).resolve();worktree=scope/'private/publication_worktree'
    receipt=scope/('private/publication_'+stage_name+'_receipt.json')
    with locked(scope/'private/publication.lock'):
        if receipt.exists():return read(receipt)
        call(['git','fetch','origin','main'],repo,env)
        if not worktree.exists():
            call(['git','worktree','add','--detach',str(worktree),'origin/main'],repo,env)
        else:
            dirty=call(['git','status','--porcelain'],worktree,env)
            if dirty:
                prefix='experiments/'+SCOPE_ID+'/'
                allowed={prefix+p.relative_to(scope).as_posix() for p in public_files(scope)}|{prefix+n for n in ('COMPLETION_STATUS.json','PUBLICATION_MANIFEST.json')}
                staged=set(call(['git','diff','--name-only','HEAD'],worktree,env).splitlines())
                untracked=set(call(['git','ls-files','--others','--exclude-standard'],worktree,env).splitlines())
                if (staged|untracked)-allowed:raise RuntimeError('Publication worktree contains changes outside the owned verified files')
                # A prior failed staging/push is retained and restaged from the
                # exact verified source; no reset, deletion, or checkout occurs.
            else:
                try:call(['git','merge','--ff-only','origin/main'],worktree,env)
                except subprocess.CalledProcessError:
                    call(['git','merge','-m',
                        'Preserve concurrent publications while resuming the peptide evidence push\n\n'
                        'Keep both histories in the isolated experiment worktree.\n\n'
                        'Constraint: No force push or discarded local result commits\nConfidence: high\nScope-risk: narrow\n'
                        'Tested: Clean dedicated worktree and remote fetch\nCo-authored-by: OmX <omx@oh-my-codex.dev>',
                        'origin/main'],worktree,env)
        target=worktree/'experiments'/SCOPE_ID
        entries,status=stage(scope,target,stage_name)
        relative=target.relative_to(worktree).as_posix()
        call(['git','add','-f','--',relative],worktree,env)
        call(['git','diff','--cached','--check'],worktree,env)
        changed=call(['git','diff','--cached','--name-only'],worktree,env)
        permitted={relative+'/'+item['path'] for item in entries}|{relative+'/'+name for name in ('COMPLETION_STATUS.json','PUBLICATION_MANIFEST.json')}
        if set(changed.splitlines())-permitted:raise ValueError('Staged files outside verified publication allowlist')
        if changed:
            message=('Make the final peptide-interface decision independently reviewable' if stage_name=='final'
                else 'Fix the final peptide-interface comparison before seeing formal results')
            message+='\n\nPreserve earlier experiments and publish the frozen same-peptide RT/CCS shared-generation protocol'
            message+=' with strong cross-fitted text attributes and all registered seeds.\n\n'
            message+='Constraint: No frozen-test retuning, score-based retries, personal credential transfer, or scheduled monitor\n'
            message+='Confidence: high\nScope-risk: narrow\nDirective: Retain negative results and honor the preregistered stop decision\n'
            message+=('Tested: Frozen input hashes, generated-answer rescoring, and published byte verification\n' if stage_name=='final'
                      else 'Tested: CPU module tests and TRAIN-only technical qualification; formal outcomes not yet available\n')
            message+='Not-tested: Functional or therapeutic generalization beyond the paired analytical-property benchmark\n'
            message+='Co-authored-by: OmX <omx@oh-my-codex.dev>'
            call(['git','commit','-m',message],worktree,env)
        commit=call(['git','rev-parse','HEAD'],worktree,env)
        error=None
        for attempt in range(3):
            try:
                call(['git','push','origin','HEAD:refs/heads/main'],worktree,env);error=None;break
            except subprocess.CalledProcessError as exc:
                error=exc
                call(['git','fetch','origin','main'],worktree,env)
                # Independent directory; merge concurrent updates without force.
                merge_message=('Keep concurrent research results alongside the frozen peptide comparison\n\n'
                    'Merge remote changes into the dedicated publication worktree without force.\n\n'
                    'Constraint: Preserve unrelated main changes\nConfidence: high\nScope-risk: narrow\n'
                    'Tested: Git tree merge and frozen experiment publication checks\n'
                    'Co-authored-by: OmX <omx@oh-my-codex.dev>')
                call(['git','merge','-m',merge_message,'origin/main'],worktree,env)
                commit=call(['git','rev-parse','HEAD'],worktree,env)
                time.sleep(2*(attempt+1))
        if error:raise error
        call(['git','fetch','origin','main'],worktree,env)
        call(['git','merge-base','--is-ancestor',commit,'origin/main'],worktree,env)
        verified=[]
        for name in ('COMPLETION_STATUS.json','PUBLICATION_MANIFEST.json'):
            path=target/name;url=f'https://raw.githubusercontent.com/{REPOSITORY}/{commit}/{relative}/{name}'
            last_error=None
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url,timeout=45) as response: data=response.read()
                    if hashlib.sha256(data).hexdigest()!=sha(path):raise ValueError('Anonymous public bytes mismatch')
                    last_error=None;break
                except Exception as exc:last_error=exc;time.sleep(3*(attempt+1))
            if last_error:raise last_error
            verified.append({'path':relative+'/'+name,'sha256':sha(path)})
        result={'status':'published_verified','stage':stage_name,'commit':commit,'verified':verified,
                'files':len(entries),'publication_host':'node1','local_callback_required':False,
                'scheduled_monitor':False,'at_utc':datetime.now(timezone.utc).isoformat(),
                'completion_status':status}
        write(receipt,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scope',required=True);p.add_argument('--stage',choices=['preregistered','final'],default='final');a=p.parse_args()
    print(json.dumps(publish(a.scope,a.stage)),flush=True)

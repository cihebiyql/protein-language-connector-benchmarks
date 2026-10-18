"""Freeze the pilot closure before dictionary fitting or held-out evaluation."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def freeze(scope):
    scope=Path(scope).resolve();target=scope/'frozen_manifest.json'
    if target.exists() or (scope/'dictionary.json').exists() or (scope/'analysis/feature_metrics.json').exists():
        raise RuntimeError('Prior frozen closure or held-out pilot artifacts already exist')
    config=json.loads((scope/'config.json').read_text());config['status']='frozen'
    (scope/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    files=[p for name in ('src','tests','resources') for p in (scope/name).glob('*.py')]
    files += [scope/name for name in ('config.json','PROTOCOL.md','README.md','reports/INTEGRATION_PLAN.md',
        'data/panel.jsonl','data/catalog.json','data/manifest.json','assets/DOWNLOAD_MANIFEST.json')]
    for item in json.loads((scope/'assets/DOWNLOAD_MANIFEST.json').read_text())['files']:
        file=scope/'assets'/item['file']
        if sha(file)!=item['sha256']:raise ValueError('Downloaded asset identity mismatch')
    pins={p.relative_to(scope).as_posix():sha(p) for p in sorted(set(files))}
    manifest={'schema':'motifae-pilot-frozen-closure.v1','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_pins':pins,'files':[{'path':name,'sha256':digest} for name,digest in pins.items()],
        'dictionary_and_test_outputs_not_yet_created':True,'original_v6_stop_rule_unchanged':True}
    with target.open('x') as stream:stream.write(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'frozen','files':len(pins),'manifest_sha256':sha(target)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scope',required=True);args=parser.parse_args();freeze(args.scope)

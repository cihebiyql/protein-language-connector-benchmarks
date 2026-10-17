"""One manual startup snapshot; no scheduler, model execution or source mutation."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tarfile

SCOPE=Path(__file__).resolve().parents[1]

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    queue=SCOPE/'runs/formal_v1'
    state=read(queue/'queue_state.json')
    snapshot={'utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':state['plan_sha256'],
        'counts':state['counts'],'expected_cases':45,'gpu_devices':[3,4,5],'scheduled_monitor':False,'cases':[]}
    files={SCOPE/'configs/formal_formal_v1.json',SCOPE/'configs/queue_formal_v1.json',
        SCOPE/'docs/freeze_formal_v1.json',SCOPE/'resources/launch_formal_v1.json'}
    for case_id,item in state['cases'].items():
        row={'case_id':case_id,'queue_status':item['status']}
        if 'attempt' in item:
            attempt=Path(item['attempt'])
            row['qualification_accepted']=(attempt/'qualification_acceptance.json').is_file()
            progress=attempt/'artifacts/progress.jsonl'
            if progress.is_file():
                for line in reversed(progress.read_text().splitlines()):
                    try:row['latest_formal_progress']=json.loads(line);break
                    except json.JSONDecodeError:continue
            if item['status']=='accepted':
                files.update(p for p in attempt.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.pt','.csv','.log'))
        snapshot['cases'].append(row)
    destination=SCOPE/'resources/startup_snapshot_v1.json'
    with destination.open('x',encoding='utf-8') as stream:json.dump(snapshot,stream,indent=2);stream.write('\n')
    files.add(destination)
    archive=SCOPE/'resources/startup_snapshot_v1.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for p in sorted(files):tar.add(p,arcname=p.relative_to(SCOPE).as_posix())
    print(json.dumps({'utc':snapshot['utc'],'counts':state['counts'],'archive':str(archive),'sha256':sha(archive),
        'active':[r for r in snapshot['cases'] if r['queue_status'] not in ('pending','resource_wait')]}),flush=True)

if __name__=='__main__':main()

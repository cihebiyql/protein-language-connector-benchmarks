"""Recheck saved assets and pilot receipts without importing model runtimes."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

def main():
    checked = 0
    items = read('sources/literature_data_manifest.json')['files'] + read('papers/methods/manifest.json') + read('sources/model_manifest.json')
    for item in items:
        if 'sha256' not in item:
            continue
        path = ROOT / item['path'].replace('\\','/')
        assert path.exists(), path
        assert path.stat().st_size == item['bytes'], path
        assert sha(path) == item['sha256'], path
        if item.get('official_sha256'):
            assert item['sha256'] == item['official_sha256']
        checked += 1
    repos = read('sources/repository_manifest.json')['repositories']
    for repo in repos:
        path = ROOT/'repos'/repo['name']
        # Use the recorded directory because some repository display names differ from checkout names.
        if not path.exists():
            path = Path(repo['local_path'])
        commit = subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'], text=True).strip()
        assert commit == repo['git_commit']
        assert not subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip()
    b1 = read('results/b1_pilot.json')
    assert b1['script_sha256'] == sha(ROOT/'scripts/run_b1_pilot.py')
    assert b1['input_sha256'] == sha(ROOT/'data/processed/benz_samples.jsonl')
    for split in ('train','validation','test'):
        rows = [json.loads(line) for line in (ROOT/f'results/b1_{split}_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        recalls, reciprocals = [], []
        for row in rows:
            ranking, truth = row['ranked_target_ids'], set(row['known_positive_target_ids'])
            assert len(ranking) == len(set(ranking)) == 34
            recalls.append(len(set(ranking[:3]) & truth)/len(truth))
            reciprocals.append(1/next(i for i,target in enumerate(ranking,1) if target in truth))
        expected = b1['scores'][split]
        assert len(rows) == expected['n']
        assert abs(sum(recalls)/len(rows)-expected['known_positive_recall_at_3']) < 1e-12
        assert abs(sum(reciprocals)/len(rows)-expected['mrr']) < 1e-12
    smoke = read('results/bridge_smoke_64aa.json')
    assert smoke['script_sha256'] == sha(ROOT/'scripts/smoke_bridge.py')
    assert smoke['bridge_sha256'] == sha(ROOT/'scripts/bridge.py')
    assert len(smoke['interfaces']) == 6
    assert all(row['adapter_weight_changed'] and row['base_models_frozen'] for row in smoke['interfaces'])
    b2 = read('results/b2_smoke.json')
    assert b2['script_sha256'] == sha(ROOT/'scripts/smoke_b2.py')
    assert b2['predictor_predictions_sha256'] == sha(ROOT/'results/b1_test_predictions.jsonl')
    assert len(b2['examples']) == 3
    assert all(row['json_parse_success'] and row['ranking_preserved'] for row in b2['examples'])
    own_paths = sorted(set(list((ROOT/'scripts').glob('*.py')) + list((ROOT/'tests').glob('*.py')) +
                           list((ROOT/'reports').glob('*.md')) + [ROOT/'README.md']))
    receipt = {'verified_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
               'asset_hash_checks':checked, 'clean_pinned_repositories':len(repos),
               'b1_metrics_recomputed':True, 'model_source_fingerprints_match':True,
               'bridge_interfaces_verified':6, 'b2_examples_verified':3,
               'files':{str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for path in own_paths}}
    (ROOT/'results/artifact_verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))

if __name__=='__main__':
    main()

"""Fetch pinned official model assets without changing the shared HF cache."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'facebook/esm2_t30_150M_UR50D': 'a695f6045e2e32885fa60af20c13cb35398ce30c',
    'Qwen/Qwen2.5-1.5B-Instruct': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
}

def fetch(repo, revision, name, official_sha256):
    dest = ROOT / 'models' / repo.replace('/', '--') / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
    if not dest.exists():
        temp = dest.with_name(dest.name + '.partial')
        for attempt in range(3):
            try:
                digest = hashlib.sha256()
                with requests.get(url, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    expected = int(response.headers.get('Content-Length', 0))
                    size = 0
                    with temp.open('wb') as handle:
                        for block in response.iter_content(4 * 1024 * 1024):
                            if block:
                                handle.write(block)
                                digest.update(block)
                                size += len(block)
                    if expected and size != expected:
                        raise ValueError(f'{size} bytes != {expected}')
                temp.replace(dest)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
    digest = hashlib.sha256()
    with dest.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    actual_sha256 = digest.hexdigest()
    if official_sha256 and actual_sha256 != official_sha256:
        raise ValueError(f'Official SHA256 mismatch: {repo}/{name}')
    print(f'OK {repo}/{name} {dest.stat().st_size}', flush=True)
    return {'repo': repo, 'revision': revision, 'file': name, 'path': str(dest.relative_to(ROOT)),
            'url': url, 'bytes': dest.stat().st_size, 'sha256': actual_sha256,
            'official_sha256': official_sha256, 'official_hash_verified': bool(official_sha256 and actual_sha256 == official_sha256)}

def main():
    jobs = []
    for repo, revision in MODELS.items():
        response = requests.get(f'https://huggingface.co/api/models/{repo}/revision/{revision}',
                                params={'blobs': 'true'}, timeout=30)
        response.raise_for_status()
        metadata = response.json()
        assert metadata['sha'] == revision
        (ROOT / 'sources').mkdir(exist_ok=True)
        (ROOT / 'sources' / (repo.replace('/', '--') + '.json')).write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        for item in metadata['siblings']:
            name = item['rfilename']
            if name.endswith(('.json', '.txt', '.md', '.safetensors')) or name == 'LICENSE':
                jobs.append((repo, revision, name, item.get('lfs', {}).get('sha256')))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda job: fetch(*job), jobs))
    (ROOT / 'sources/model_manifest.json').write_text(json.dumps(receipts, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()

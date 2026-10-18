"""Bounded official HTTP range recovery with final publisher MD5 validation."""
import argparse,hashlib,json,os,time,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

URL='https://zenodo.org/records/20547228/files/MotifAE_k3_step_80000.pt?download=1'
SIZE=419601512;MD5='cabb07008f26705f07e18344607a2d51';CHUNK=8*1024*1024
def fetch(directory,start,end):
    p=directory/f'{start:012d}-{end:012d}.bin'
    if p.exists() and p.stat().st_size==end-start+1:return p
    for attempt in range(3):
        try:
            request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}','User-Agent':'MotifAE-research/1.0'})
            with urllib.request.urlopen(request,timeout=60) as response:
                if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{SIZE}':
                    raise ValueError('Unexpected official byte range response')
                data=response.read(end-start+2)
            if len(data)!=end-start+1:raise ValueError('Incomplete range')
            temporary=p.with_suffix('.tmp');temporary.write_bytes(data);temporary.replace(p)
            return p
        except Exception:
            if attempt==2:raise
            time.sleep(2*(attempt+1))

def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',required=True);a=p.parse_args()
    root=Path(a.scope)/'assets';out=root/'MotifAE_k3_step_80000.pt'
    if out.exists():
        if hashlib.md5(out.read_bytes()).hexdigest()!=MD5:raise ValueError('Existing model mismatch')
        return
    directory=root/'checkpoint_ranges';directory.mkdir(exist_ok=True)
    ranges=[(start,min(SIZE-1,start+CHUNK-1)) for start in range(0,SIZE,CHUNK)]
    prefix=root/'MotifAE_k3_step_80000.pt.part'
    if prefix.exists():
        complete=prefix.stat().st_size//CHUNK
        with prefix.open('rb') as stream:
            for i in range(complete):
                start,end=ranges[i];part=directory/f'{start:012d}-{end:012d}.bin'
                data=stream.read(end-start+1)
                if not part.exists():part.write_bytes(data)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(fetch,directory,start,end):(start,end) for start,end in ranges}
        for i,future in enumerate(as_completed(futures),1):
            future.result();print(json.dumps({'completed_ranges':i,'total_ranges':len(ranges)}),flush=True)
    assembled=root/'MotifAE_k3_step_80000.pt.assembled.part'
    hashes={k:hashlib.new(k) for k in ('md5','sha256')}
    with assembled.open('wb') as stream:
        for start,end in ranges:
            data=(directory/f'{start:012d}-{end:012d}.bin').read_bytes();stream.write(data)
            for digest in hashes.values():digest.update(data)
    if assembled.stat().st_size!=SIZE or hashes['md5'].hexdigest()!=MD5:raise ValueError('Assembled official checksum mismatch')
    assembled.replace(out)
    receipt={'source':URL,'bytes':SIZE,'md5':MD5,'sha256':hashes['sha256'].hexdigest(),'official_checksum_passed':True,'range_count':len(ranges)}
    (root/'CHECKPOINT_DOWNLOAD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()

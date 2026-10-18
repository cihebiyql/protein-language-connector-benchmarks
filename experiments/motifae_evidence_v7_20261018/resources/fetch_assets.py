"""Download fixed official inference assets; never execute downloaded code."""
import argparse,csv,hashlib,json,os,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REV='8dd6a7588d30726fef98c789dac3042370e829c5'
BASE=f'https://raw.githubusercontent.com/CHAOHOU-97/MotifAE/{REV}/'
SMALL={
 'MotifAE.pdf':'https://shenlab.github.io/pdfs/papers/Hou-et-al-2026-MotifAE.pdf',
 'README.author.md':BASE+'README.md','LICENSE.author':BASE+'LICENSE',
 'dictionary.author.py':BASE+'model/dictionary.py','config.author.py':BASE+'model/config.py',
 'ELM_motif.author.ipynb':BASE+'notebook/ELM_motif.ipynb',
 'reference_embedding.npz':BASE+'data/embed/MOD_PIKK_1_A0A024R7R0_0_719.npz',
 'reference_annotations.csv':BASE+'data/elm_instances_seq_label_iupred_max1022.csv',
 'zenodo_metadata.json':'https://zenodo.org/api/records/20547228'}

def download(directory,name,url):
    destination=directory/name
    if not destination.exists():
        temporary=destination.with_suffix(destination.suffix+'.part')
        request=urllib.request.Request(url,headers={'User-Agent':'MotifAE-bounded-research-pilot/1.0'})
        with urllib.request.urlopen(request,timeout=120) as response,temporary.open('wb') as stream:
            while True:
                block=response.read(1024*1024)
                if not block:break
                stream.write(block)
        temporary.replace(destination)
    hashes={algorithm:hashlib.new(algorithm) for algorithm in ('sha256','md5')}
    with destination.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            for digest in hashes.values():digest.update(block)
    result={'file':name,'url':url,'bytes':destination.stat().st_size,**{k:v.hexdigest() for k,v in hashes.items()}}
    if name=='MotifAE_k3_step_80000.pt' and (result['bytes']!=419601512 or result['md5']!='cabb07008f26705f07e18344607a2d51'):
        raise ValueError('Official k3 download checksum differs')
    print(json.dumps(result),flush=True);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',required=True);a=p.parse_args()
    scope=Path(a.scope);directory=scope/'assets';directory.mkdir(parents=True,exist_ok=True)
    items={**SMALL,'MotifAE_k3_step_80000.pt':'https://zenodo.org/api/records/20547228/files/MotifAE_k3_step_80000.pt/content'}
    with ThreadPoolExecutor(max_workers=3) as pool:
        records=list(pool.map(lambda x:download(directory,*x),items.items()))
    with (directory/'reference_annotations.csv').open(encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    row=next(r for r in rows if r.get('id')=='MOD_PIKK_1_A0A024R7R0_0_719')
    candidates=[v for k,v in row.items() if isinstance(v,str) and len(v)==719 and v.isalpha() and v.upper()==v]
    if len(candidates)!=1:raise ValueError('Reference full sequence schema ambiguous')
    (directory/'reference_sequence.txt').write_text(candidates[0]+'\n',encoding='ascii')
    records.append({'file':'reference_sequence.txt','source':'official reference annotation id MOD_PIKK_1_A0A024R7R0_0_719',
        'bytes':(directory/'reference_sequence.txt').stat().st_size,'sha256':hashlib.sha256((directory/'reference_sequence.txt').read_bytes()).hexdigest()})
    (directory/'DOWNLOAD_MANIFEST.json').write_text(json.dumps({'official_repository_commit':REV,'files':records,'all_elm_reference_assets_private':True},indent=2)+'\n')

if __name__=='__main__':main()

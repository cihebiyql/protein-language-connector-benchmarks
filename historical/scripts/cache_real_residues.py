"""Frozen ESM residue cache shared by real T1/T3 experiments."""
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'data/cache/real_residues.pt'

def build():
    sequences = set()
    for path in [ROOT/'data/processed/benz_samples.jsonl'] + sorted((ROOT/'data/processed').glob('t3_*pairs.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            for key in ('sequence','sequence_a','sequence_b'):
                sequence = row.get(key)
                if sequence and set(sequence) <= set('ACDEFGHIKLMNPQRSTVWY') and len(sequence)<=64:
                    sequences.add(sequence)
    sequences = sorted(sequences)
    if CACHE.exists():
        cache = torch.load(CACHE, map_location='cpu', weights_only=True)
        if set(sequences) <= set(cache['sequences']):
            return cache
    path = ROOT/'models/facebook--esm2_t30_150M_UR50D'
    tokenizer = AutoTokenizer.from_pretrained(path,local_files_only=True)
    model = AutoModel.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,
                                      add_pooling_layer=False).cuda().eval().requires_grad_(False)
    features = {layer:torch.zeros(len(sequences),64,640,dtype=torch.bfloat16) for layer in (15,30)}
    mask = torch.zeros(len(sequences),64)
    with torch.no_grad():
        for start in range(0,len(sequences),64):
            seqs = sequences[start:start+64]
            tokens = tokenizer(seqs,return_tensors='pt',padding=True).to('cuda')
            hidden = model(**tokens,output_hidden_states=True).hidden_states
            for i,seq in enumerate(seqs):
                assert tokens['attention_mask'][i].sum().item()==len(seq)+2
                mask[start+i,:len(seq)]=1
                for layer in features:
                    value = hidden[layer][i,1:len(seq)+1].cpu()
                    assert torch.isfinite(value).all()
                    features[layer][start+i,:len(seq)]=value
    cache = {'sequences':sequences,'mask':mask,'features15':features[15],'features30':features[30],
             'model_revision':'a695f6045e2e32885fa60af20c13cb35398ce30c',
             'bos_eos_padding_removed':True}
    CACHE.parent.mkdir(exist_ok=True,parents=True)
    torch.save(cache,CACHE)
    del model,hidden,tokens
    torch.cuda.empty_cache()
    print(json.dumps({'cached_sequences':len(sequences),'path':str(CACHE)}),flush=True)
    return cache

if __name__=='__main__':
    torch.set_num_threads(4)
    build()

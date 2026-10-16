"""Matched three-seed T1 training and real message interventions."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from cache_real_residues import ROOT,build
from real_bridge_model import FrozenBridgeScorer,ALIASES
from real_data import load_t1,ranking_metrics,epoch_order,make_donors,paired_group_bootstrap

OUT=ROOT/'results/real_run_20261015'

def write(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',choices=['L30-J0','L15-J14'],required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[17,42,2026])
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--microbatch',type=int,default=2)
    parser.add_argument('--eval-batch',type=int,default=16)
    args=parser.parse_args()
    assert args.epochs==30 and args.microbatch in (2,4,8)
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True,parents=True)
    rows,target_ids=load_t1()
    assert len(target_ids)==34 and all(len(row['sequence'])==16 for row in rows)
    tmap={target:i for i,target in enumerate(target_ids)}
    splits={s:[i for i,row in enumerate(rows) if row['split']==s] for s in ('train','validation','test')}
    test_rows=[rows[i] for i in splits['test']]
    donors=make_donors(test_rows,seed=42,n=10)
    write(OUT/'bridge_donors.json',{'seed':42,'test_sample_ids':[r['sample_id'] for r in test_rows],'donor_indices':donors})
    cache=build()
    seqmap={s:i for i,s in enumerate(cache['sequences'])}
    layer,insertion=(30,0) if args.config=='L30-J0' else (15,14)
    features=cache[f'features{layer}'][[seqmap[r['sequence']] for r in rows],:16].cuda()
    masks=torch.ones(len(rows),16,device='cuda')
    del cache
    model=FrozenBridgeScorer(insertion)
    prompt,start=model.encode('t1',16)
    positive_indices=[[tmap[t] for t in row['targets']] for row in rows]

    def evaluate(selected,mode='correct',donor_number=None):
        scores=[]
        with torch.no_grad():
            for begin in range(0,len(selected),args.eval_batch):
                current=selected[begin:begin+args.eval_batch]
                source=current if donor_number is None else [splits['test'][donors[k][donor_number]] for k in range(begin,begin+len(current))]
                logits=model(features[source],masks[source],[prompt]*len(current),start,message_mode=mode)
                scores.append(logits.cpu().numpy())
        return np.concatenate(scores)

    suite=[]
    for seed in args.seeds:
        run=OUT/f'{args.config}_seed{seed}'
        run.mkdir(exist_ok=True)
        if (run/'result.json').exists():
            suite.append(json.loads((run/'result.json').read_text(encoding='utf-8')))
            continue
        config={'task':'T1 fixed-catalog single-token ranking','configuration':args.config,'seed':seed,
                'epochs':30,'effective_batch_size':64,'microbatch_size':args.microbatch,
                'lr':3e-4,'optimizer':'AdamW betas(.9,.999) weight_decay.01','selection':'earliest highest validation MRR',
                'alias_map':dict(zip(target_ids,ALIASES)),'alias_token_ids':model.alias_token_ids,
                'input_sha256':hashlib.sha256((ROOT/'data/processed/benz_samples.jsonl').read_bytes()).hexdigest(),
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'model_script_sha256':hashlib.sha256((ROOT/'scripts/real_bridge_model.py').read_bytes()).hexdigest(),
                'loss':'full-vocabulary single-answer-token CE, averaged over documented positives per biological sample',
                'candidate_scoring':'frozen LM head candidate rows; no trainable classification head',
                'prompt_length':len(prompt),'molecular_slots':16}
        write(run/'config.json',config)
        (run/'train_real_bridge.executed.py').write_bytes(Path(__file__).read_bytes())
        (run/'real_bridge_model.executed.py').write_bytes((ROOT/'scripts/real_bridge_model.py').read_bytes())
        model.reset_adapter(seed)
        optimizer=torch.optim.AdamW(model.adapter.parameters(),lr=3e-4,weight_decay=.01)
        history=[]
        best=-1
        steps=0
        started=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,31):
            order=epoch_order(splits['train'],seed,epoch).tolist()
            loss_sum=0
            epoch_started=time.perf_counter()
            for offset in range(0,len(order),64):
                effective=order[offset:offset+64]
                optimizer.zero_grad(set_to_none=True)
                for mb in range(0,len(effective),args.microbatch):
                    selected=effective[mb:mb+args.microbatch]
                    logits=model(features[selected],masks[selected],[prompt]*len(selected),start,full_vocab=True)
                    losses=model.soft_positive_loss(logits,[positive_indices[i] for i in selected])
                    assert torch.isfinite(losses).all()
                    (losses.sum()/len(effective)).backward()
                    loss_sum+=losses.detach().sum().item()
                assert all(p.grad is None for p in model.qwen.parameters())
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.adapter.parameters())
                optimizer.step()
                steps+=1
            val_scores=evaluate(splits['validation'])
            validation,_=ranking_metrics(val_scores,[rows[i] for i in splits['validation']],target_ids)
            entry={'epoch':epoch,'optimizer_steps':steps,'train_loss':loss_sum/len(order),
                   'validation':validation,'seconds':time.perf_counter()-epoch_started}
            history.append(entry)
            if validation['mrr']>best:
                best=validation['mrr']
                best_epoch=epoch
                torch.save({'adapter':model.adapter.state_dict(),'epoch':epoch,'config':config},run/'best.pt')
            write(run/'history.json',history)
            print(json.dumps({'config':args.config,'seed':seed,**entry}),flush=True)
        assert steps==900
        state=torch.load(run/'best.pt',map_location='cuda',weights_only=True)
        model.adapter.load_state_dict(state['adapter'])
        correct=evaluate(splits['test'])
        off=evaluate(splits['test'],mode='off')
        mismatched=np.stack([evaluate(splits['test'],donor_number=i) for i in range(10)],axis=1)
        conditions={}
        per_sample={}
        predictions={}
        for condition,scores in [('correct',correct),('off',off)]:
            conditions[condition],predictions[condition]=ranking_metrics(scores,test_rows,target_ids)
            per_sample[condition]={key:np.array([p[key] for p in predictions[condition]]) for key in ('recall_at_3','reciprocal_rank')}
        mismatch_preds=[ranking_metrics(mismatched[:,i],test_rows,target_ids)[1] for i in range(10)]
        per_sample['mismatched']={key:np.mean([[p[key] for p in ps] for ps in mismatch_preds],axis=0) for key in ('recall_at_3','reciprocal_rank')}
        conditions['mismatched']={'n':len(test_rows),'donors_per_sample':10,
                                 'known_positive_recall_at_3':float(per_sample['mismatched']['recall_at_3'].mean()),
                                 'mrr':float(per_sample['mismatched']['reciprocal_rank'].mean())}
        effects={condition:{key:paired_group_bootstrap(per_sample['correct'][key],per_sample[condition][key],test_rows,seed=2026,n_boot=2000)
                            for key in ('recall_at_3','reciprocal_rank')} for condition in ('off','mismatched')}
        np.savez_compressed(run/'test_scores.npz',correct=correct,off=off,mismatched=mismatched,
                            sample_ids=np.array([r['sample_id'] for r in test_rows]))
        records=[]
        for i,row in enumerate(test_rows):
            records.append({'sample_id':row['sample_id'],'group_id':row['group_id'],
                            'correct':predictions['correct'][i],'off':predictions['off'][i],
                            'mismatched':[ps[i] for ps in mismatch_preds],
                            'donor_sample_ids':[test_rows[d]['sample_id'] for d in donors[i]]})
        (run/'test_predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n',encoding='utf-8')
        result={'config':config,'best_epoch':best_epoch,'optimizer_steps':steps,'history':history,
                'test':conditions,'message_effects':effects,'training_and_evaluation_seconds':time.perf_counter()-started,
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
        write(run/'result.json',result)
        suite.append(result)
        print(json.dumps({'completed':args.config,'seed':seed,'best_epoch':best_epoch,'test':conditions}),flush=True)
    write(OUT/f'{args.config}_summary.json',{'configuration':args.config,'runs':suite})

if __name__=='__main__':
    main()

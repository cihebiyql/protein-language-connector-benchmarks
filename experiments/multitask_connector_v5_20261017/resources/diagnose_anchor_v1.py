"""Bounded TRAIN-only shape-versus-causality diagnosis of saved failed canaries."""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    scope, output = Path(args.scope_root), Path(args.output)
    sys.path.insert(0, str(scope))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '3':
        raise ValueError('Diagnosis may only use physical GPU3')
    import torch
    import transformers
    from src.mt_train import configure, load_components, load_candidate_vectors, Collator, file_sha
    from src.mt_data import load_task, load_catalog
    from src.mt_extract import FeatureStore
    configure(17)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)
    torch.cuda.reset_peak_memory_stats()
    config = json.loads((scope / 'configs/development_v2.json').read_text())
    store = FeatureStore(scope / 'cache')
    source_pins = {p.name: file_sha(p) for p in sorted((scope/'src').glob('*.py'))}
    report = {'schema': 'TRAIN-anchor-shape-diagnosis.v1', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'torch':torch.__version__, 'transformers':transformers.__version__, 'cuda':torch.version.cuda,
        'physical_gpu':3, 'allocator_cap_bytes':8*1024**3, 'source_pins':source_pins,
        'script_sha256':file_sha(__file__), 'config_sha256':file_sha(scope/'configs/development_v2.json'),
        'validation_or_test_opened':False, 'new_training_updates':0, 'tasks':{}}

    def pack(result):
        return {key: ({name: tensor.detach().cpu() for name,tensor in value.items()} if key=='routes' and value is not None
                       else value.detach().cpu() if torch.is_tensor(value) else value)
                for key,value in result.items() if key in ('prompt_hidden','numeric_z','ranking_logits','routes')}

    def compare(left, right):
        result={}
        for key in ('prompt_hidden','numeric_z','ranking_logits'):
            if key in left:
                a,b=left[key].float(),right[key].float()
                result[key]={'equal':torch.equal(a,b),'allclose_1e5':torch.allclose(a,b,atol=1e-5,rtol=1e-5),
                    'max_abs_difference':float((a-b).abs().max()), 'l2_difference':float((a-b).norm())}
        for name in ('view','bank'):
            if left.get('routes') is not None:
                a,b=left['routes'][name],right['routes'][name]
                result['routes_'+name]={'equal':torch.equal(a,b),'allclose_1e5':torch.allclose(a,b,atol=1e-5,rtol=1e-5),
                    'max_abs_difference':float((a-b).abs().max())}
        return result

    def serialize(value):
        if torch.is_tensor(value):return {'dtype':str(value.dtype),'shape':list(value.shape),'values':value.tolist()}
        if isinstance(value,dict):return {k:serialize(v) for k,v in value.items()}
        return value

    for task in ('meltome','elm'):
        attempt=scope/'runs/qualification_v1'/(task+'__M3__qwen25_1p5b__seed17')/'artifacts'
        checkpoint=attempt/'checkpoint.pt'
        pin_before=file_sha(checkpoint)
        train=load_task(scope/'data',task,'train')
        row=train[:1]
        catalog=load_catalog(scope/'data') if task=='elm' else []
        vectors,_=load_candidate_vectors(scope/'cache',catalog) if task=='elm' else (None,None)
        model,tokenizer,assets=load_components(config,task,'M3','qwen25_1p5b','cuda:0',vectors)
        selected=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_trainable_state_dict(selected['trainable_state'])
        model.eval()
        collator=Collator(store,tokenizer,'cuda:0',catalog)
        original=collator(row,supervised=False)
        batches={'original':original}
        for name,token in (('future_A',1),('future_B',2)):
            suffix=torch.full((1,2),token,dtype=torch.long,device='cuda:0')
            batches[name]={**original,'input_ids':torch.cat((original['input_ids'],suffix),dim=1),
                'attention_mask':torch.cat((original['attention_mask'],torch.ones_like(suffix)),dim=1)}
        mutated={**batches['future_A'],'input_ids':batches['future_A']['input_ids'].clone()}
        location=int(original['prompt_end_index'][0])-5
        old_token=int(mutated['input_ids'][0,location])
        mutated['input_ids'][0,location]=(old_token+17)%model.base_lm.config.vocab_size
        batches['prefix_mutation']=mutated
        outcomes={}
        with torch.inference_mode():
            for off in (False,True):
                channel='connector_off' if off else 'connector_on'
                outcomes[channel]={name:pack(model(**batch,off=off)) for name,batch in batches.items()}
        comparisons={channel:{'shape_change_original_vs_future_A':compare(values['original'],values['future_A']),
                    'same_shape_future_A_vs_B':compare(values['future_A'],values['future_B']),
                    'same_shape_prefix_positive_control':compare(values['future_A'],values['prefix_mutation'])}
                    for channel,values in outcomes.items()}
        state_unchanged=all(torch.equal(value.cpu(),model.trainable_state_dict()[key])
            for key,value in selected['trainable_state'].items())
        report['tasks'][task]={'checkpoint_sha256_before':pin_before,'checkpoint_sha256_after':file_sha(checkpoint),
            'trainable_state_unchanged':state_unchanged,'source_failure':json.loads((attempt/'failed.json').read_text()),
            'source_anchor_record':json.loads((attempt/'train_anchor_qualification.json').read_text()),
            'source_split':'train','sample_id':row[0]['sample_id'],'input_lengths':{k:int(b['input_ids'].shape[1]) for k,b in batches.items()},
            'prompt_end_index':int(original['prompt_end_index'][0]),'suffix_token_ids_A':[1,1],'suffix_token_ids_B':[2,2],
            'prefix_mutation':{'location':location,'old_token':old_token,'new_token':int(mutated['input_ids'][0,location])},
            'base_dtype':str(next(model.base_lm.parameters()).dtype),'gate':float(model.alpha.tanh()),
            'assets':assets,'comparisons':comparisons,'outcomes':serialize(outcomes)}
        print(json.dumps({'task':task,'comparisons':comparisons,'checkpoint_unchanged':file_sha(checkpoint)==pin_before,
            'trainable_state_unchanged':state_unchanged}),flush=True)
        del model,tokenizer,selected,batches,outcomes,original,mutated,collator
        gc.collect()
        torch.cuda.empty_cache()
    report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
    report['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    if {p.name:file_sha(p) for p in sorted((scope/'src').glob('*.py'))}!=source_pins:
        raise ValueError('Source changed while diagnosing')
    with (output/'diagnosis.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


if __name__=='__main__':main()

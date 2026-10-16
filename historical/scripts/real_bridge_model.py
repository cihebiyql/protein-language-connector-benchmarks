"""Frozen Qwen scoring through the same 563201-parameter residue adapter."""
from contextlib import nullcontext
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from bridge import ResidueBridge, add_message, inject_after
from cache_real_residues import ROOT

ALIASES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ01234567')

class FrozenBridgeScorer(nn.Module):
    def __init__(self, insertion_layer):
        super().__init__()
        path=ROOT/'models/Qwen--Qwen2.5-1.5B-Instruct'
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.qwen=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,
                                                      attn_implementation='sdpa').cuda().eval().requires_grad_(False)
        self.adapter=ResidueBridge().cuda()
        self.insertion_layer=insertion_layer
        self.alias_token_ids=[self.tokenizer.encode(x,add_special_tokens=False) for x in ALIASES]
        assert all(len(x)==1 for x in self.alias_token_ids)
        self.alias_token_ids=[x[0] for x in self.alias_token_ids]
        assert len(set(self.alias_token_ids))==34

    def reset_adapter(self,seed):
        torch.manual_seed(seed)
        self.adapter=ResidueBridge().cuda()

    def encode(self,task,slots,target_ids=None):
        if task=='t1':
            content=('Read this molecular message: [MOLECULAR]\n'
                     'Rank the interaction target by its label. Valid target labels: '+ ' '.join(ALIASES)+
                     '. Reply with one label only.')
        else:
            assert task=='t3' and target_ids
            content=('The molecular message contains 64 reserved slots for peptide A then 64 for peptide B: [MOLECULAR]\n'
                     'Target binding domain: '+str(target_ids)+'. Which peptide has stronger measured binding? '
                     'Reply A or B only.')
        prompt=self.tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
        left,right=prompt.split('[MOLECULAR]')
        prefix=self.tokenizer.encode(left,add_special_tokens=False)
        ids=prefix+[self.tokenizer.encode('!',add_special_tokens=False)[0]]*slots+self.tokenizer.encode(right,add_special_tokens=False)
        assert len(ids)<=512
        return ids,len(prefix)

    def forward(self,features,residue_mask,prompts,start,*,message_mode='correct',full_vocab=False):
        length=max(map(len,prompts))
        device=features.device
        ids=torch.full((len(prompts),length),self.tokenizer.pad_token_id,device=device,dtype=torch.long)
        mask=torch.zeros_like(ids)
        positions=[]
        for i,prompt in enumerate(prompts):
            ids[i,:len(prompt)]=torch.tensor(prompt,device=device)
            mask[i,:len(prompt)]=1
            positions.append(len(prompt)-1)
        message=self.adapter(features.float())*residue_mask.unsqueeze(-1)
        if message_mode=='off':
            message=message*0
        embeds=self.qwen.get_input_embeddings()(ids)
        if self.insertion_layer==0:
            embeds=add_message(embeds,message,start)
            context=nullcontext()
        else:
            context=inject_after(self.qwen,self.insertion_layer,message,start)
        with context:
            hidden=self.qwen.model(inputs_embeds=embeds,attention_mask=mask,use_cache=False).last_hidden_state
        last=hidden[torch.arange(len(prompts),device=device),torch.tensor(positions,device=device)]
        weights=self.qwen.lm_head.weight
        if not full_vocab:
            weights=weights[self.alias_token_ids]
        return torch.nn.functional.linear(last,weights).float()

    def soft_positive_loss(self,logits,positive_indices):
        log_probs=logits.log_softmax(-1)
        return torch.stack([-log_probs[i,[self.alias_token_ids[j] for j in positives]].mean()
                            for i,positives in enumerate(positive_indices)])

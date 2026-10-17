"""V4 loss-equivalence, backbone-isolation and independent-deadline regressions."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from src import mm_train as train
from src import mm_worker_limit as limit
from src import mm_data as data
from src import mm_extract as extract
from tests.test_train import TinyModel, protocol, rows, collate


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_accumulation_equals_full_batch_with_unequal_tokens_and_tail(self):
        records = rows('train', 7)
        def varying(selected, supervised=True):
            batch = collate(selected, supervised)
            batch['input_ids'] = torch.tensor([[1 + int(row['sample_id'].split(':')[-1]) % 3, 2, 3, 2, 3] for row in selected])
            batch['attention_mask'] = torch.ones_like(batch['input_ids'])
            if supervised:
                batch['labels'] = torch.tensor([[-100, -100, 3, 2, 3] if int(row['sample_id'].split(':')[-1]) % 2 else [-100, -100, -100, -100, 3] for row in selected])
            return batch
        original = TinyModel()
        states, histories = [], []
        for micro in (4, 2, 1):
            model = deepcopy(original)
            cfg = protocol()
            cfg['training'].update(batch_size=4, effective_batch_size=4, micro_batch_size=micro,
                                   grad_accumulation=4 // micro, epochs=2)
            with tempfile.TemporaryDirectory() as directory:
                result = train.fit_model(model, records, rows('validation', 3), varying, cfg,
                                         {'seed':17}, Path(directory), 'cpu')
                states.append(model.trainable_state_dict())
                histories.append(json.loads((Path(directory) / 'history.json').read_text()))
                self.assertEqual(result['optimizer_steps'], 4)
                self.assertEqual(result['examples_seen'], 14)
        for state in states[1:]:
            for name, value in states[0].items():
                self.assertTrue(torch.allclose(value, state[name], atol=3e-7, rtol=3e-6), name)
        for history in histories[1:]:
            for expected, actual in zip(histories[0], history):
                self.assertAlmostEqual(expected['train_answer_loss_sum'], actual['train_answer_loss_sum'], places=5)
                self.assertEqual(expected['train_answer_token_count'], actual['train_answer_token_count'])
                self.assertAlmostEqual(expected['train_numeric_mse_z'], actual['train_numeric_mse_z'], places=5)
        self.assertEqual(math.ceil(2062 / 4) * 10, 5160)
        self.assertEqual(math.ceil(2063 / 4) * 10, 5160)

    def test_accumulation_shape_guard(self):
        for values in ((2, 2, 5), (True, 2, 2), (0, 4, 4)):
            with self.assertRaises(ValueError):
                train.batch_dimensions(dict(zip(('micro_batch_size','grad_accumulation','effective_batch_size'), values)))

    def test_backbone_exact_identity_and_checkpoint_cannot_cross(self):
        cfg = protocol()
        setting = train.backbone_settings(cfg, 'qwen25_1p5b')
        self.assertEqual(setting['hook_layer'], 13)
        for key, value in (('hook_layer',14),('revision','0'*40),('hidden_size',2560)):
            bad = deepcopy(cfg); bad['backbones']['qwen25_1p5b'][key] = value
            with self.assertRaises(ValueError): train.backbone_settings(bad,'qwen25_1p5b')
        identity={'scope_id':train.SCOPE_ID,'backbone':'qwen25_1p5b','seed':17}
        payload={'scope_id':train.SCOPE_ID,'identity':identity,'epoch':1,'selection':{'best_epoch':1}}
        train.verify_checkpoint_identity(payload,identity)
        with self.assertRaises(ValueError):
            train.verify_checkpoint_identity(payload,{**identity,'backbone':'qwen3_4b_2507'})

    def test_non_thinking_template_is_explicit_only_for_qwen3(self):
        from tests.test_data import Tokenizer, row
        class SpyTokenizer(Tokenizer):
            def apply_chat_template(self,*args,**kwargs):
                self.seen=kwargs
                return 'rendered'
        tok=SpyTokenizer(); data.prompt_text(row(),tok)
        self.assertNotIn('enable_thinking',tok.seen)
        tok._mm_backbone_id='qwen3_4b_2507'; data.prompt_text(row(),tok)
        self.assertIs(tok.seen['enable_thinking'],False)

    def test_longest_train_qualification_excludes_val_test(self):
        class Tok: pass
        selected=rows('train',7)
        def lengths(part,supervised=True):
            n=int(part[0]['sample_id'].split(':')[-1])+3
            return {'input_ids':torch.ones(1,n,dtype=torch.long),
                    'attention_mask':torch.ones(1,n,dtype=torch.long),
                    'prompt_end_index':torch.tensor([n-2])}
        receipt=train.train_length_qualification(selected,Tok(),lengths,4)
        self.assertEqual(receipt['all_train_rows_inspected'],7)
        self.assertEqual(receipt['first_effective_batch_indices'],[6,5,4,3])
        self.assertEqual(receipt['max_train_prompt_tokens'],8)
        with self.assertRaises(ValueError):
            train.train_length_qualification(rows('validation'),Tok(),lengths,4)

    def test_cache_consumer_pins_producer_not_installed_esm(self):
        from tests.test_data import TinyESM, Tokenizer
        with tempfile.TemporaryDirectory() as parent:
            root=Path(parent)/'cache'
            semantics={'verified':True,'source_sha256':extract.REMOTE_TF4481_SOURCE_SHA,
                       'transformers_version':'4.48.1'}
            manifest=extract.extract_cache(TinyESM(640),Tokenizer(),['ACD'],root,semantics,extract.MODEL_PINS)
            current=extract.file_sha(extract.__file__)
            with patch.object(extract,'FROZEN_EXTRACTOR_SHA',current), \
                 patch.object(extract,'inspect_esm_semantics',side_effect=AssertionError('consumer must not inspect installed ESM')):
                store=extract.SequenceStore(root,manifest_sha256=extract.file_sha(root/'manifest.json'))
                self.assertEqual(len(store.verify()['verified_payloads']),1)
            with self.assertRaisesRegex(ValueError,'immutable source closure'):
                extract.SequenceStore(root)
            manifest['extractor_sha256']='0'*64
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'extractor source'):
                extract.SequenceStore(root,manifest_sha256=extract.file_sha(root/'manifest.json'))

    def test_worker_guard_rejects_missing_naive_expired_or_nonfinite_budgets(self):
        future=datetime.fromtimestamp(2000,tz=timezone.utc).isoformat()
        valid={'MM_WORKER_ABSOLUTE_DEADLINE':future,'MM_WORKER_MONOTONIC_DEADLINE':'50'}
        self.assertEqual(limit.guard_budget(valid,wall_now=1000,monotonic_now=40),(2000,50))
        for env in ({}, {**valid,'MM_WORKER_ABSOLUTE_DEADLINE':'2026-01-01T00:00:00'},
                    {**valid,'MM_WORKER_MONOTONIC_DEADLINE':'nan'},
                    {**valid,'MM_WORKER_MONOTONIC_DEADLINE':'39'}):
            with self.assertRaises((ValueError,KeyError)):
                limit.guard_budget(env,wall_now=1000,monotonic_now=40)

    @unittest.skipUnless(sys.platform.startswith('linux'),'Linux exact-parent deadline guard')
    def test_independent_guard_kills_only_own_worker_when_parent_monitor_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt=Path(directory)/'expired.json'
            code=("from src.mm_worker_limit import worker_deadline_guard;import time;"
                  "g=worker_deadline_guard(%r);g.__enter__();print('ready',flush=True);time.sleep(5)" % str(receipt))
            env=os.environ.copy()
            env.update(MM_WORKER_ABSOLUTE_DEADLINE=datetime.fromtimestamp(time.time()+1.5,tz=timezone.utc).isoformat(),
                       MM_WORKER_MONOTONIC_DEADLINE=str(time.monotonic()+1.5))
            peer=subprocess.Popen([sys.executable,'-c','import time;time.sleep(5)'])
            worker=subprocess.Popen([sys.executable,'-c',code],env=env,stdout=subprocess.PIPE,text=True)
            try:
                self.assertEqual(worker.stdout.readline().strip(),'ready')
                self.assertEqual(worker.wait(timeout=4),-9)
                self.assertIsNone(peer.poll())
                self.assertEqual(json.loads(receipt.read_text())['target_pid'],worker.pid)
            finally:
                if worker.poll() is None: worker.kill();worker.wait(timeout=2)
                peer.terminate();peer.wait(timeout=2);worker.stdout.close()

    def test_real_random_tiny_qwen2_and_qwen3_hooks_gradients_and_prompt_anchor(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
        from src.mm_model import MolecularMemoryLM
        for cfg_cls, model_cls in ((Qwen2Config,Qwen2ForCausalLM),(Qwen3Config,Qwen3ForCausalLM)):
            cfg=cfg_cls(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=4,
                        num_attention_heads=4,num_key_value_heads=2,head_dim=4,max_position_embeddings=64,
                        attention_dropout=0,pad_token_id=0,bos_token_id=1,eos_token_id=2)
            cfg._attn_implementation='eager'
            base=model_cls(cfg).eval().requires_grad_(False)
            model=MolecularMemoryLM(base,'M3',hidden_dim=16,esm_dim=6,memory_dim=8,num_heads=2,hook_layer=1)
            ids=torch.tensor([[0,3,4,5,6],[2,3,4,7,8]])
            batch={'input_ids':ids,'attention_mask':torch.tensor([[0,1,1,1,1],[1,1,1,1,1]]),
                   'prompt_end_index':torch.tensor([2,2]),
                   'memory_inputs':{name:torch.randn(2,8,6) for name in ('p15','p30','t15','t30')}}
            batch['memory_inputs'].update(condition_ids=torch.zeros(2,10,dtype=torch.long),observed=torch.zeros(2,10,dtype=torch.bool))
            with torch.no_grad():
                on=model(**batch);off=model(**batch,off=True)
                self.assertTrue(torch.equal(on['logits'],off['logits']))
                model.alpha.fill_(.3)
                expected=model(**batch)['numeric_z']
                changed={**batch,'input_ids':ids.clone()};changed['input_ids'][:,3:]=12
                self.assertTrue(torch.equal(expected,model(**changed)['numeric_z']))
            result=model(**batch)
            (result['logits'].square().mean()+result['numeric_z'].square().mean()).backward()
            self.assertTrue(all(p.grad is None for p in base.parameters()))
            self.assertGreater(float(model.projections[0].weight.grad.abs().sum()),0)
            self.assertEqual(len(base.model.layers[1]._forward_hooks),0)
            self.assertEqual(len(base.model.norm._forward_hooks),0)


if __name__=='__main__': unittest.main(verbosity=2)

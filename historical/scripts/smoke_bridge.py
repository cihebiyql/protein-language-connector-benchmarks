"""Bounded architecture canary: synthetic inputs, never a biological benchmark score."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from bridge import ResidueBridge, bridge_forward

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--residues', type=int, choices=(16, 64), default=16)
    parser.add_argument('--output', default='results/bridge_smoke.json')
    args = parser.parse_args()
    assert 64 <= args.length <= 512 and 1 <= args.batch_size <= 2 and 1 <= args.steps <= 5
    torch.manual_seed(17)
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    device = 'cuda'
    esm_path = ROOT / 'models/facebook--esm2_t30_150M_UR50D'
    qwen_path = ROOT / 'models/Qwen--Qwen2.5-1.5B-Instruct'
    esm_tokenizer = AutoTokenizer.from_pretrained(esm_path, local_files_only=True)
    esm = AutoModel.from_pretrained(esm_path, local_files_only=True, dtype=torch.bfloat16,
                                    add_pooling_layer=False).to(device).eval().requires_grad_(False)
    # Canonical alphabet fixtures test tensor communication only; these are not experimental peptides.
    sequences = [s * (args.residues // 16) for s in ['ACDEFGHIKLMNPQRS', 'SRQPNMLKIHGFEDCA'][:args.batch_size]]
    tokens = esm_tokenizer(sequences, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        hidden = esm(**tokens, output_hidden_states=True).hidden_states
        representations = {layer: hidden[layer][:, 1:args.residues+1].float().detach() for layer in (15, 30)}
    assert all(x.shape == (args.batch_size, args.residues, 640) and torch.isfinite(x).all() for x in representations.values())
    del esm, hidden, tokens
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(qwen_path, local_files_only=True, dtype=torch.bfloat16,
                                                attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content':
        'Reserved molecular slots: [SLOTS]\nEngineering test only. Reply with OK.'}], tokenize=False, add_generation_prompt=True)
    left, right = prompt.split('[SLOTS]')
    prefix = tokenizer.encode(left, add_special_tokens=False)
    suffix = tokenizer.encode(right, add_special_tokens=False)
    answer = tokenizer.encode('OK' + tokenizer.eos_token, add_special_tokens=False)
    slot_id = tokenizer.encode('!', add_special_tokens=False)[0]
    raw = prefix + [slot_id] * args.residues + suffix + answer
    assert len(raw) <= args.length
    pad = tokenizer.pad_token_id
    ids = torch.tensor([raw + [pad] * (args.length-len(raw))] * args.batch_size, device=device)
    mask = torch.tensor([[1]*len(raw) + [0]*(args.length-len(raw))] * args.batch_size, device=device)
    labels = torch.full_like(ids, -100)
    labels[:, len(raw)-len(answer):len(raw)] = ids[:, len(raw)-len(answer):len(raw)]
    position = len(raw) - len(answer) - 1
    rows = []
    for source_layer in (15, 30):
        for insertion in (0, 7, 14):
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(17)
            adapter = ResidueBridge().to(device)
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=3e-4)
            features = representations[source_layer]
            before = adapter.projection[1].weight.detach().clone()
            losses, gradient_norms = [], []
            started = time.perf_counter()
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                output = bridge_forward(model, ids, mask, adapter(features), len(prefix), insertion, labels)
                assert torch.isfinite(output.loss)
                losses.append(output.loss.item())
                output.loss.backward()
                grads = [p.grad for p in adapter.parameters()]
                assert all(g is not None and torch.isfinite(g).all() for g in grads)
                norm = torch.sqrt(sum(g.float().square().sum() for g in grads)).item()
                assert norm > 0
                gradient_norms.append(norm)
                assert not any(p.grad is not None for p in model.parameters())
                optimizer.step()
                del output
            with torch.no_grad():
                message = adapter(features)
                correct = bridge_forward(model, ids, mask, message, len(prefix), insertion).logits[:, position].float()
                off = bridge_forward(model, ids, mask, message*0, len(prefix), insertion).logits[:, position].float()
                changed = bridge_forward(model, ids, mask, adapter(features.flip(1)), len(prefix), insertion).logits[:, position].float()
            torch.cuda.synchronize()
            row = {'esm_layer': source_layer, 'qwen_after_layer': insertion, 'losses': losses,
                   'gradient_norms': gradient_norms, 'adapter_parameters': sum(p.numel() for p in adapter.parameters()),
                   'adapter_weight_changed': not torch.equal(before, adapter.projection[1].weight),
                   'base_models_frozen': True, 'message_off_max_logit_delta': (correct-off).abs().max().item(),
                   'reordered_residue_max_logit_delta': (correct-changed).abs().max().item(),
                   'seconds': time.perf_counter()-started, 'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                   'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20}
            assert row['adapter_weight_changed'] and row['message_off_max_logit_delta'] > 0
            rows.append(row)
            print(json.dumps(row), flush=True)
            del adapter, optimizer, correct, off, changed, message
            torch.cuda.empty_cache()
    result = {'scope': 'engineering_canary_only_not_biological_accuracy', 'seed': 17,
              'synthetic_sequences': sequences, 'batch_size': args.batch_size,
              'padded_sequence_length': args.length, 'nonpadding_length': len(raw),
              'real_residues': args.residues, 'steps_per_interface': args.steps,
              'precision': 'BF16 base models / FP32 adapter', 'python': platform.python_version(),
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'gpu': torch.cuda.get_device_name(), 'interfaces': rows,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'bridge_sha256': hashlib.sha256((ROOT/'scripts/bridge.py').read_bytes()).hexdigest()}
    dest = ROOT/args.output
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()

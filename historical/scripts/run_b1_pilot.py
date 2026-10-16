"""One-seed, validation-selected ESM2/MLP pilot on a fixed audited Benz split."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

def metrics(scores, rows, target_ids):
    rankings = np.argsort(-scores, axis=1, kind='stable')
    recalls, reciprocals, predictions = [], [], []
    for order, row in zip(rankings, rows):
        truth = set(row['targets'])
        ranked = [target_ids[i] for i in order]
        recalls.append(len(truth.intersection(ranked[:3])) / len(truth))
        reciprocals.append(1 / next(rank for rank, target in enumerate(ranked, 1) if target in truth))
        predictions.append({'sample_id': row['sample_id'], 'known_positive_target_ids': row['targets'],
                            'ranked_target_ids': ranked, 'recall_at_3': recalls[-1], 'reciprocal_rank': reciprocals[-1]})
    return {'n': len(rows), 'known_positive_recall_at_3': float(np.mean(recalls)),
            'mrr': float(np.mean(reciprocals))}, predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()
    assert 1 <= args.epochs <= 30 and 1 <= args.batch_size <= 128
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    rows_path = ROOT/'data/processed/benz_samples.jsonl'
    raw = rows_path.read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    rows = [row for row in rows if row['split'] in ('train', 'validation', 'test')]
    target_ids = [row['target_id'] for row in json.loads((ROOT/'data/processed/benz_targets.json').read_text(encoding='utf-8'))]
    target_map = {target: i for i, target in enumerate(target_ids)}
    assert rows and len(target_ids) == len(target_map)
    assert all(row['sequence'] and len(row['sequence']) <= 64 and row['targets'] for row in rows)
    split_groups = {split: {row['group_id'] for row in rows if row['split'] == split}
                    for split in ('train', 'validation', 'test')}
    assert all(not split_groups[a].intersection(split_groups[b]) for a,b in
               [('train','validation'), ('train','test'), ('validation','test')])
    fingerprint = hashlib.sha256(raw).hexdigest()
    cache_path = ROOT/'data/cache/benz_esm150m.pt'
    model_path = ROOT/'models/facebook--esm2_t30_150M_UR50D'
    encoding_started = time.perf_counter()
    cache_reused = cache_path.exists()
    if cache_reused:
        cache = torch.load(cache_path, map_location='cpu', weights_only=True)
        assert cache['input_sha256'] == fingerprint and cache['sample_ids'] == [row['sample_id'] for row in rows]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        encoder = AutoModel.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16,
                                            add_pooling_layer=False).cuda().eval().requires_grad_(False)
        layers = {15: [], 30: []}
        lengths = []
        with torch.no_grad():
            for start in range(0, len(rows), 64):
                batch = rows[start:start+64]
                tokens = tokenizer([row['sequence'] for row in batch], padding=True, return_tensors='pt').to('cuda')
                hidden = encoder(**tokens, output_hidden_states=True).hidden_states
                for i, row in enumerate(batch):
                    length = len(row['sequence'])
                    lengths.append(length)
                    assert tokens['attention_mask'][i].sum().item() == length + 2
                    for layer in layers:
                        value = hidden[layer][i, 1:length+1].float().mean(0).cpu()
                        assert value.shape == (640,) and torch.isfinite(value).all()
                        layers[layer].append(value)
                if start % 640 == 0:
                    print(f'Encoded {min(start+64,len(rows))}/{len(rows)}', flush=True)
        cache = {'input_sha256': fingerprint, 'sample_ids': [row['sample_id'] for row in rows],
                 'model_revision': 'a695f6045e2e32885fa60af20c13cb35398ce30c',
                 'residue_mean_bos_eos_padding_removed': True,
                 'features_15': torch.stack(layers[15]), 'features_30': torch.stack(layers[30])}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        del encoder, hidden, tokens
        torch.cuda.empty_cache()
    encoding_seconds = time.perf_counter()-encoding_started
    features = cache['features_30'].cuda()
    indices = {split: [i for i,row in enumerate(rows) if row['split']==split] for split in split_groups}
    assert all(indices.values())
    targets = torch.zeros(len(rows), len(target_ids), device='cuda')
    for i,row in enumerate(rows):
        targets[i, [target_map[target] for target in row['targets']]] = 1 / len(row['targets'])
    # Statistics are fitted on training features only.
    center = features[indices['train']].mean(0)
    scale = features[indices['train']].std(0).clamp_min(1e-5)
    features = (features-center)/scale
    # Encoder construction must not change head initialization between a fresh and cached run.
    torch.manual_seed(args.seed)
    head = nn.Sequential(nn.Linear(640,256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256,len(target_ids))).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    best_mrr, stale, history = -1.0, 0, []
    best_state, best_epoch = None, None
    training_started = time.perf_counter()
    for epoch in range(1,args.epochs+1):
        head.train()
        order = torch.tensor(indices['train'],device='cuda')
        order = order[torch.randperm(len(order),device='cuda')]
        total = 0.0
        for batch in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = head(features[batch])
            # Each biological sample has equal weight; only documented positives are targets.
            loss = -(targets[batch] * logits.log_softmax(-1)).sum(-1).mean()
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            total += loss.item()*len(batch)
        head.eval()
        with torch.no_grad():
            validation, _ = metrics(head(features[indices['validation']]).cpu().numpy(),
                                    [rows[i] for i in indices['validation']], target_ids)
        history.append({'epoch': epoch, 'train_loss': total/len(order), 'validation': validation})
        if validation['mrr'] > best_mrr:
            best_mrr, stale, best_epoch = validation['mrr'], 0, epoch
            best_state = {name: value.detach().cpu().clone() for name,value in head.state_dict().items()}
        else:
            stale += 1
        print(json.dumps(history[-1]),flush=True)
        if stale >= 5:
            break
    head.load_state_dict(best_state)
    head.eval()
    results_dir = ROOT/'results'
    results_dir.mkdir(exist_ok=True)
    scores_by_split = {}
    for split, selected in indices.items():
        with torch.no_grad():
            scores = head(features[selected]).cpu().numpy()
        scores_by_split[split], predictions = metrics(scores,[rows[i] for i in selected],target_ids)
        (results_dir/f'b1_{split}_predictions.jsonl').write_text('\n'.join(json.dumps(row) for row in predictions)+'\n',encoding='utf-8')
    torch.save({'head': best_state, 'center': center.cpu(), 'scale': scale.cpu(), 'target_ids':target_ids,
                'input_sha256':fingerprint, 'best_epoch':best_epoch, 'seed':args.seed}, results_dir/'b1_pilot_head.pt')
    result = {'scope':'one_seed_engineering_pilot_not_final_cross_source_benchmark',
              'model':'frozen ESM2-150M layer30 mean + 640/256/targets MLP',
              'seed':args.seed, 'learning_rate':args.lr, 'batch_size':args.batch_size,
              'loss':'equal-sample mean cross-entropy over documented positives; missing interactions are not confirmed negatives',
              'selection':'validation MRR only; patience=5; no test-based selection',
              'target_count':len(target_ids), 'best_epoch':best_epoch, 'input_sha256':fingerprint,
              'cache_reused':cache_reused, 'encoding_seconds':encoding_seconds,
              'head_training_seconds':time.perf_counter()-training_started,
              'scores':scores_by_split, 'history':history,
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (results_dir/'b1_pilot.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'best_epoch':best_epoch,'scores':scores_by_split}),flush=True)

if __name__ == '__main__':
    main()

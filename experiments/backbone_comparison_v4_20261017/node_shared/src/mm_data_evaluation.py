"""Explicit evaluation-only gold teacher forcing without changing pinned mm_data.

The evaluator supplies independently verified test gold only after checkpoint
selection. This helper cannot grant that access; its explicit phase argument is
an accidental-use guard, not a replacement for the runner's selection gate.
"""
from .mm_data import answer_text, collate as input_collate


def collate_evaluation(rows, tokenizer, store, device='cpu', *, phase):
    """Create original answer+EOS labels for test C1, never change source split.

    All rows must be test and carry finite ``pkd`` injected by the independent
    evaluation stage. No evaluation flag, sample identity or gold numeric target
    is added to model kwargs. Numeric readout keeps the pre-answer prompt index.
    """
    if phase != 'evaluate':
        raise ValueError('test gold teacher forcing requires explicit evaluate phase')
    if not rows or any(row.get('split') != 'test' for row in rows):
        raise ValueError('evaluation helper accepts only test rows')
    import torch
    # Validate all gold before accessing any feature cache. The pinned ordinary
    # collator remains responsible for the unchanged anonymous prompt/memory.
    suffixes = [tokenizer.encode(answer_text(row), add_special_tokens=False) + [tokenizer.eos_token_id]
                for row in rows]
    if any(len(suffix) < 2 for suffix in suffixes):
        raise ValueError('empty answer tokenization')
    batch = input_collate(rows, tokenizer, store, device=device, supervised=False)
    prefixes = [tokens[mask.bool()] for tokens, mask in zip(batch['input_ids'], batch['attention_mask'])]
    length = max(len(prefix) + len(suffix) for prefix, suffix in zip(prefixes, suffixes))
    ids = torch.full((len(rows), length), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    ends = []
    for i, (prefix, suffix) in enumerate(zip(prefixes, suffixes)):
        offset = length - len(prefix) - len(suffix)
        boundary = offset + len(prefix)
        ids[i, offset:boundary] = prefix
        answer = torch.tensor(suffix, dtype=torch.long, device=device)
        ids[i, boundary:] = answer
        labels[i, boundary:] = answer
        attention[i, offset:] = 1
        ends.append(boundary - 1)
    batch.update(input_ids=ids, attention_mask=attention, labels=labels,
                 prompt_end_index=torch.tensor(ends, dtype=torch.long, device=device))
    return batch

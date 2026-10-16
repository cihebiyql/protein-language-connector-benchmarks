"""Post-forward answer-token diagnostics; never an optimization/selection loss.

The legacy answer is encoded separately from its prompt, then exactly one EOS is
appended. We verify those existing IDs with fast-tokenizer offsets; no IDs or
labels are rewritten. Categories refer to target token positions, so the NLL
function applies the same causal shift as legacy ``supervised_nll``.

All losses use natural logarithms (nats). Category means have different
denominators and cannot be added. ``contribution_nats_per_full_token`` uses the
same full-answer denominator for each category and is additive, including when
comparing correct/off modes on the identical labelled answer.
"""
import math
import re

FORMAT, NUMERIC, MIXED = 0, 1, 2
IGNORE = -100
CATEGORY_NAMES = {FORMAT: 'format', NUMERIC: 'numeric', MIXED: 'mixed'}
_NUMBER = r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?'
_ANSWER = re.compile(r'\s*\{\s*"pKD"\s*:\s*(?P<number>' + _NUMBER + r')\s*\}\s*')


def answer_token_categories(tokenizer, text, answer_ids):
    """Return categories for the supplied, unchanged answer IDs including EOS.

    Only an exact single finite numeric pKD JSON field is accepted. A normal
    token wholly inside the numeric character span is numeric; one intersecting
    both number and format is mixed. EOS/special tokens and everything outside
    that span are format. Invalid offsets or any ID mismatch fail closed.
    """
    if not getattr(tokenizer, 'is_fast', False):
        raise ValueError('A fast tokenizer with character offsets is required')
    if not isinstance(text, str):
        raise ValueError('Answer text must be a string')
    match = _ANSWER.fullmatch(text)
    if match is None or not math.isfinite(float(match.group('number'))):
        raise ValueError('Answer requires exactly one finite numeric pKD field')
    provided = list(answer_ids)
    if any(not isinstance(i, int) or isinstance(i, bool) for i in provided):
        raise ValueError('Answer token IDs must be integers')
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        return_special_tokens_mask=True)
    if 'input_ids' not in encoded or 'offset_mapping' not in encoded:
        raise ValueError('Fast tokenizer did not return IDs and offsets')
    ids = list(encoded['input_ids'])
    offsets = list(encoded['offset_mapping'])
    specials = list(encoded.get('special_tokens_mask', [0] * len(ids)))
    eos = tokenizer.eos_token_id
    if not isinstance(eos, int) or isinstance(eos, bool) or not ids:
        raise ValueError('A nonempty answer encoding and one integer EOS are required')
    if ids != list(tokenizer.encode(text, add_special_tokens=False)) or provided != ids + [eos]:
        raise ValueError('Offset tokenization differs from original answer IDs plus EOS')
    if len(offsets) != len(ids) or len(specials) != len(ids):
        raise ValueError('Offsets/special mask do not align to answer IDs')
    # Qwen's fast tokenizer losslessly decodes these JSON strings. Refuse a
    # tokenizer that normalizes the text rather than silently moving its spans.
    decoded = tokenizer.decode(ids, skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
    if decoded != text:
        raise ValueError('Answer tokenization is not an exact text round trip')
    start_number, end_number = match.span('number')
    covered = [False] * len(text)
    previous_end = 0
    result = []
    all_special = set(getattr(tokenizer, 'all_special_ids', []))
    for token, offset, special in zip(ids, offsets, specials):
        if (not isinstance(offset, (tuple, list)) or len(offset) != 2
                or any(not isinstance(v, int) or isinstance(v, bool) for v in offset)
                or special not in (0, 1)):
            raise ValueError('Malformed token offset or special mask')
        begin, end = offset
        is_special = bool(special) or token in all_special
        if is_special and (begin, end) == (0, 0):
            result.append(FORMAT)
            continue
        if not 0 <= begin < end <= len(text) or begin < previous_end:
            raise ValueError('Offsets must be nonempty, ordered and within the answer')
        if not is_special and tokenizer.decode(
                [token], skip_special_tokens=False,
                clean_up_tokenization_spaces=False) != text[begin:end]:
            raise ValueError('Token text does not agree with its declared offset')
        previous_end = end
        for i in range(begin, end):
            covered[i] = True
        if is_special or end <= start_number or begin >= end_number:
            result.append(FORMAT)
        elif begin >= start_number and end <= end_number:
            result.append(NUMERIC)
        else:
            result.append(MIXED)
    if any(not yes and not char.isspace() for char, yes in zip(text, covered)):
        raise ValueError('Offsets do not cover every non-whitespace answer character')
    return result + [FORMAT]  # The exactly verified appended EOS.


def batch_token_categories(tokenizer, answer_texts, input_ids, labels, attention_mask=None):
    """Align answer categories to an existing legacy collate batch, including padding."""
    import torch
    if (not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor)
            or labels.ndim != 2 or input_ids.shape != labels.shape
            or input_ids.dtype != torch.long or labels.dtype != torch.long
            or input_ids.device != labels.device or len(answer_texts) != len(labels)):
        raise ValueError('Input IDs and labels must be aligned int64 [batch, text] tensors')
    if attention_mask is not None:
        if (attention_mask.shape != labels.shape or attention_mask.device != labels.device
                or not torch.all((attention_mask == 0) | (attention_mask == 1))):
            raise ValueError('Attention mask must be aligned and binary')
        if torch.any((labels != IGNORE) & ~attention_mask.bool()):
            raise ValueError('Padding cannot carry a supervised answer label')
    if labels.shape[1] == 0 or torch.any(labels[:, 0] != IGNORE):
        raise ValueError('A prompt position must precede every causal answer')
    categories = torch.full_like(labels, IGNORE)
    for row, text in enumerate(answer_texts):
        positions = torch.nonzero(labels[row] != IGNORE, as_tuple=False).flatten()
        if not len(positions) or not torch.equal(
                positions, torch.arange(int(positions[0]), int(positions[-1]) + 1,
                                        device=positions.device)):
            raise ValueError('Supervised answer tokens must form one nonempty suffix span')
        if attention_mask is not None:
            active = torch.nonzero(attention_mask[row], as_tuple=False).flatten()
            if not len(active) or int(active[-1]) != int(positions[-1]):
                raise ValueError('The supervised EOS must end the active text')
        answer_ids = labels[row, positions]
        if not torch.equal(answer_ids, input_ids[row, positions]):
            raise ValueError('Existing answer labels differ from input IDs')
        values = answer_token_categories(tokenizer, text, answer_ids.detach().cpu().tolist())
        categories[row, positions] = torch.tensor(values, dtype=torch.long, device=labels.device)
    return categories


def token_nll_breakdown(output_or_logits, labels, categories):
    """Partition existing forward logits into answer-token loss sums and counts.

    Diagnostics detach logits and perform no forward pass. Float32 per-token CE
    follows the legacy criterion. Python sums compose the diagnostic full value
    exactly from its three category sums. The original float32 sum reduction is
    separately returned as ``reference_supervised_nll``; small reduction-rounding
    differences are exposed, never assigned to an arbitrary token category.
    """
    import torch
    from torch.nn import functional as F
    logits = getattr(output_or_logits, 'logits', output_or_logits)
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 3
            or not logits.is_floating_point() or labels.ndim != 2
            or logits.shape[:2] != labels.shape or categories.shape != labels.shape
            or labels.dtype != torch.long or categories.dtype != torch.long
            or logits.device != labels.device or categories.device != labels.device):
        raise ValueError('Logits, labels and categories must align to original text positions')
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite logits in answer-token diagnostics')
    if torch.any((labels == IGNORE) != (categories == IGNORE)):
        raise ValueError('Categories must ignore exactly the ignored labels')
    valid = labels != IGNORE
    if torch.any(valid & ((labels < 0) | (labels >= logits.shape[-1]))):
        raise ValueError('Supervised label outside logits vocabulary')
    if torch.any(valid & ((categories < FORMAT) | (categories > MIXED))):
        raise ValueError('Unknown answer-token category')
    if labels.shape[1] < 2 or torch.any(labels[:, 0] != IGNORE):
        raise ValueError('A prompt position must precede every causal answer')
    with torch.no_grad():
        scores, targets = logits.detach()[:, :-1].float(), labels[:, 1:]
        shifted_categories = categories[:, 1:]
        count = int((targets != IGNORE).sum())
        if not count:
            raise ValueError('No supervised answer tokens')
        flat_scores, flat_targets = scores.reshape(-1, scores.shape[-1]), targets.reshape(-1)
        losses = F.cross_entropy(flat_scores, flat_targets, ignore_index=IGNORE,
                                 reduction='none').reshape_as(targets)
        reference = F.cross_entropy(flat_scores, flat_targets, ignore_index=IGNORE,
                                    reduction='sum')
        if not torch.isfinite(losses).all() or not torch.isfinite(reference):
            raise ValueError('Nonfinite answer-token cross entropy')
        values = losses.double().cpu().tolist()
        kinds = shifted_categories.cpu().tolist()

    def summarize(loss_rows, category_rows):
        buckets = {name: [] for name in CATEGORY_NAMES.values()}
        for row_values, row_kinds in zip(loss_rows, category_rows):
            for value, kind in zip(row_values, row_kinds):
                if kind != IGNORE:
                    buckets[CATEGORY_NAMES[kind]].append(value)
        result = {name: {'loss_sum': math.fsum(bucket), 'count': len(bucket)}
                  for name, bucket in buckets.items()}
        result['full'] = {'loss_sum': sum(result[name]['loss_sum'] for name in CATEGORY_NAMES.values()),
                          'count': sum(result[name]['count'] for name in CATEGORY_NAMES.values())}
        denominator = result['full']['count']
        for item in result.values():
            item['nll_nats_per_token'] = item['loss_sum'] / item['count'] if item['count'] else None
            item['contribution_nats_per_full_token'] = item['loss_sum'] / denominator if denominator else None
        return result

    result = summarize(values, kinds)
    result['per_sample'] = [summarize([row_values], [row_kinds])
                            for row_values, row_kinds in zip(values, kinds)]
    original = float(reference)
    result['reference_supervised_nll'] = {'loss_sum': original, 'count': count,
                                         'nll_nats_per_token': original / count}
    result['accounting'] = {'category_count_equals_full': result['full']['count'] == count,
                            'diagnostic_sum_minus_reference_fp32_sum': result['full']['loss_sum'] - original,
                            'rounding_policy': 'retain float32 per-token CE; no category correction',
                            'checkpoint_selection': 'unchanged legacy full-answer supervised_nll only'}
    result['units'] = {'loss_sum': 'nats', 'nll_nats_per_token': 'nats/category target token',
                       'contribution_nats_per_full_token': 'nats/full-answer target token'}
    return result

"""Actual independently executed formulas; no model loading or source mutation."""
from collections import Counter, defaultdict
import json
import math

def average(values): return math.fsum(values) / len(values)

def rank(values):
    counts = Counter(values)
    offset, mapping = 0, {}
    for value in sorted(counts):
        mapping[value] = offset + (counts[value] + 1) / 2
        offset += counts[value]
    return [mapping[value] for value in values]

def regression(gold, predicted):
    errors = [abs(a - b) for a, b in zip(gold, predicted)]
    x, y = rank(gold), rank(predicted)
    dx, dy = [v - average(x) for v in x], [v - average(y) for v in y]
    denominator = math.sqrt(math.fsum(v*v for v in dx) * math.fsum(v*v for v in dy))
    return {'mae': average(errors), 'rmse': math.sqrt(average([value*value for value in errors])),
        'spearman': math.fsum(a*b for a, b in zip(dx, dy)) / denominator if denominator else None}

def score(task, gold, predictions, candidates):
    expected = {row['sample_id'] for row in gold}
    supplied = {row['sample_id']: row for row in predictions}
    assert len(expected) == len(gold) and len(supplied) == len(predictions) and set(supplied) == expected
    if task == 'elm':
        per_class, per_row, recalls1, recalls3 = defaultdict(list), [], [], []
        for row in gold:
            ranking = supplied[row['sample_id']]['ranking']
            assert len(ranking) == len(candidates) and set(ranking) == set(candidates)
            positions = [ranking.index(label) + 1 for label in row['positive_ids']]
            reciprocals = [1.0 / value for value in positions]
            per_row.append(average(reciprocals))
            recalls1.append(sum(value <= 1 for value in positions) / len(positions))
            recalls3.append(sum(value <= 3 for value in positions) / len(positions))
            for label, value in zip(row['positive_ids'], reciprocals): per_class[label].append(value)
        return {'class_macro_mrr': average([average(values) for values in per_class.values()]),
            'mrr': average(per_row), 'recall_at_1': average(recalls1), 'recall_at_3': average(recalls3)}
    values = [supplied[row['sample_id']]['prediction'] for row in gold]
    assert all(type(value) in (int, float) and math.isfinite(value) for value in values)
    metrics = regression([row['target'] for row in gold], values)
    grouped = defaultdict(list)
    for row, value in zip(gold, values): grouped[row['group_id']].append((row['target'], value))
    per_group = [regression([pair[0] for pair in values], [pair[1] for pair in values]) for values in grouped.values()]
    for name in ('mae', 'rmse', 'spearman'):
        metrics['group_macro_' + name] = average([group[name] for group in per_group]) if all(group[name] is not None for group in per_group) else None
    return metrics

def close(a, b):
    return a is None and b is None or a is not None and b is not None and math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-10)

def parse(task, text):
    def pairs(values):
        if len(dict(values)) != len(values): raise ValueError('Duplicate field')
        return dict(values)
    def invalid(value): raise ValueError('Nonfinite JSON')
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
        field = {'meltome': 'melting_temperature_c', 'elm': 'function', 'rt': 'retention_time_min'}[task]
        if type(value) is not dict or set(value) != {field}: return None
        answer = value[field]
        if task == 'elm': return answer if type(answer) is str and answer else None
        return answer if type(answer) in (int, float) and math.isfinite(answer) else None
    except (ValueError, TypeError, RecursionError):
        return None


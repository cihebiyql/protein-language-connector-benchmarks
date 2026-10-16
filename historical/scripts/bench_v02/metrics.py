"""Inspectable v0.2 metrics; outputs are calculations, never benchmark claims.

Missing/nonfinite regression predictions must be accounted for by the caller's
generation validity report; they are rejected here rather than silently dropped.
Undefined statistics are None so outputs serialize as strict JSON.
"""
from collections import defaultdict
from collections.abc import Mapping

import numpy as np


def _numbers(values, name):
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a finite one-dimensional sequence')
    return array


def _same_length(*sequences):
    if len({len(sequence) for sequence in sequences}) > 1:
        raise ValueError('Inputs must have equal lengths')


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def _average_ranks(values):
    """Ranks start at one and tied values receive their average rank."""
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def _spearman(values, predictions):
    if len(values) < 2:
        return None
    x, y = _average_ranks(values), _average_ranks(predictions)
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.clip(np.dot(x, y) / denominator, -1, 1)) if denominator else None


def regression(values, predictions, group_ids=None):
    """Exact-label regression and independent-group macro MAE.

    This operator has no interpretation for censored ranges or mixed endpoints.
    Group order follows first occurrence, preserving inspectable group labels.
    """
    values, predictions = _numbers(values, 'values'), _numbers(predictions, 'predictions')
    ids = list(range(len(values))) if group_ids is None else list(group_ids)
    _same_length(values, predictions, ids)
    indices = defaultdict(list)
    for index, group_id in enumerate(ids):
        indices[group_id].append(index)
    groups = []
    for group_id, positions in indices.items():
        groups.append({'group_id': group_id, 'n': len(positions),
                       'mae': _mean(np.abs(values[positions] - predictions[positions])),
                       'spearman': _spearman(values[positions], predictions[positions])})
    return {'n': len(values), 'mae': _mean(np.abs(values - predictions)),
            'spearman': _spearman(values, predictions), 'n_groups': len(groups),
            'group_macro_mae': _mean([group['mae'] for group in groups]), 'groups': groups}


def grouped_regression(values, predictions, condition_ids, target_ids=None, group_ids=None):
    """Keep condition/target strata separate; do not pool different RT scales."""
    values, predictions = _numbers(values, 'values'), _numbers(predictions, 'predictions')
    conditions = list(condition_ids)
    targets = [None] * len(values) if target_ids is None else list(target_ids)
    groups = list(range(len(values))) if group_ids is None else list(group_ids)
    _same_length(values, predictions, conditions, targets, groups)
    positions = defaultdict(list)
    for index, key in enumerate(zip(conditions, targets)):
        positions[key].append(index)
    strata = []
    for (condition, target), rows in positions.items():
        score = regression(values[rows], predictions[rows], [groups[index] for index in rows])
        strata.append({'condition_id': condition, 'target_id': target, **score})
    return {'n': len(values), 'pooled': False, 'strata': strata}


def known_positive_retrieval(ranked_lists, positive_sets):
    """Known-positive ranking; unannotated classes are unknown, not negatives.

    Sample MRR uses the first known positive. Class macro MRR uses each true
    class's own rank, averages within that class, then averages across classes.
    A positive absent from the submitted ranked list has reciprocal rank zero.
    """
    rankings, positives = list(ranked_lists), list(positive_sets)
    _same_length(rankings, positives)
    per_sample, by_class = [], defaultdict(list)
    for ranking, positive in zip(rankings, positives):
        ranking, positive = list(ranking), set(positive)
        if len(set(ranking)) != len(ranking):
            raise ValueError('Duplicate class in ranked list')
        if not positive:
            raise ValueError('Each sample requires at least one known positive')
        positions = {class_id: index + 1 for index, class_id in enumerate(ranking)}
        reciprocal = [1 / positions[class_id] if class_id in positions else 0 for class_id in positive]
        for class_id in positive:
            rank = positions.get(class_id)
            by_class[class_id].append({'mrr': 1 / rank if rank is not None else 0,
                                       'recall_at_1': float(rank is not None and rank <= 1),
                                       'recall_at_3': float(rank is not None and rank <= 3)})
        per_sample.append({'mrr': max(reciprocal),
                           'recall_at_1': len(positive.intersection(ranking[:1])) / len(positive),
                           'recall_at_3': len(positive.intersection(ranking[:3])) / len(positive)})
    # Stable output independent of the iteration order of a set of positives.
    per_class = [{'class_id': class_id, 'n': len(by_class[class_id]),
                  **{metric: _mean([row[metric] for row in by_class[class_id]])
                     for metric in ('mrr', 'recall_at_1', 'recall_at_3')}}
                 for class_id in sorted(by_class, key=lambda item: (type(item).__name__, repr(item)))]
    return {'n': len(per_sample),
            **{key: _mean([row[key] for row in per_sample])
               for key in ('mrr', 'recall_at_1', 'recall_at_3')},
            'class_macro_mrr': _mean([row['mrr'] for row in per_class]),
            'class_macro_recall_at_1': _mean([row['recall_at_1'] for row in per_class]),
            'class_macro_recall_at_3': _mean([row['recall_at_3'] for row in per_class]),
            'n_classes': len(per_class), 'per_sample': per_sample, 'per_class': per_class}


def pair_direction(labels, predictions, group_ids=None):
    """Score adjudicated binary pair directions A/B; ambiguity is not a label.

    Balanced accuracy averages recalls of both required directions. It is
    undefined when the evaluation data contain only one direction, and the
    supported one-class recall is reported separately in per_class.
    """
    labels, predictions = list(labels), list(predictions)
    groups = list(range(len(labels))) if group_ids is None else list(group_ids)
    _same_length(labels, predictions, groups)
    if any(value not in ('A', 'B') for value in labels + predictions):
        raise ValueError('Pair directions must be adjudicated A/B labels; ambiguous/unknown are not labels')
    correct = [float(label == predicted) for label, predicted in zip(labels, predictions)]
    by_group = defaultdict(list)
    for group, is_correct in zip(groups, correct):
        by_group[group].append(is_correct)
    per_group = [{'group_id': group, 'n': len(scores), 'accuracy': _mean(scores)}
                 for group, scores in by_group.items()]
    per_class = []
    for direction in ('A', 'B'):
        scores = [is_correct for label, is_correct in zip(labels, correct) if label == direction]
        per_class.append({'label': direction, 'n': len(scores), 'recall': _mean(scores)})
    return {'n': len(labels), 'accuracy': _mean(correct),
            'balanced_accuracy': _mean([row['recall'] for row in per_class])
            if all(row['n'] for row in per_class) else None,
            'n_groups': len(per_group), 'group_macro_accuracy': _mean([row['accuracy'] for row in per_group]),
            'per_group': per_group, 'per_class': per_class}


def intervention_effects(correct, off, mismatch, larger_is_better=True):
    """Compute U and D after averaging donors inside each recipient.

    Scores must be sample-additive (e.g. per-record absolute error, not a
    pre-aggregated correlation). Empty donor lists preserve missing coverage.
    """
    correct, off = _numbers(correct, 'correct'), _numbers(off, 'off')
    mismatch = list(mismatch)
    _same_length(correct, off, mismatch)
    sign = 1 if larger_is_better else -1
    rows = []
    for index, donors in enumerate(mismatch):
        if donors is None:
            donors = []
        elif np.isscalar(donors):
            donors = [donors]
        donors = _numbers(donors, 'mismatch donors')
        donor_mean = _mean(donors)
        rows.append({'n_donors': len(donors), 'mismatch': donor_mean,
                     'u': float(sign * (correct[index] - off[index])),
                     'd': float(sign * (correct[index] - donor_mean)) if len(donors) else None})
    covered = [index for index, row in enumerate(rows) if row['n_donors']]
    return {'n': len(rows), 'n_mismatch_covered': len(covered),
            'u': _mean([row['u'] for row in rows]),
            'd': _mean([rows[index]['d'] for index in covered]),
            'u_common_coverage': _mean([rows[index]['u'] for index in covered]),
            'correct': _mean(correct), 'off': _mean(off),
            'mismatch': _mean([rows[index]['mismatch'] for index in covered]),
            'score_direction': 'larger' if larger_is_better else 'smaller', 'per_recipient': rows}


def atomic_facts(claims, required_fact_ids):
    """Score adjudicated atomic claims with four distinct evidence statuses.

    A claim is {'status': ..., 'fact_id': ...}; fact_id is needed for coverage.
    Repeated facts do not increase required coverage. Empty precision is None.
    """
    counts = dict.fromkeys(('supported', 'contradicted', 'unsupported', 'unknown'), 0)
    required = set(required_fact_ids)
    covered = set()
    for claim in claims:
        if not isinstance(claim, Mapping) or claim.get('status') not in counts:
            raise ValueError('Claim status must be supported/contradicted/unsupported/unknown')
        status = claim['status']
        counts[status] += 1
        if status == 'supported' and claim.get('fact_id') in required:
            covered.add(claim['fact_id'])
    n = sum(counts.values())
    return {'n_claims': n, 'counts': counts,
            'supported_precision': counts['supported'] / n if n else None,
            'required_coverage': len(covered) / len(required) if required else 0.0,
            'n_required': len(required), 'n_required_covered': len(covered),
            'contradiction_rate': counts['contradicted'] / n if n else None,
            'unsupported_rate': counts['unsupported'] / n if n else None,
            'unknown_rate': counts['unknown'] / n if n else None,
            'empty_answer': n == 0}


def _booleans(values, name):
    values = list(values)
    if any(not isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError(f'{name} must contain correctness booleans')
    return values


def causal_fact_intervention(target_before, target_after_expected_correct,
                             irrelevant_before, irrelevant_after):
    """Internal computational intervention, not a biological causality claim.

    Target success is measured among originally correct target facts. Unrelated
    retention compares submitted fact values before/after; caller must define
    the unrelated fact set before test and adjudicate its correctness separately.
    """
    before = _booleans(target_before, 'target_before')
    after = _booleans(target_after_expected_correct, 'target_after_expected_correct')
    _same_length(before, after)
    if isinstance(irrelevant_before, Mapping) or isinstance(irrelevant_after, Mapping):
        if not isinstance(irrelevant_before, Mapping) or not isinstance(irrelevant_after, Mapping):
            raise ValueError('Unrelated facts must both be mappings or sequences')
        if set(irrelevant_before) != set(irrelevant_after):
            raise ValueError('Unrelated fact IDs must match')
        irrelevant_pairs = [(value, irrelevant_after[key]) for key, value in irrelevant_before.items()]
    else:
        irrelevant_before, irrelevant_after = list(irrelevant_before), list(irrelevant_after)
        _same_length(irrelevant_before, irrelevant_after)
        irrelevant_pairs = list(zip(irrelevant_before, irrelevant_after))
    eligible = sum(before)
    successes = sum(old and new for old, new in zip(before, after))
    retained = sum(old == new for old, new in irrelevant_pairs)
    return {'n_target_facts': len(before), 'n_target_eligible': eligible,
            'n_target_success': successes, 'target_intervention_success': successes / eligible if eligible else None,
            'target_after_expected_accuracy': _mean(after),
            'n_irrelevant_facts': len(irrelevant_pairs), 'n_irrelevant_retained': retained,
            'irrelevant_fact_retention': retained / len(irrelevant_pairs) if irrelevant_pairs else None}


def _probability_difference(full_scores, other_scores):
    full, other = _numbers(full_scores, 'full_scores'), _numbers(other_scores, 'other_scores')
    _same_length(full, other)
    if np.any((full < 0) | (full > 1)) or np.any((other < 0) | (other > 1)):
        raise ValueError('Scores must be probabilities in [0, 1] of the same original decision')
    differences = (full - other).tolist()
    return {'n': len(full), 'mean': _mean(differences), 'per_sample': differences}


def sufficiency(full_scores, rationale_scores):
    """p(original decision | full) − p(original decision | rationale); lower is better."""
    return _probability_difference(full_scores, rationale_scores)


def comprehensiveness(full_scores, removed_scores):
    """p(original decision | full) − p(original decision | rationale removed); higher is better."""
    return _probability_difference(full_scores, removed_scores)

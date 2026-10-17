"""Model-free fixed-population metrics; missing predictions never improve scores."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from statistics import mean

REGRESSION_FIELDS = {"meltome": "melting_temperature_c", "rt": "retention_time_min"}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _align(rows, predictions):
    rows, predictions = list(rows), list(predictions)
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate reference sample_id")
    supplied = {}
    for prediction in predictions:
        sample_id = prediction["sample_id"]
        if sample_id in supplied:
            raise ValueError("duplicate prediction sample_id")
        if sample_id not in set(ids):
            raise ValueError("unknown prediction sample_id")
        supplied[sample_id] = prediction
    return rows, [supplied.get(sample_id) for sample_id in ids]


def _coverage(n, valid):
    return {"n_total": n, "n_valid": valid, "n_failed": n - valid,
            "fraction": valid / n if n else None}


def _ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for i in order[start:end]:
            ranks[i] = (start + 1 + end) / 2
        start = end
    return ranks


def _spearman(actual, predicted):
    if len(actual) < 2:
        return None
    x, y = _ranks(actual), _ranks(predicted)
    mx, my = mean(x), mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else None


def _regression(actual, predicted):
    if not actual:
        return {"mae": None, "rmse": None, "spearman": None}
    return {"mae": mean(abs(a - p) for a, p in zip(actual, predicted)),
            "rmse": math.sqrt(mean((a - p) ** 2 for a, p in zip(actual, predicted))),
            "spearman": _spearman(actual, predicted)}


def score_regression(rows, predictions):
    rows, aligned = _align(rows, predictions)
    if any(not _finite(r.get("target")) for r in rows):
        raise ValueError("reference target must be a finite non-boolean number")
    valid = [p is not None and _finite(p.get("prediction")) for p in aligned]
    complete = bool(rows) and all(valid)
    result = {"coverage": _coverage(len(rows), sum(valid)), "complete": complete,
              "mae": None, "rmse": None, "spearman": None,
              "group_macro_mae": None, "group_macro_rmse": None,
              "group_macro_spearman": None, "groups": [],
              "group_spearman_policy": "null if any observed group has undefined Spearman"}
    if not complete:
        return result
    result.update(_regression([r["target"] for r in rows], [p["prediction"] for p in aligned]))
    grouped = defaultdict(list)
    for row, prediction in zip(rows, aligned):
        grouped[row["group_id"]].append((row["target"], prediction["prediction"]))
    for group_id in sorted(grouped, key=str):
        pairs = grouped[group_id]
        result["groups"].append({"group_id": group_id, "n": len(pairs),
                                 **_regression([x[0] for x in pairs], [x[1] for x in pairs])})
    for metric in ("mae", "rmse", "spearman"):
        values = [g[metric] for g in result["groups"]]
        result["group_macro_" + metric] = mean(values) if values and all(v is not None for v in values) else None
    return result


def score_ranking(rows, predictions, candidate_ids):
    rows, aligned = _align(rows, predictions)
    candidate_ids = list(candidate_ids)
    universe = set(candidate_ids)
    if not candidate_ids or len(universe) != len(candidate_ids):
        raise ValueError("candidate catalog must be nonempty and unique")
    for row in rows:
        positives = row.get("positive_ids")
        if not isinstance(positives, list) or not positives or len(set(positives)) != len(positives) or not set(positives) <= universe:
            raise ValueError("reference positives must be unique known catalog IDs")
    def valid(prediction):
        ranking = prediction.get("ranking") if prediction else None
        return (isinstance(ranking, list) and len(ranking) == len(candidate_ids)
                and all(isinstance(x, str) for x in ranking) and len(set(ranking)) == len(ranking)
                and set(ranking) == universe)
    flags = [valid(p) for p in aligned]
    complete = bool(rows) and all(flags)
    result = {"coverage": _coverage(len(rows), sum(flags)), "complete": complete,
              "class_macro_mrr": None, "mrr": None, "recall_at_1": None,
              "recall_at_3": None, "classes": [],
              "interpretation": "Known-positive ranking; unannotated candidates are not experimental negatives"}
    if not complete:
        return result
    per_sample, classes = [], defaultdict(list)
    for row, prediction in zip(rows, aligned):
        ranking = prediction["ranking"]
        positions = {candidate_id: i + 1 for i, candidate_id in enumerate(ranking)}
        positive = row["positive_ids"]
        reciprocal = [1 / positions[c] for c in positive]
        per_sample.append({"mrr": mean(reciprocal),
                           "recall_at_1": sum(positions[c] <= 1 for c in positive) / len(positive),
                           "recall_at_3": sum(positions[c] <= 3 for c in positive) / len(positive)})
        for candidate_id, value in zip(positive, reciprocal):
            classes[candidate_id].append(value)
    for key in ("mrr", "recall_at_1", "recall_at_3"):
        result[key] = mean(p[key] for p in per_sample)
    result["classes"] = [{"candidate_id": c, "n_annotations": len(values), "mrr": mean(values)}
                         for c, values in sorted(classes.items())]
    result["class_macro_mrr"] = mean(c["mrr"] for c in result["classes"])
    return result


def parse_task_answer(task, text):
    if task not in (*REGRESSION_FIELDS, "elm"):
        raise ValueError("unknown task")
    if not isinstance(text, str):
        return None
    def unique(pairs):
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError("duplicate JSON field")
        return dict(pairs)
    try:
        value = json.loads(text, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (ValueError, TypeError, RecursionError):
        return None
    field = "function" if task == "elm" else REGRESSION_FIELDS[task]
    if not isinstance(value, dict) or set(value) != {field}:
        return None
    answer = value[field]
    if task == "elm":
        return answer if isinstance(answer, str) and bool(answer) else None
    return float(answer) if _finite(answer) else None


def score_language(task, rows, predictions, catalog=None):
    rows, aligned = _align(rows, predictions)
    parsed = [parse_task_answer(task, p.get("text")) if p else None for p in aligned]
    if task in REGRESSION_FIELDS:
        result = score_regression(rows, [{"sample_id": r["sample_id"], "prediction": p}
                                         for r, p in zip(rows, parsed)])
        result["channel"] = "strict_language_json"
        return result
    if task != "elm":
        raise ValueError("unknown task")
    if not catalog:
        raise ValueError("ELM language scoring requires frozen catalog")
    description_to_ids = defaultdict(set)
    ids = set()
    for candidate in catalog:
        if candidate["candidate_id"] in ids:
            raise ValueError("duplicate catalog ID")
        ids.add(candidate["candidate_id"])
        description_to_ids[candidate["description"]].add(candidate["candidate_id"])
    for row in rows:
        if not row.get("positive_ids") or not set(row["positive_ids"]) <= ids:
            raise ValueError("unknown reference positive")
    recognized = [p is not None and p in description_to_ids for p in parsed]
    correct = [bool(description_to_ids.get(p, set()) & set(r["positive_ids"]))
               for r, p in zip(rows, parsed)]
    return {"channel": "strict_language_json", "coverage": _coverage(len(rows), sum(recognized)),
            "json_coverage": _coverage(len(rows), sum(p is not None for p in parsed)),
            "exact_description_known_positive_accuracy": sum(correct) / len(rows) if rows else None,
            "n_correct": sum(correct), "n_unrecognized_description": sum(p is not None and p not in description_to_ids for p in parsed),
            "denominator": "all frozen test rows, including invalid and missing outputs",
            "interpretation": "Exact canonical-description known-positive match; no BLEU or free-text semantic grading"}


def calibration_radius(rows, predictions, alpha=0.1):
    if not _finite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    rows, aligned = _align(rows, predictions)
    if any(r.get("split", "calibration") != "calibration" for r in rows):
        raise ValueError("interval radius may use calibration rows only")
    if any(not _finite(r.get("target")) for r in rows):
        raise ValueError("invalid calibration target")
    complete = bool(rows) and all(p is not None and _finite(p.get("prediction")) for p in aligned)
    result = {"radius": None, "alpha": alpha, "calibration_groups": 0, "quantile_order": None,
              "complete": complete, "unbounded": False,
              "method": "ceil((G+1)*(1-alpha)) order statistic of calibration group-max absolute residual",
              "interpretation": "Report empirical same-condition group coverage and interval width; no shifted-distribution guarantee"}
    if not complete:
        return result
    groups = defaultdict(list)
    for row, prediction in zip(rows, aligned):
        groups[row["group_id"]].append(abs(row["target"] - prediction["prediction"]))
    scores = sorted(max(values) for values in groups.values())
    order = math.ceil((len(scores) + 1) * (1 - alpha))
    result.update(calibration_groups=len(scores), quantile_order=order, unbounded=order > len(scores))
    if order <= len(scores):
        result["radius"] = scores[order - 1]
    return result


def score_intervals(calibration_rows, calibration_predictions, test_rows, test_predictions, alpha=0.1):
    calibration_rows, test_rows = list(calibration_rows), list(test_rows)
    if {r["sample_id"] for r in calibration_rows} & {r["sample_id"] for r in test_rows}:
        raise ValueError("calibration and test sample IDs overlap")
    if any(r.get("split", "test") != "test" for r in test_rows):
        raise ValueError("empirical interval coverage requires frozen test rows")
    result = calibration_radius(calibration_rows, calibration_predictions, alpha)
    test_rows, aligned = _align(test_rows, test_predictions)
    valid = [p is not None and _finite(p.get("prediction")) for p in aligned]
    result.update(test_coverage=_coverage(len(test_rows), sum(valid)), empirical_group_coverage=None,
                  empirical_sample_coverage=None, interval_width=None, test_groups=0)
    if not result["complete"] or result["radius"] is None or not test_rows or not all(valid):
        return result
    groups = defaultdict(list)
    hits = []
    for row, prediction in zip(test_rows, aligned):
        if not _finite(row.get("target")):
            raise ValueError("invalid interval test target")
        hit = abs(row["target"] - prediction["prediction"]) <= result["radius"]
        hits.append(hit)
        groups[row["group_id"]].append(hit)
    result.update(empirical_group_coverage=mean(all(values) for values in groups.values()),
                  empirical_sample_coverage=mean(hits), interval_width=2 * result["radius"], test_groups=len(groups))
    return result


def build_metrics(task, rows, readout, language=None, c1=None, catalog=None):
    """Shared score schema. C1 is the same checkpoint with its connector off."""
    rows = list(rows)
    if task == "elm":
        if not catalog:
            raise ValueError("ELM requires frozen catalog")
        scorer = lambda predictions: score_ranking(rows, predictions, [c["candidate_id"] for c in catalog])
        primary = "class_macro_mrr"
        on_key, off_key = "ranking", "ranking_off"
    elif task in REGRESSION_FIELDS:
        scorer = lambda predictions: score_regression(rows, predictions)
        primary = "spearman" if task == "meltome" else "mae"
        on_key, off_key = "prediction", "prediction_off"
    else:
        raise ValueError("unknown task")
    readout = list(readout)
    result = {"readout": scorer(readout),
              "language": score_language(task, rows, language, catalog) if language is not None else None,
              "C1_off": None, "C1_off_minus_on": None}
    if c1 is not None:
        # Validate C1 on predictions too, so an artifact cannot substitute a
        # different on branch while retaining a plausible off branch.
        c1 = list(c1)
        _, expected = _align(rows, readout)
        _, observed = _align(rows, c1)
        if any((a or {}).get(on_key) != (b or {}).get(on_key) for a, b in zip(expected, observed)):
            raise ValueError("C1 on predictions differ from selected readout")
        on = scorer([{"sample_id": p["sample_id"], on_key: p.get(on_key)} for p in c1])
        if on != result["readout"]:
            raise ValueError("C1 on predictions differ from selected readout")
        result["C1_off"] = scorer([{"sample_id": p["sample_id"], on_key: p.get(off_key)} for p in c1])
        a, b = result["readout"][primary], result["C1_off"][primary]
        result["C1_off_minus_on"] = b - a if a is not None and b is not None else None
    return result

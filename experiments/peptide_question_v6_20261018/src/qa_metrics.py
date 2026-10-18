"""Frozen generated-answer scoring for paired RT and CCS peptide questions.

Only whitespace is stripped. Missing and malformed answers stay in every
denominator. The conjunction question is composition, not a third property.
This module deliberately has no model or tensor dependency.
"""
from collections.abc import Mapping
import math


QUESTIONS = ("rt", "ccs", "both")


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _thresholds(thresholds):
    if not isinstance(thresholds, Mapping):
        raise ValueError("thresholds must map rt and ccs to finite numbers")
    return {name: _finite_number(thresholds.get(name), f"threshold {name}") for name in ("rt", "ccs")}


def label(row, question, thresholds):
    """Return the fixed strict-greater-than answer; equality is ``no``."""
    if question not in QUESTIONS:
        raise ValueError("unknown question")
    bounds = _thresholds(thresholds)
    answers = {name: _finite_number(row.get(name), name) > bounds[name] for name in ("rt", "ccs")}
    result = all(answers.values()) if question == "both" else answers[question]
    return "yes" if result else "no"


def parse_answer(raw):
    """Accept exactly lowercase yes/no after stripping whitespace, without repair."""
    if not isinstance(raw, str):
        return None
    answer = raw.strip()
    return answer if answer in ("yes", "no") else None


def _question_metrics(items):
    # Matrix rows are truth; columns are generated no/yes/invalid.
    confusion = {truth: {predicted: 0 for predicted in ("no", "yes", "invalid")}
                 for truth in ("no", "yes")}
    for truth, predicted in items:
        confusion[truth][predicted if predicted is not None else "invalid"] += 1
    counts = {truth: sum(confusion[truth].values()) for truth in confusion}
    recalls = {truth: confusion[truth][truth] / count if count else None
               for truth, count in counts.items()}
    total = sum(counts.values())
    correct = sum(confusion[truth][truth] for truth in confusion)
    valid = total - sum(confusion[truth]["invalid"] for truth in confusion)
    balanced = (recalls["no"] + recalls["yes"]) / 2 if all(counts.values()) else None
    majority = "yes" if counts["yes"] > counts["no"] else "no"
    return {
        "count": total, "correct": correct, "valid": valid, "invalid": total - valid,
        "accuracy": correct / total, "balanced_accuracy": balanced,
        "valid_rate": valid / total, "class_counts": counts,
        "class_recall": recalls, "confusion": confusion,
        "majority_baseline": {"label": majority, "accuracy": counts[majority] / total,
                              "balanced_accuracy": 0.5 if all(counts.values()) else None},
    }


def score(rows, predictions, thresholds):
    """Score one fixed peptide panel from actual generated strings.

    Rows contain id, rt, ccs and optionally group_id. Predictions contain id,
    question and raw. Unexpected or duplicate keys indicate corrupt alignment
    and raise. Missing expected keys are counted as invalid, never dropped.
    Balanced accuracy is null if a panel contains only one class for a question.
    """
    bounds = _thresholds(thresholds)
    rows, predictions = list(rows), list(predictions)
    if not rows:
        raise ValueError("cannot score an empty peptide panel")
    by_id = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("every peptide row needs a nonempty string id")
        if row["id"] in by_id:
            raise ValueError("duplicate peptide id")
        for name in ("rt", "ccs"):
            _finite_number(row.get(name), name)
        by_id[row["id"]] = row
    indexed = {}
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            raise ValueError("prediction must be a mapping")
        identifier, question = prediction.get("id"), prediction.get("question")
        if not isinstance(identifier, str) or identifier not in by_id or question not in QUESTIONS:
            raise ValueError("unexpected prediction id or question")
        key = (identifier, question)
        if key in indexed:
            raise ValueError("duplicate prediction id/question")
        indexed[key] = parse_answer(prediction.get("raw"))
    items = {question: [] for question in QUESTIONS}
    per_peptide = []
    for row in rows:
        identifier = row["id"]
        truths, answers, correct = {}, {}, {}
        for question in QUESTIONS:
            truths[question] = label(row, question, bounds)
            answers[question] = indexed.get((identifier, question))
            correct[question] = truths[question] == answers[question]
            items[question].append((truths[question], answers[question]))
        per_peptide.append({"id": identifier, "group_id": row.get("group_id", identifier),
                            "truth": truths, "prediction": answers, "correct": correct,
                            "all_correct": all(correct.values())})
    questions = {question: _question_metrics(values) for question, values in items.items()}
    properties = [questions[question]["balanced_accuracy"] for question in ("rt", "ccs")]
    primary = sum(properties) / 2 if all(value is not None for value in properties) else None
    total = len(rows) * len(QUESTIONS)
    joint_correct = sum(item["all_correct"] for item in per_peptide)
    return {
        "schema": "peptide-qa-generated-score.v1", "thresholds": bounds,
        "primary": primary, "primary_name": "mean_rt_ccs_balanced_accuracy",
        "questions": questions, "peptide_count": len(rows),
        "expected_prediction_count": total, "received_prediction_count": len(indexed),
        "missing_prediction_count": total - len(indexed),
        "valid_rate": sum(value["valid"] for value in questions.values()) / total,
        "joint_all_correct": joint_correct / len(rows), "joint_all_correct_count": joint_correct,
        "composition_question": "both", "independent_properties": ["rt", "ccs"],
        "per_peptide": per_peptide,
    }

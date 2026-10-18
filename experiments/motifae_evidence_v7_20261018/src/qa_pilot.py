"""Frozen zero-shot Qwen evidence-card pilot; per-query inputs and answers stay private."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

ARMS = ("sequence", "regex", "raw_probe", "motifae", "motifae_shuffled", "template")
QUESTIONS = ("class", "position", "activity", "evidence")
CARD_PATHS = {arm: f"data/cards_{arm}.jsonl" for arm in ("regex", "raw_probe", "motifae")}
QUESTION_TEXT = {
    "class": 'What is the strongest supported motif class? Return exactly {"class_id":"<catalog ID or unknown>"}.',
    "position": 'Where is the strongest supported motif span in this short peptide, using 1-based inclusive coordinates? Return exactly {"start":<integer or null>,"end":<integer or null>}.',
    "activity": 'Do these supplied data establish experimentally measured activity of this isolated short peptide? Return exactly {"activity":"established"} or {"activity":"not_established"}.',
    "evidence": 'Which supplied evidence record supports the strongest motif association? Return exactly {"evidence_id":"E<number>"} or {"evidence_id":null}.',
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evidence_card(source: dict, sequence_length: int | None = None) -> dict:
    """Whitelist prediction fields; reject accidental annotation/identity propagation."""
    length = source["sequence_length"] if sequence_length is None else sequence_length
    associations = []
    allowed = ("class_id", "description", "feature_id", "start", "end", "peak_position",
               "peak", "threshold", "normalized_score", "association_evidence", "activity_claim")
    for index, item in enumerate(source.get("associations", []), 1):
        association = {key: item[key] for key in allowed if key in item}
        association["evidence_id"] = f"E{index}"
        associations.append(association)
    unknown = [{key: feature[key] for key in ("feature_id", "peak", "position") if key in feature}
               for feature in source.get("unknown_active_top_features", [])]
    return {"sequence_length": length, "coordinate_system": "1-based-inclusive",
            "associations": associations, "unknown_active_top_features": unknown,
            "activity_claim": "not established"}


def derangement(sample_ids: list[str], seed: int = 991) -> dict[str, str]:
    ids = sorted(sample_ids)
    if len(ids) < 2:
        raise ValueError("Wrong-card control requires at least two peptides")
    random.Random(seed).shuffle(ids)
    return {sample_id: ids[(i + 1) % len(ids)] for i, sample_id in enumerate(ids)}


def make_messages(question: str, catalog: list[dict], *, sequence=None, card=None):
    if question not in QUESTIONS or (sequence is None) == (card is None):
        raise ValueError("One question and exactly one input channel required")
    vocabulary = [{"class_id": item["class_id"], "description": item["description"]} for item in catalog]
    system = (
        "Answer only the requested question as one strict JSON object, without explanation or markdown. "
        "Use only the supplied input and candidate vocabulary. Evidence associations are ranked in supplied order, "
        "with E1 strongest. An unnamed feature does not imply a known class. Use unknown/null when unsupported. "
        "The vocabulary alone is not evidence. Positions are 1-based inclusive within the short peptide. "
        "Parent-protein annotations and model/regex associations do not establish experimental activity of an "
        "isolated short peptide; unannotated positions are not proven biological negatives. "
        "Do not invent experimental validation or evidence record identifiers."
    )
    payload = {"candidate_vocabulary": vocabulary, "question": QUESTION_TEXT[question]}
    if sequence is not None:
        payload["input"] = {"short_sequence": sequence, "sequence_length": len(sequence)}
    else:
        payload["input"] = evidence_card(card)
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)}]


def parse_answer(raw: str, question: str, sequence_length: int, class_ids: set[str], card: dict | None):
    """Strict JSON: no extraction, key repair, numeric coercion, or valid-row filtering."""
    try:
        def unique_pairs(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError("duplicate JSON key")
                obj[key] = value
            return obj
        obj = json.loads(raw, object_pairs_hook=unique_pairs,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if not isinstance(obj, dict):
            raise ValueError("not an object")
        keys = {"class": {"class_id"}, "position": {"start", "end"},
                "activity": {"activity"}, "evidence": {"evidence_id"}}[question]
        if set(obj) != keys:
            raise ValueError("unexpected or missing keys")
        if question == "class":
            value = obj["class_id"]
            if not isinstance(value, str) or value not in class_ids | {"unknown"}:
                raise ValueError("class outside catalog")
        elif question == "position":
            start, end = obj["start"], obj["end"]
            if not (start is None and end is None):
                if type(start) is not int or type(end) is not int or not (1 <= start <= end <= sequence_length):
                    raise ValueError("position outside peptide")
        elif question == "activity":
            if obj["activity"] not in ("established", "not_established"):
                raise ValueError("invalid activity value")
        else:
            value = obj["evidence_id"]
            allowed = {a["evidence_id"] for a in (card or {}).get("associations", [])}
            if value is not None and (not isinstance(value, str) or value not in allowed):
                raise ValueError("invalid evidence ID")
        return obj, None
    except (ValueError, TypeError, KeyError) as exc:
        return None, str(exc)


def template_answer(question: str, card: dict):
    top = card.get("associations", [])
    top = top[0] if top else None
    if question == "class":
        return {"class_id": top["class_id"] if top else "unknown"}
    if question == "position":
        return {"start": top["start"] if top else None, "end": top["end"] if top else None}
    if question == "activity":
        return {"activity": "not_established"}
    return {"evidence_id": top["evidence_id"] if top else None}


def _iou(a, b):
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    return overlap / (max(a[1], b[1]) - min(a[0], b[0]) + 1)


def evaluate(records: list[dict], samples: dict[str, dict], cards_by_arm: dict[str, dict]):
    details, metrics = [], {}
    for arm in ARMS:
        per_question = {}
        for question in QUESTIONS:
            items = [r for r in records if r["arm"] == arm and r["question"] == question]
            counts = Counter(); ious = 0.0
            for record in items:
                sample = samples[record["sample_id"]]
                parsed = record["parsed"]
                card = cards_by_arm.get(arm, {}).get(record["sample_id"])
                associations = (card or {}).get("associations", [])
                spans = [s for s in sample.get("known_positive_spans", []) if s["fully_contained"]]
                gold_classes = {s["class_id"] for s in spans} or {sample["class_id"]}
                gold_spans = [(s["short_start"], s["short_end"]) for s in spans]
                gold_spans = gold_spans or [(sample["short_motif_start"], sample["short_motif_end"])]
                outcome = {"sample_id": record["sample_id"], "arm": arm, "question": question,
                           "valid": parsed is not None}
                counts["valid"] += parsed is not None
                if parsed is None:
                    outcome["error"] = record.get("parse_error")
                elif question == "class":
                    recovered = parsed["class_id"] in gold_classes
                    supported = parsed["class_id"] in {a["class_id"] for a in associations} if card is not None else None
                    counts["known_positive_recovered"] += recovered
                    if card is not None:
                        counts["card_supported_nonunknown"] += supported
                        counts["unsupported_nonunknown"] += parsed["class_id"] != "unknown" and not supported
                    outcome.update(known_positive_recovered=recovered, card_supported_nonunknown=supported)
                elif question == "position":
                    if parsed["start"] is not None:
                        pred = (parsed["start"], parsed["end"])
                        best = max(_iou(pred, gold) for gold in gold_spans)
                        anchor = _iou(pred, (sample["short_motif_start"], sample["short_motif_end"]))
                        contained = any(a["start"] <= pred[0] <= pred[1] <= a["end"] for a in associations) if card is not None else None
                        ious += best
                        counts["known_positive_overlap"] += best > 0
                        counts["anchor_overlap"] += anchor > 0
                        if card is not None:
                            counts["card_contained_span"] += contained
                        outcome.update(best_known_positive_iou=best, anchor_iou=anchor, card_contained_span=contained)
                elif question == "activity":
                    counts["activity_overclaim"] += parsed["activity"] == "established"
                    counts["correct_not_established"] += parsed["activity"] == "not_established"
                else:
                    cited = parsed["evidence_id"]
                    expected = associations[0]["evidence_id"] if associations else None
                    counts["strongest_record_or_correct_abstention"] += cited == expected
                    counts["valid_nonnull_citation"] += cited is not None
                details.append(outcome)
            denominator = len(samples)
            row = {"expected": denominator, "observed": len(items), "valid": counts["valid"],
                   "invalid": len(items) - counts["valid"], "valid_rate": counts["valid"] / denominator,
                   "card_faithfulness_applicable": arm != "sequence"}
            for key, value in counts.items():
                if key != "valid":
                    row[key + "_count"] = value
                    row[key + "_rate"] = value / denominator
            if question == "position":
                row["mean_best_known_positive_iou_full_denominator"] = ious / denominator
            per_question[question] = row
        metrics[arm] = per_question
    return {"schema_version": 1, "test_peptides": len(samples), "arms": metrics,
            "denominator_policy": "All test peptides; invalid/missing answers never dropped or repaired.",
            "position_policy": "Any fully contained known-positive span; anchor overlap separately reported.",
            "scope": "Exploratory known-positive annotation transfer and evidence faithfulness; not isolated activity validation.",
            "template_is_not_qwen": True,
            "class_recovery_is_not_specificity": True}, details


def verify_freeze(scope: Path):
    frozen_path = scope / "frozen_manifest.json"
    if not frozen_path.is_file():
        raise RuntimeError("Formal QA requires frozen_manifest.json")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    pins = dict(frozen.get("source_pins", {}))
    for item in frozen.get("files", []):
        pins[item["path"]] = item["sha256"]
    required = {"config.json", "src/qa_pilot.py", "data/panel.jsonl", "data/catalog.json"}
    if not required <= set(pins):
        raise RuntimeError("Frozen manifest missing required QA input/source pins")
    for rel, pin in pins.items():
        expected = pin["sha256"] if isinstance(pin, dict) else pin
        path = (scope / rel).resolve()
        if not path.is_relative_to(scope.resolve()) or path.is_symlink() or sha(path) != expected:
            raise RuntimeError(f"Frozen source mismatch: {rel}")
    dictionary_hash = (scope / "dictionary.sha256").read_text(encoding="utf-8").strip().split()[0]
    if sha(scope / "dictionary.json") != dictionary_hash:
        raise RuntimeError("Frozen dictionary fingerprint mismatch")
    contract_paths = ["frozen_manifest.json", "config.json", "src/qa_pilot.py", "data/panel.jsonl",
                      "data/catalog.json", "dictionary.json", "dictionary.sha256", *CARD_PATHS.values()]
    return {"schema_version": 1, "private": True, "dictionary_sha256": dictionary_hash,
            "pins": {rel: sha(scope / rel) for rel in contract_paths},
            "frozen_before_test_qa": True, "train_qwen": False}


class Qwen:
    def __init__(self, config):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(2)
        torch.manual_seed(991)
        torch.cuda.manual_seed_all(991)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError("Formal Qwen requires launcher-selected CUDA GPU")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(config["qwen_path"], local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            config["qwen_path"], local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").to("cuda").eval()
        self.model.requires_grad_(False)
        self.max_new_tokens = config["qa"]["max_new_tokens"]

    def generate(self, messages):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                                         use_cache=True, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run_smoke(scope: Path, generator=None):
    config = json.loads((scope / "config.json").read_text(encoding="utf-8"))
    generator = generator or Qwen(config)
    catalog = [{"class_id": "synthetic_example", "description": "Synthetic formatting-only association"}]
    card = evidence_card({"sequence_length": 12, "associations": [
        {"class_id": "synthetic_example", "description": "Synthetic formatting only", "start": 3, "end": 6,
         "feature_id": "synthetic", "peak": 1.0, "threshold": .5}], "unknown_active_top_features": []})
    records = []
    for question in QUESTIONS:
        messages = make_messages(question, catalog, card=card)
        raw = generator.generate(messages)
        parsed, error = parse_answer(raw, question, 12, {"synthetic_example"}, card)
        records.append({"question": question, "raw": raw, "parsed": parsed, "parse_error": error})
    receipt = {"phase": "synthetic_format_smoke", "reads_real_test": False, "score_gate": False,
               "records": records, "model_completed_four_generations": len(records) == 4}
    output = scope / "runs/qa/smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(dump(receipt))
    return receipt


def run_formal(scope: Path, generator=None, resume=False):
    contract = verify_freeze(scope)
    run_dir = scope / "runs/qa"
    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / "run_contract.json"
    raw_path = run_dir / "raw_answers.jsonl"
    if resume:
        if not contract_path.exists() or json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("Technical resume requires exactly identical source/card/dictionary hashes")
    else:
        if raw_path.exists() or contract_path.exists():
            raise RuntimeError("QA outputs already exist; use explicit --resume only for identical technical recovery")
        with contract_path.open("x", encoding="utf-8") as stream:
            stream.write(dump(contract))
    # Contract and frozen dictionary are recorded before reading any TEST examples.
    samples = {r["sample_id"]: r for r in read_rows(scope / "data/panel.jsonl") if r["split"] == "test"}
    if len(samples) != 67:
        raise RuntimeError("Frozen protocol expects all 67 TEST peptides")
    catalog = json.loads((scope / "data/catalog.json").read_text(encoding="utf-8"))["classes"]
    class_ids = {r["class_id"] for r in catalog}
    originals = {arm: {r["sample_id"]: r for r in read_rows(scope / rel)} for arm, rel in CARD_PATHS.items()}
    if any(set(items) != set(samples) for items in originals.values()):
        raise RuntimeError("Each card arm must contain exactly the full TEST set")
    config = json.loads((scope / "config.json").read_text(encoding="utf-8"))
    if tuple(config["qa"]["arms"]) != ARMS or tuple(config["qa"]["questions"]) != QUESTIONS:
        raise RuntimeError("Frozen QA arm or question order differs")
    donors = derangement(list(samples), config["qa"]["shuffle_seed"])
    cards = {arm: {sid: evidence_card(card) for sid, card in items.items()} for arm, items in originals.items()}
    cards["motifae_shuffled"] = {sid: evidence_card(originals["motifae"][donors[sid]], len(samples[sid]["short_sequence"]))
                                 for sid in samples}
    cards["template"] = cards["motifae"]
    existing = read_rows(raw_path) if raw_path.exists() else []
    done = set()
    for record in existing:
        key = (record["sample_id"], record["arm"], record["question"])
        if key in done or key[0] not in samples or key[1] not in ARMS or key[2] not in QUESTIONS:
            raise RuntimeError("Invalid/duplicate existing technical resume record")
        sample = samples[key[0]]
        card = cards.get(key[1], {}).get(key[0])
        parsed, error = parse_answer(record["raw"], key[2], len(sample["short_sequence"]), class_ids, card)
        if parsed != record["parsed"] or error != record.get("parse_error"):
            raise RuntimeError("Existing raw answer differs from its stored strict parse")
        expected_donor = donors[key[0]] if key[1] == "motifae_shuffled" else None
        if record.get("donor_sample_id") != expected_donor:
            raise RuntimeError("Existing wrong-card donor differs")
        if key[1] != "template":
            messages = make_messages(key[2], catalog,
                                     sequence=sample["short_sequence"] if key[1] == "sequence" else None, card=card)
            expected_prompt = hashlib.sha256(dump(messages).encode()).hexdigest()
            if record.get("prompt_sha256") != expected_prompt:
                raise RuntimeError("Existing prompt differs from frozen inputs")
        done.add(key)
    generator = generator or Qwen(config)
    records = list(existing)
    with raw_path.open("a" if resume else "x", encoding="utf-8") as stream:
        for sample_id in sorted(samples):
            sample = samples[sample_id]
            for arm in ARMS:
                card = cards.get(arm, {}).get(sample_id)
                for question in QUESTIONS:
                    key = (sample_id, arm, question)
                    if key in done:
                        continue
                    if arm == "template":
                        raw = json.dumps(template_answer(question, card), sort_keys=True)
                        prompt_sha = None
                    else:
                        messages = make_messages(question, catalog, sequence=sample["short_sequence"] if arm == "sequence" else None,
                                                 card=card)
                        prompt_sha = hashlib.sha256(dump(messages).encode()).hexdigest()
                        raw = generator.generate(messages)
                    parsed, error = parse_answer(raw, question, len(sample["short_sequence"]), class_ids, card)
                    record = {"sample_id": sample_id, "arm": arm, "question": question, "raw": raw,
                              "parsed": parsed, "parse_error": error, "prompt_sha256": prompt_sha,
                              "donor_sample_id": donors[sample_id] if arm == "motifae_shuffled" else None}
                    stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush()
                    records.append(record)
    metrics, details = evaluate(records, samples, cards)
    metrics.update(status="complete", expected_records=67 * len(ARMS) * len(QUESTIONS),
                   observed_records=len(records), qwen_generations=67 * 5 * 4,
                   template_answers=67 * 4, run_contract_sha256=sha(contract_path),
                   raw_answers_sha256=sha(raw_path))
    if len(records) != metrics["expected_records"]:
        raise RuntimeError("Incomplete QA matrix")
    # Recheck byte pins after inference; never publish a mixed-version evaluation.
    if verify_freeze(scope) != contract:
        raise RuntimeError("Frozen inputs changed during QA")
    details_path = run_dir / "evaluation_details.jsonl"
    details_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in details), encoding="utf-8")
    output = scope / "analysis/qa_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dump(metrics), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--resume", action="store_true", help="Explicit identical-input technical recovery only")
    args = parser.parse_args()
    scope = args.scope.resolve()
    result = run_smoke(scope) if args.phase == "smoke" else run_formal(scope, resume=args.resume)
    print(json.dumps({k: v for k, v in result.items() if k in {"status", "observed_records", "phase", "model_completed_four_generations"}}, sort_keys=True))


if __name__ == "__main__":
    main()

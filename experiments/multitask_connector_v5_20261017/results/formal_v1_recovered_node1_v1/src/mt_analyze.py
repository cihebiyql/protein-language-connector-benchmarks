"""Independently rescore the frozen 45-case matrix, including unfinished cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

from .mt_metrics import build_metrics, score_intervals


def read_json(path):
    def unique(pairs):
        if len(dict(pairs)) != len(pairs):
            raise ValueError("duplicate JSON key")
        return dict(pairs)
    result = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    def check(value):
        if isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
    check(result)
    return result


def read_jsonl(path):
    # Reuse strict JSON parsing without silently ignoring malformed lines.
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    result = []
    for line in lines:
        if not line.strip():
            continue
        def unique(pairs):
            if len(dict(pairs)) != len(pairs):
                raise ValueError("duplicate JSONL key")
            return dict(pairs)
        value = json.loads(line, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        if not isinstance(value, dict):
            raise ValueError("prediction/event JSONL rows must be objects")
        result.append(value)
    return result


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fixed_cases():
    return [(task, method, backbone, seed)
            for task in ("meltome", "elm", "rt")
            for method, backbone in (("ESM", None), ("M0", "qwen25_1p5b"), ("M3", "qwen25_1p5b"),
                                     ("M0", "qwen3_4b_2507"), ("M3", "qwen3_4b_2507"))
            for seed in (17, 42, 2026)]


def _identity(case):
    return case["task"], case["method"], None if case["method"] == "ESM" else case["backbone"], case["seed"]


def primary_metric(task):
    return {"meltome": "spearman", "elm": "class_macro_mrr", "rt": "mae"}[task]


def paired_comparisons(registry):
    indexed = {_identity(row): row for row in registry}
    comparisons = []
    for task in ("meltome", "elm", "rt"):
        metric = primary_metric(task)
        for backbone in ("qwen25_1p5b", "qwen3_4b_2507"):
            for left, right in (("M0", "M3"), ("ESM", "M0"), ("ESM", "M3")):
                pairs = []
                for seed in (17, 42, 2026):
                    a = indexed.get((task, left, None if left == "ESM" else backbone, seed))
                    b = indexed.get((task, right, backbone, seed))
                    av = (a or {}).get("primary_value") if (a or {}).get("analysis_status") == "accepted" else None
                    bv = (b or {}).get("primary_value") if (b or {}).get("analysis_status") == "accepted" else None
                    pairs.append({"seed": seed, "left": av, "right": bv,
                                  "left_minus_right": av - bv if av is not None and bv is not None else None})
                differences = [p["left_minus_right"] for p in pairs if p["left_minus_right"] is not None]
                complete = len(differences) == 3
                comparisons.append({"task": task, "backbone": backbone, "left": left, "right": right,
                                    "metric": metric, "difference": "left-minus-right",
                                    "positive_difference_favors": left if task != "rt" else right,
                                    "pairs": pairs, "n_complete_pairs": len(differences), "required_seeds": 3,
                                    "mean": mean(differences) if complete else None,
                                    "sample_sd": stdev(differences) if complete else None,
                                    "right_better_all_three": (all(d < 0 for d in differences) if task != "rt"
                                                               else all(d > 0 for d in differences)) if complete else None})
    return comparisons


def arm_summaries(registry):
    summaries = []
    for task, method, backbone, _ in fixed_cases()[::3]:
        rows = [r for r in registry if _identity(r)[:3] == (task, method, backbone)]
        metrics = {}
        keys = ("class_macro_mrr", "mrr", "recall_at_1", "recall_at_3") if task == "elm" else (
            "mae", "rmse", "spearman", "group_macro_mae", "group_macro_rmse", "group_macro_spearman")
        for key in keys:
            values = [(r.get("recomputed_metrics") or {}).get("readout", {}).get(key)
                      for r in rows if r["analysis_status"] == "accepted"]
            valid = [v for v in values if v is not None]
            complete = len(valid) == 3 and len({r["seed"] for r in rows if r["analysis_status"] == "accepted"}) == 3
            metrics[key] = {"n_valid_seeds": len(valid), "mean": mean(valid) if complete else None,
                            "sample_sd": stdev(valid) if complete else None}
        summaries.append({"task": task, "method": method, "backbone": backbone, "metrics": metrics})
    return summaries


def _verify_plan_pins(plan):
    for name in ("source_pins", "asset_pins"):
        for path, expected in plan.get(name, {}).items():
            expected = expected.get("sha256") if isinstance(expected, dict) else expected
            if file_sha(path) != expected:
                raise ValueError("frozen plan pin changed: " + str(path))


def recompute(task, method, artifacts, rows, catalog, *, saved=None):
    """Pure CPU recomputation; the runtime validator authenticates artifacts first."""
    artifacts = Path(artifacts)
    readout = read_jsonl(artifacts / "readout_predictions.jsonl")
    language = read_jsonl(artifacts / "LM_predictions.jsonl") if method != "ESM" else None
    c1 = read_jsonl(artifacts / "C1_predictions.jsonl") if method != "ESM" else None
    return build_metrics(task, rows, readout, language, c1, catalog)


def collect(plan_path, queue_root, data_root, *, validator=None, data_module=None):
    from . import mt_data
    data_module = data_module or mt_data
    plan_path, queue_root, data_root = Path(plan_path), Path(queue_root).resolve(), Path(data_root).resolve()
    plan = read_json(plan_path)
    cases = plan["cases"]
    identities = [_identity(case) for case in cases]
    if len(cases) != 45 or len(set(identities)) != 45 or set(identities) != set(fixed_cases()):
        raise ValueError("plan must register the exact fresh 45-case matrix")
    if len({c["case_id"] for c in cases}) != 45:
        raise ValueError("duplicate plan case_id")
    _verify_plan_pins(plan)
    if Path(plan["paths"]["data_root"]).resolve() != data_root:
        raise ValueError("analysis data root differs from frozen plan")
    manifest = data_module.load_manifest(data_root)
    rows = {task: data_module.load_task(data_root, task, "test") for task in ("meltome", "elm", "rt")}
    catalog = data_module.load_catalog(data_root)
    ledger_path = queue_root / "queue_events.jsonl"
    if ledger_path.exists():
        queue_binding = read_json(queue_root / "queue_binding.json")
        if queue_binding.get("plan_sha256") != file_sha(plan_path):
            raise ValueError("queue is bound to a different frozen plan")
    events = read_jsonl(ledger_path) if ledger_path.exists() else []
    states, known = {}, {case["case_id"] for case in cases}
    for event in events:
        if "case_id" not in event:
            continue
        if event["case_id"] not in known:
            raise ValueError("ledger references a case outside frozen plan")
        states[event["case_id"]] = event
    if validator is None:
        from .mt_queue import accepted_attempt
        validator = accepted_attempt
    registry = []
    for case in cases:
        event = states.get(case["case_id"], {"status": "pending"})
        entry = {**case, "queue_status": event["status"], "analysis_status": "not_accepted",
                 "reason": event.get("reason"), "primary_metric": primary_metric(case["task"]),
                 "primary_value": None, "recomputed_metrics": None, "capacity": None}
        if event["status"] == "accepted":
            try:
                attempt = Path(event["attempt"])
                attempt = (queue_root / attempt).resolve() if not attempt.is_absolute() else attempt.resolve()
                if not attempt.is_relative_to(queue_root):
                    raise ValueError("accepted attempt escapes queue root")
                binding = validator(plan, attempt, case)
                if event.get("binding") != binding or binding.get("plan_sha256") != file_sha(plan_path):
                    raise ValueError("accepted event binding differs from current artifacts")
                artifacts = attempt / "artifacts"
                run = read_json(artifacts / "run_manifest.json")
                if run.get("phase") != "formal" or run.get("status") not in ("complete", "succeeded"):
                    raise ValueError("accepted artifact is not a completed fresh formal run")
                if run.get("scope_id") != plan["scope_id"]:
                    raise ValueError("accepted artifact belongs to a different scope")
                saved = read_json(artifacts / "metrics.json")
                actual = recompute(case["task"], case["method"], artifacts, rows[case["task"]], catalog, saved=saved)
                if case["task"] == "rt":
                    calibration_rows = data_module.load_task(data_root, "rt", "calibration")
                    calibration_predictions = read_jsonl(artifacts / "calibration_predictions.jsonl")
                    actual["calibration"] = score_intervals(calibration_rows, calibration_predictions, rows["rt"],
                        read_jsonl(artifacts / "readout_predictions.jsonl"), alpha=0.1)
                for channel, fresh in actual.items():
                    if canonical(saved.get(channel)) != canonical(fresh):
                        raise ValueError("saved " + channel + " metrics differ from independent rescoring")
                entry.update(analysis_status="accepted", reason=None, binding=binding,
                             primary_value=actual["readout"][entry["primary_metric"]],
                             recomputed_metrics=actual, capacity=run.get("model_policy", run.get("capacity", run.get("capacity_report"))),
                             runtime=run.get("runtime"), resources=run.get("resources"),
                             run_manifest_sha256=file_sha(artifacts / "run_manifest.json"))
                entry["device_budget"] = read_json(attempt / "device_budget.json") if (attempt / "device_budget.json").exists() else None
                entry["supervision"] = read_json(attempt / "supervision/finished.json") if (attempt / "supervision/finished.json").exists() else None
            except (OSError, ValueError, KeyError, TypeError) as exc:
                entry.update(analysis_status="revalidation_rejected", reason=str(exc))
        registry.append(entry)
    return {"schema": "multitask-independent-analysis.v1", "scope_id": plan["scope_id"],
            "created_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": file_sha(plan_path),
            "queue_ledger_sha256": file_sha(ledger_path) if ledger_path.exists() else None,
            "data_manifest_sha256": file_sha(data_root / "manifest.json"),
            "data_source_pins": manifest.get("source_pins"),
            "analysis_source_sha256": file_sha(__file__), "expected_cases": 45,
            "queue_status_counts": dict(Counter(row["queue_status"] for row in registry)),
            "analysis_status_counts": dict(Counter(row["analysis_status"] for row in registry)),
            "cases": registry, "arm_summaries": arm_summaries(registry), "paired_comparisons": paired_comparisons(registry),
            "interpretation": "Fresh fits only; primary head scores, language and C1 are separate. Mean and sample SD require all three paired seeds. Seed repeats are not biological replicates."}


def write_report(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# 三任务连接器独立复核", "", f"计划45组；状态：{result['queue_status_counts']}。", "",
             "已接受且独立复算：" + str(result["analysis_status_counts"].get("accepted", 0)) + "组。",
             "", "## 固定主指标", "", "热稳定性：Spearman；功能描述：已知正例的类别宏平均MRR；保留时间：MAE。",
             "语言生成和C1单独报告，缺失或非有限输出不会被删去后重新计算总体分数。", "",
             "## 同随机种子的对照", "", "|任务|底座|比较|完成对数|均值（左减右）|样本标准差|", "|---|---|---|---:|---:|---:|"]
    for row in result["paired_comparisons"]:
        lines.append(f"|{row['task']}|{row['backbone']}|{row['left']}−{row['right']}|{row['n_complete_pairs']}/3|{row['mean']}|{row['sample_sd']}|")
    lines.extend(["", "只有三个相同种子全部完成才汇总均值与标准差。当前未完成位置保持原状态，不以旧实验补齐。",
                  "类别未标注候选仅用于相对排序，不能解释为实验阴性；跨底座不宣称参数量或计算量相等。"])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "queue-root", "data-root", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("analysis output must be fresh")
    result = collect(args.plan, args.queue_root, args.data_root)
    write_report(result, args.output)
    print(json.dumps(result["analysis_status_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()

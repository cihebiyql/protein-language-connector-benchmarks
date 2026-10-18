"""Aggregate-only, predeclared exploratory report; never runs model evaluation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


ARMS = ("sequence", "regex", "raw_probe", "motifae", "motifae_shuffled", "template")
COMPETITORS = ("sequence", "regex", "raw_probe")
QUESTIONS = ("class", "position", "activity", "evidence")
DEFAULT_SIGNALS = {"min_auc_gain": 0.02, "min_class_gain": 0.02,
                   "min_wrong_card_drop": 0.02, "max_overclaim_increase": 0.0}
EXPECTED_TEST_ROWS = 67
EXPECTED_ANSWERS = EXPECTED_TEST_ROWS * len(ARMS) * len(QUESTIONS)
AGGREGATE_FIELDS = {"macro_parent", "defined_parents", "defined_rows", "total_rows",
                    "all_row_mean_including_unsupported", "covered_row_mean", "covered_rows"}
QA_FIELDS = {"expected", "observed", "valid", "invalid", "valid_rate", "card_faithfulness_applicable",
             "known_positive_recovered_count", "known_positive_recovered_rate",
             "card_supported_nonunknown_count", "card_supported_nonunknown_rate",
             "unsupported_nonunknown_count", "unsupported_nonunknown_rate",
             "known_positive_overlap_count", "known_positive_overlap_rate",
             "anchor_overlap_count", "anchor_overlap_rate", "card_contained_span_count",
             "card_contained_span_rate", "activity_overclaim_count", "activity_overclaim_rate",
             "correct_not_established_count", "correct_not_established_rate",
             "strongest_record_or_correct_abstention_count", "strongest_record_or_correct_abstention_rate",
             "valid_nonnull_citation_count", "valid_nonnull_citation_rate",
             "mean_best_known_positive_iou_full_denominator"}
FEATURE_NAMES = {
    "top32_full_short_jaccard", "top32_context_window_short_jaccard",
    "anchor_feature_context_window_peak", "anchor_feature_short_peak", "anchor_feature_short_context_ratio",
}
for _arm in ("motifae", "raw_probe"):
    FEATURE_NAMES.update({_arm + "_covered_anchor", _arm + "_covered_any_known_positive"})
    for _k in (1, 3):
        FEATURE_NAMES.update({f"{_arm}_ranking_top{_k}_positive_hit",
                              f"{_arm}_ranking_top{_k}_known_positive_recall",
                              f"{_arm}_card_top{_k}_positive_hit"})
    for _kind in ("full", "short"):
        FEATURE_NAMES.update({f"{_arm}_{_kind}_anchor_auc", f"{_arm}_{_kind}_anchor_argmax_hit",
                              f"{_arm}_{_kind}_qualified_anchor_auc"})
for _kind in ("full", "short"):
    for _suffix in ("match_count", "known_positive_residue_recall", "known_positive_span_recovered",
                    "known_positive_span_count", "known_positive_overlap"):
        FEATURE_NAMES.add(f"regex_{_kind}_{_suffix}")

LIMITATIONS = [
    "这是单一固定划分的探索性试验，不是多随机种子稳定提升证据。",
    "新面板只来自原 B1 TRAIN；原验证集、测试集和 v6 实验不受改变。",
    "未标注残基只是定位参考，不是实验确认的生物学阴性。",
    "短肽由已标注母蛋白区域截取，未提供孤立短肽活性实验。",
    "特征抽取成功不等于完成生物学解释或实现语言模型翻译。",
    "命名依赖 TRAIN 选择及资格集固定门槛；低覆盖类别保留，不重新划分或降低门槛。",
    "两个特征方案的资格类别可能不同；给定锚点类别的配对 AUROC 只比较两者均定义的同一条样本，并单独报告覆盖。",
    "功能类别指标衡量已知正例恢复，不能解释为特异性或对未标注类别的真阴性判断。",
    "活性问题的正确边界答案始终是未建立证据，不能作为独立的分子信息理解能力。",
    "模板是机械读取证据卡的对照，不是 Qwen 推理能力或新的训练模型。",
    "本试验信号不替代、改变或推翻 v6 原有方向暂停规则。",
]


def dumps(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def safe_scalar(value) -> bool:
    return value is None or isinstance(value, bool) or numeric(value)


def read_json(path: Path) -> tuple[dict | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return None, "invalid_object"
        return value, None
    except (ValueError, OSError, UnicodeError):
        # Never expose exception text or paths into public summaries.
        return None, "unreadable_or_invalid_json"


def paired_short_auc(rows: list[dict]) -> dict:
    """Pair same anchor row first, then equal-weight its parent mean gain."""
    parents = defaultdict(list)
    paired_rows = 0
    for row in rows:
        a, b = row.get("motifae_short_anchor_auc"), row.get("raw_probe_short_anchor_auc")
        if numeric(a) and numeric(b) and 0 <= a <= 1 and 0 <= b <= 1:
            parent = row.get("parent_accession")
            if not isinstance(parent, str) or not parent:
                continue
            parents[parent].append(float(a) - float(b))
            paired_rows += 1
    gains = [statistics.mean(values) for values in parents.values()]
    return {
        "mean_gain": statistics.mean(gains) if gains else None,
        "sample_sd": statistics.stdev(gains) if len(gains) >= 2 else None,
        "paired_parents": len(gains), "paired_rows": paired_rows,
        "excluded_unpaired_rows": len(rows) - paired_rows,
        "calculation": "same-anchor row MotifAE AUROC minus raw-H33 AUROC; average within parent; equal-weight mean of parent gains",
        "inference": "descriptive sample SD; no bootstrap, p-value or multi-seed stability claim",
    }


def feature_complete(feature: dict | None) -> bool:
    if not feature or feature.get("test_rows") != EXPECTED_TEST_ROWS:
        return False
    rows = feature.get("private_per_row")
    if not isinstance(rows, list) or len(rows) != EXPECTED_TEST_ROWS:
        return False
    identifiers = [row.get("sample_id") for row in rows if isinstance(row, dict)]
    if (len(identifiers) != EXPECTED_TEST_ROWS or not all(isinstance(x, str) and x for x in identifiers)
            or len(set(identifiers)) != EXPECTED_TEST_ROWS):
        return False
    required = {"parent_accession", "anchor_class_id", "motifae_short_anchor_auc", "raw_probe_short_anchor_auc"}
    return all(required <= set(row) for row in rows)


def qa_complete(qa: dict | None) -> bool:
    if not qa or qa.get("status") != "complete":
        return False
    if (qa.get("test_peptides") != EXPECTED_TEST_ROWS or qa.get("expected_records") != EXPECTED_ANSWERS
            or qa.get("observed_records") != EXPECTED_ANSWERS):
        return False
    for arm in ARMS:
        questions = qa.get("arms", {}).get(arm, {})
        for question in QUESTIONS:
            cell = questions.get(question, {})
            if cell.get("expected") != EXPECTED_TEST_ROWS or cell.get("observed") != EXPECTED_TEST_ROWS:
                return False
            valid, invalid = cell.get("valid"), cell.get("invalid")
            if not numeric(valid) or not numeric(invalid) or valid < 0 or invalid < 0:
                return False
            if valid + invalid != EXPECTED_TEST_ROWS:
                return False
    return True


def sanitize_feature(feature: dict | None) -> dict:
    feature = feature or {}
    classes = {}
    for arm in ("motifae", "raw_probe"):
        values = feature.get("qualified_classes", {}).get(arm, [])
        # Only controlled catalog identifiers; neither descriptions nor arbitrary text.
        classes[arm] = [value for value in values if isinstance(value, str)
                        and re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", value)]
    metrics = {}
    for name, summary in feature.get("metrics", {}).items():
        if name not in FEATURE_NAMES or not isinstance(summary, dict):
            continue
        metrics[name] = {key: value for key, value in summary.items()
                         if key in AGGREGATE_FIELDS and safe_scalar(value)}
    return {"test_rows": feature.get("test_rows") if numeric(feature.get("test_rows")) else None,
            "test_parents": feature.get("test_parents") if numeric(feature.get("test_parents")) else None,
            "qualified_classes": classes, "qualified_class_counts": {arm: len(v) for arm, v in classes.items()},
            "metrics": metrics}


def sanitize_qa(qa: dict | None) -> dict:
    qa = qa or {}
    arms = {}
    for arm in ARMS:
        arms[arm] = {}
        for question in QUESTIONS:
            cell = qa.get("arms", {}).get(arm, {}).get(question, {})
            arms[arm][question] = {key: value for key, value in cell.items()
                                   if key in QA_FIELDS and safe_scalar(value)}
    return {"test_peptides": qa.get("test_peptides") if numeric(qa.get("test_peptides")) else None,
            "expected_records": qa.get("expected_records") if numeric(qa.get("expected_records")) else None,
            "observed_records": qa.get("observed_records") if numeric(qa.get("observed_records")) else None,
            "qwen_generations": qa.get("qwen_generations") if numeric(qa.get("qwen_generations")) else None,
            "template_answers": qa.get("template_answers") if numeric(qa.get("template_answers")) else None,
            "arms": arms, "full_denominator": EXPECTED_TEST_ROWS,
            "template_is_not_qwen": True, "class_recovery_is_not_specificity": True}


def data_counts(manifest: dict | None) -> dict:
    output = {}
    for split in ("train", "qualification", "test"):
        source = (manifest or {}).get("summary", {}).get("splits", {}).get(split, {})
        output[split] = {new: source.get(old) if numeric(source.get(old)) else None
                         for old, new in (("rows", "rows"), ("parents", "parent_count"),
                                          ("components", "group_count"), ("classes", "class_count"))}
    return output


def question_rate(qa: dict, arm: str, question: str, field: str) -> float | None:
    cell = qa.get("arms", {}).get(arm, {}).get(question, {})
    count = cell.get(field + "_count")
    rate = cell.get(field + "_rate")
    if numeric(count) and 0 <= count <= EXPECTED_TEST_ROWS:
        observed = count / EXPECTED_TEST_ROWS
        if numeric(rate) and not math.isclose(rate, observed, abs_tol=1e-12):
            return None
        return observed
    if numeric(rate) and 0 <= rate <= 1:
        return float(rate)
    # QA's Counter omits zero-count activity keys when every answer is invalid.
    # Do not manufacture a safe-overclaim rate for a missing aggregate field.
    return None


def determine_gates(feature: dict, qa: dict, signals: dict) -> tuple[dict, dict]:
    paired = paired_short_auc(feature["private_per_row"])
    class_rates = {arm: question_rate(qa, arm, "class", "known_positive_recovered")
                   for arm in (*COMPETITORS, "motifae", "motifae_shuffled")}
    overclaim = {arm: question_rate(qa, arm, "activity", "activity_overclaim")
                 for arm in (*COMPETITORS, "motifae")}
    competitor_values = [class_rates[arm] for arm in COMPETITORS]
    best_class = max(competitor_values) if all(x is not None for x in competitor_values) else None
    class_gain = class_rates["motifae"] - best_class if class_rates["motifae"] is not None and best_class is not None else None
    wrong_drop = (class_rates["motifae"] - class_rates["motifae_shuffled"]
                  if class_rates["motifae"] is not None and class_rates["motifae_shuffled"] is not None else None)
    others = [overclaim[arm] for arm in COMPETITORS]
    best_overclaim = min(others) if all(x is not None for x in others) else None
    overclaim_increase = (overclaim["motifae"] - best_overclaim
                         if overclaim["motifae"] is not None and best_overclaim is not None else None)

    def at_least(value, limit):
        return value is not None and value >= limit

    gates = {
        "paired_short_auc_gain": {"observed": paired["mean_gain"], "minimum": signals["min_auc_gain"],
                                  "passed": at_least(paired["mean_gain"], signals["min_auc_gain"])},
        "qwen_class_gain_over_best_control": {"observed": class_gain, "minimum": signals["min_class_gain"],
                                              "passed": at_least(class_gain, signals["min_class_gain"])},
        "wrong_card_class_recovery_drop": {"observed": wrong_drop, "minimum": signals["min_wrong_card_drop"],
                                            "passed": at_least(wrong_drop, signals["min_wrong_card_drop"])},
        "activity_overclaim_increase": {"observed": overclaim_increase, "maximum": signals["max_overclaim_increase"],
                                         "passed": overclaim_increase is not None and overclaim_increase <= signals["max_overclaim_increase"]},
    }
    comparison = {"paired_short_auc": paired, "qwen_class_recovery_rates": class_rates,
                  "best_qwen_control_class_recovery_rate": best_class,
                  "qwen_activity_overclaim_rates": overclaim,
                  "best_qwen_control_activity_overclaim_rate": best_overclaim,
                  "qwen_control_arms": list(COMPETITORS), "all_gates_required": True}
    return gates, comparison


def build_summary(feature: dict | None, qa: dict | None, manifest: dict | None,
                  config: dict | None, *, failure_stage: str | None = None,
                  source_digests: dict | None = None) -> dict:
    signals = {**DEFAULT_SIGNALS, **(config or {}).get("pilot_signals", {})}
    valid_signals = (set(signals) == set(DEFAULT_SIGNALS)
                     and all(numeric(value) and value >= 0 for value in signals.values()))
    feature_ok, qa_ok = feature_complete(feature), qa_complete(qa)
    safe_stage = failure_stage if failure_stage in {
        "assets", "features", "extraction", "dictionary", "qa", "report", "publication", "publish", "pipeline"
    } else "unspecified" if failure_stage else None
    technical = bool(failure_stage) or not feature_ok or not qa_ok or not valid_signals
    gates, comparison = {}, {"paired_short_auc": paired_short_auc((feature or {}).get("private_per_row", []))}
    if not technical:
        gates, comparison = determine_gates(feature, qa, signals)
        if any(gate["observed"] is None for gate in gates.values()):
            technical = True
    status = ("technical_inconclusive" if technical else "complete_pilot_promising"
              if all(gate["passed"] for gate in gates.values()) else "complete_pilot_no_incremental_gain")
    qualified_dict = (config or {}).get("dictionary", {})
    safe_signals = {key: value for key, value in signals.items() if key in DEFAULT_SIGNALS and numeric(value)}
    safe_digests = {key: value for key, value in (source_digests or {}).items()
                    if re.fullmatch(r"[a-z0-9_]+_sha256", key) and
                    (value is None or isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value))}
    return {
        "schema_version": 1, "status": status, "scope": "exploratory_motifae_evidence_pilot",
        "completion": {"feature_pipeline_complete": feature_ok, "qa_pipeline_complete": qa_ok,
                       "expected_test_peptides": EXPECTED_TEST_ROWS, "expected_total_answers": EXPECTED_ANSWERS,
                       "failure_stage": safe_stage, "signal_configuration_valid": valid_signals},
        "decision_scope": "exploratory signal only; v6 direction-pause rule unchanged",
        "stable_multi_seed_gain_established": False,
        "pilot_signals": safe_signals, "gates": gates, "comparison": comparison,
        "dictionary_qualification": {key: qualified_dict.get(key, default)
                                       if numeric(qualified_dict.get(key, default)) else None
                                       for key, default in (("qualification_auc", 0.70),
                                                            ("min_train_parents", 2),
                                                            ("min_qualification_parents", 2))},
        "data_counts": data_counts(manifest), "feature": sanitize_feature(feature), "qa": sanitize_qa(qa),
        "limitations": LIMITATIONS, "source_digests": safe_digests,
        "publication_scope": "aggregates and digests only; no per-query biological data, cards, answers or dictionary",
    }


def fmt(value, percent=False) -> str:
    if not numeric(value):
        return "未定义"
    return f"{value * 100:.2f}%" if percent else f"{value:.4f}"


def render_report(summary: dict) -> str:
    status = summary["status"]
    lead = {
        "complete_pilot_promising": "本轮探索性试验达到全部预先固定的信号门槛，值得进一步验证；尚未证明多随机种子的稳定提升。",
        "complete_pilot_no_incremental_gain": "本轮完整探索性试验未同时达到预先固定的增益门槛，当前没有足够的 MotifAE 额外收益信号。",
        "technical_inconclusive": "本轮存在未完成或未通过完整性检查的环节，技术结果不足以判断 MotifAE 是否有效。",
    }[status]
    completion = summary["completion"]
    lines = [lead, "", f"状态：`{status}`。特征评价完整：{completion['feature_pipeline_complete']}；问答矩阵完整：{completion['qa_pipeline_complete']}。",
             "", "问答完整矩阵固定为 67 条 TEST 短肽 × 6 个方案 × 4 个问题，共 1,608 条回答，其中 Qwen 生成 1,340 条、机械模板 268 条。所有问答比率的分母均保留 67 条；无效答案不删除、不修补。",
             "", "这套试验检查关联特征能否形成有用证据卡。它不更改 v6 的训练、评测、停止规则，也不把单次试验当作该方向已经获救或已应暂停的结论。", ""]
    lines += ["| 数据分区 | 样本数 | 母蛋白数 | 分组数 | 类别数 |", "|---|---:|---:|---:|---:|"]
    for split, label in (("train", "TRAIN"), ("qualification", "资格集"), ("test", "TEST")):
        counts = summary["data_counts"][split]
        lines.append(f"| {label} | {counts['rows']} | {counts['parent_count']} | {counts['group_count']} | {counts['class_count']} |")
    lines += ["", "特征定位的 AUROC 使用标注残基与未标注参考位置作比较，后者不代表生物学阴性。", "",
              "| 特征方案 | 全长给定类别 AUROC | 短肽给定类别 AUROC | 已命名类别短肽 AUROC | 通过资格类别数 |",
              "|---|---:|---:|---:|---:|"]
    feature = summary["feature"]
    for arm, label in (("motifae", "MotifAE"), ("raw_probe", "同骨干原始 H33 线性探针")):
        values = [feature["metrics"].get(f"{arm}_{kind}", {}).get("macro_parent")
                  for kind in ("full_anchor_auc", "short_anchor_auc", "short_qualified_anchor_auc")]
        lines.append(f"| {label} | {fmt(values[0])} | {fmt(values[1])} | {fmt(values[2])} | {feature['qualified_class_counts'][arm]} |")
    paired = summary["comparison"]["paired_short_auc"]
    lines += ["", f"控制同一锚点样本后，先在母蛋白内平均差值，再对母蛋白等权平均：MotifAE − 原始 H33 的短肽 AUROC 差值为 **{fmt(paired['mean_gain'])}**，母蛋白差值的样本标准差为 {fmt(paired['sample_sd'])}；共 {paired['paired_parents']} 个母蛋白、{paired['paired_rows']} 条配对样本，另有 {paired['excluded_unpaired_rows']} 条无法配对。此处标准差仅作描述。",
              "", f"全长与短肽前 32 特征 Jaccard：{fmt(feature['metrics'].get('top32_full_short_jaccard', {}).get('macro_parent'))}；只取母蛋白中对应短肽窗口再比较：{fmt(feature['metrics'].get('top32_context_window_short_jaccard', {}).get('macro_parent'))}。", "",
              "| 问答方案 | 已知正例类别恢复率 | 定位与已知正例重叠率 | 全部问题 JSON 有效率 | 活性过度声称率 | 最强证据引用/正确弃答率 |",
              "|---|---:|---:|---:|---:|---:|"]
    labels = {"sequence": "Qwen＋序列", "regex": "Qwen＋正则证据", "raw_probe": "Qwen＋原始 H33 探针证据",
              "motifae": "Qwen＋MotifAE 证据", "motifae_shuffled": "Qwen＋换错短肽证据", "template": "机械模板（不是 Qwen）"}
    for arm in ARMS:
        cells = summary["qa"]["arms"][arm]
        class_rate = cells["class"].get("known_positive_recovered_rate")
        overlap = cells["position"].get("known_positive_overlap_rate")
        activity = cells["activity"].get("activity_overclaim_rate")
        citation = cells["evidence"].get("strongest_record_or_correct_abstention_rate")
        valid = ([cell.get("valid") for cell in cells.values()])
        valid_rate = sum(valid) / (EXPECTED_TEST_ROWS * len(QUESTIONS)) if all(numeric(x) for x in valid) else None
        lines.append(f"| {labels[arm]} | {fmt(class_rate, True)} | {fmt(overlap, True)} | {fmt(valid_rate, True)} | {fmt(activity, True)} | {fmt(citation, True)} |")
    if summary["gates"]:
        lines += ["", "| 预先固定的探索性信号 | 实际值 | 门槛 | 是否达到 |", "|---|---:|---:|---|"]
        names = {"paired_short_auc_gain": "配对母蛋白平均短肽 AUROC 增益", "qwen_class_gain_over_best_control": "类别恢复率超过最佳 Qwen 对照",
                 "wrong_card_class_recovery_drop": "换错证据后类别恢复率下降", "activity_overclaim_increase": "相对最低 Qwen 对照过度声称率的增加"}
        for key, gate in summary["gates"].items():
            limit = ("≥ " + fmt(gate["minimum"])) if "minimum" in gate else "≤ " + fmt(gate["maximum"])
            lines.append(f"| {names[key]} | {fmt(gate['observed'])} | {limit} | {'是' if gate['passed'] else '否'} |")
    lines += ["", "上述门槛必须同时满足；Qwen 对照限定为序列、正则和原始 H33 探针三个方案。机械模板单独报告，不参与 Qwen 增益门槛。", ""]
    lines += [f"- {text}" for text in summary["limitations"]]
    lines += ["", "公开内容仅含汇总数字、方法边界和文件指纹；逐条序列、坐标、证据卡、模型回答及完整字典保留在私有实验目录。", ""]
    return "\n".join(lines)


def preserve_and_write(path: Path, value: str) -> None:
    """Derived report may progress after technical recovery; preserve prior bytes."""
    encoded = value.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = path.read_bytes()
        if old == encoded:
            return
        archive = path.parent / "history" / (path.stem + "_" + digest_bytes(old) + path.suffix)
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            archive.write_bytes(old)
        elif archive.read_bytes() != old:
            raise RuntimeError("Report history fingerprint collision")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def run(scope: Path, failure_stage: str | None = None) -> dict:
    scope = scope.resolve()
    feature, feature_error = read_json(scope / "analysis/feature_metrics.json")
    qa, qa_error = read_json(scope / "analysis/qa_metrics.json")
    manifest, manifest_error = read_json(scope / "data/manifest.json")
    config, config_error = read_json(scope / "config.json")
    sources = {
        "config_sha256": digest_file(scope / "config.json"),
        "frozen_manifest_sha256": digest_file(scope / "frozen_manifest.json"),
        "dictionary_sha256": digest_file(scope / "dictionary.json"),
        "feature_metrics_sha256": digest_file(scope / "analysis/feature_metrics.json"),
        "qa_metrics_sha256": digest_file(scope / "analysis/qa_metrics.json"),
        "qa_raw_answers_sha256": digest_file(scope / "runs/qa/raw_answers.jsonl"),
        "qa_run_contract_sha256": digest_file(scope / "runs/qa/run_contract.json"),
        "cache_receipts_sha256": digest_bytes(dumps((feature or {}).get("cache_file_receipts", {})).encode()),
    }
    if config_error or manifest_error:
        failure_stage = failure_stage or "report"
    summary = build_summary(feature, qa, manifest, config, failure_stage=failure_stage, source_digests=sources)
    summary["input_health"] = {"feature_metrics": feature_error or "readable",
                               "qa_metrics": qa_error or "readable", "data_manifest": manifest_error or "readable",
                               "configuration": config_error or "readable"}
    preserve_and_write(scope / "reports/SUMMARY.json", dumps(summary))
    preserve_and_write(scope / "reports/REPORT.md", render_report(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--failure-stage")
    args = parser.parse_args()
    summary = run(args.scope, args.failure_stage)
    print(json.dumps({"status": summary["status"], "completion": summary["completion"]}, sort_keys=True))


if __name__ == "__main__":
    main()

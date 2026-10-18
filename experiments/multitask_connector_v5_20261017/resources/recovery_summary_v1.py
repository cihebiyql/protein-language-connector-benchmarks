"""Describe frozen V5 paired results; never select runs or change scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

SEEDS = (17, 42, 2026)
TASKS = ("meltome", "elm", "rt")
BACKBONES = ("qwen25_1p5b", "qwen3_4b_2507")
METRICS = {"meltome": "spearman", "elm": "class_macro_mrr", "rt": "mae"}
TASK_NAMES = {"meltome": "热稳定性", "elm": "功能描述", "rt": "保留时间"}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _read(path):
    def unique(pairs):
        if len(dict(pairs)) != len(pairs):
            raise ValueError("Duplicate JSON key")
        return dict(pairs)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _comparison(source, task, backbone, left):
    """Positive gain always favors M3; missing seeds retain null aggregates."""
    result = {"task": task, "backbone": backbone, "left": left, "right": "M3",
              "metric": METRICS[task], "required_seeds": list(SEEDS),
              "gain_definition": "M0-M3" if task == "rt" and left == "M0" else
              "ESM-M3" if task == "rt" else "M3-" + left,
              "positive_gain_favors": "M3", "source_present": source is not None,
              "pairs": [], "complete_three_seeds": False, "n_complete_pairs": 0,
              "mean_gain": None, "sample_sd_gain": None,
              "wins": 0, "ties": 0, "losses": 0,
              "source_right_better_all_three": None, "three_of_three_favors_M3": None,
              "interpretation": "种子未齐；保留缺失位置，均值和样本标准差为null。"}
    if source is None:
        return result
    if source.get("metric") != METRICS[task]:
        raise ValueError("Comparison metric differs from frozen task metric")
    source_pairs = source.get("pairs", [])
    ids = [row.get("seed") for row in source_pairs]
    if len(ids) != len(set(ids)) or any(seed not in SEEDS for seed in ids):
        raise ValueError("Duplicate or unexpected seed in frozen comparison")
    indexed = {row["seed"]: row for row in source_pairs}
    gains = []
    for seed in SEEDS:
        row = indexed.get(seed, {})
        av, bv = row.get("left"), row.get("right")
        gain = ((av - bv) if task == "rt" else (bv - av)) if _finite(av) and _finite(bv) else None
        result["pairs"].append({"seed": seed, "left": av, "right": bv, "gain": gain})
        if gain is not None:
            gains.append(gain)
    result.update(n_complete_pairs=len(gains), wins=sum(g > 0 for g in gains),
                  ties=sum(g == 0 for g in gains), losses=sum(g < 0 for g in gains),
                  source_right_better_all_three=source.get("right_better_all_three"))
    complete = (len(gains) == 3 and set(ids) == set(SEEDS)
                and source.get("n_complete_pairs") == 3 and source.get("required_seeds") == 3)
    result["complete_three_seeds"] = complete
    if complete:
        all_positive = all(g > 0 for g in gains)
        source_flag = source.get("right_better_all_three")
        result.update(mean_gain=mean(gains), sample_sd_gain=stdev(gains),
                      three_of_three_favors_M3=(source_flag is True and all_positive),
                      source_direction_consistent=(type(source_flag) is bool and source_flag == all_positive))
        if result["three_of_three_favors_M3"]:
            result["interpretation"] = "3/3同方向有利于M3；这是训练重复的描述性一致性。"
        else:
            result["interpretation"] = "三个种子已齐，结果未满足3/3同方向有利于M3。"
        if not result["source_direction_consistent"]:
            result["interpretation"] += "源方向标记与逐种子数值不一致，未据此宣称稳定提升。"
    return result


def _overall(rows):
    completed = sum(row["complete_three_seeds"] for row in rows)
    if completed != 6:
        return {"complete_strata": completed, "expected_strata": 6,
                "all_strata_mean_gain_positive": None, "supports_universal_stable_advantage": None,
                "status": "incomplete",
                "conclusion": f"六个任务×底座层中{completed}/6具有完整三种子；尚不汇总跨任务、跨底座的普遍稳定性结论。"}
    positive_means = all(row["mean_gain"] > 0 for row in rows)
    consistent = all(row["three_of_three_favors_M3"] for row in rows)
    return {"complete_strata": 6, "expected_strata": 6,
            "all_strata_mean_gain_positive": positive_means,
            "supports_universal_stable_advantage": consistent,
            "status": "all_six_directionally_consistent" if consistent else "not_universally_stable",
            "conclusion": ("在本轮固定基准下，六个任务×底座层的三个种子均同方向有利于M3；支持该实验范围内的方向一致性。"
                           if consistent else "六个任务×底座层均完成三种子，但未全部呈现3/3同方向有利；当前结果不支持M3跨任务、跨底座的普遍稳定优势。")}


def _number(value):
    return "null" if value is None else format(value, ".8g")


def _comparison_table(rows):
    lines = ["|任务|底座|指标|完整种子|M3增益均值 ± 样本SD|赢/平/输|3/3同方向有利|",
             "|---|---|---|---:|---:|---:|---|"]
    for row in rows:
        lines.append("|{}|{}|{}|{}/3|{} ± {}|{}/{}/{}|{}|".format(
            TASK_NAMES[row["task"]], row["backbone"], row["metric"], row["n_complete_pairs"],
            _number(row["mean_gain"]), _number(row["sample_sd_gain"]), row["wins"], row["ties"], row["losses"],
            "待齐" if row["three_of_three_favors_M3"] is None else "是" if row["three_of_three_favors_M3"] else "否"))
    return lines


def write_summary(analysis_path, output_dir):
    """Write REPORT.md/summary.json and return their paths plus summary data.

    The output directory must be fresh, preserving every preceding report.
    This reader does not mutate the analysis or decide which attempt is accepted.
    """
    analysis_path = Path(analysis_path).resolve(strict=True)
    raw = analysis_path.read_bytes()
    analysis = _read(analysis_path)
    if analysis.get("schema") != "multitask-independent-analysis.v1" or analysis.get("expected_cases") != 45:
        raise ValueError("Expected the original canonical45 independent analysis")
    index = {}
    for item in analysis.get("paired_comparisons", []):
        key = (item.get("task"), item.get("backbone"), item.get("left"), item.get("right"))
        if key in index:
            raise ValueError("Duplicate paired comparison")
        index[key] = item
    comparisons = {left: [_comparison(index.get((task, backbone, left, "M3")), task, backbone, left)
                          for task in TASKS for backbone in BACKBONES] for left in ("M0", "ESM")}
    secondary = []
    for arm in analysis.get("arm_summaries", []):
        for metric, values in arm.get("metrics", {}).items():
            complete = (values.get("n_valid_seeds") == 3 and _finite(values.get("mean"))
                        and _finite(values.get("sample_sd")))
            secondary.append({"task": arm["task"], "method": arm["method"], "backbone": arm.get("backbone"),
                              "metric": metric, "n_valid_seeds": values.get("n_valid_seeds", 0),
                              "mean": values.get("mean") if complete else None,
                              "sample_sd": values.get("sample_sd") if complete else None})
    summary = {"schema": "v5-recovery-interpretation.v1", "analysis_filename": analysis_path.name,
               "analysis_sha256": hashlib.sha256(raw).hexdigest(),
               "plan_sha256": analysis.get("plan_sha256"),
               "data_manifest_sha256": analysis.get("data_manifest_sha256"),
               "analysis_status_counts": analysis.get("analysis_status_counts"),
               "queue_status_counts": analysis.get("queue_status_counts"),
               "expected_cases": 45, "seeds": list(SEEDS),
               "M3_vs_M0": comparisons["M0"], "M3_vs_ESM": comparisons["ESM"],
               "overall_M3_vs_M0": _overall(comparisons["M0"]),
               "frozen_arm_metrics": secondary,
               "interpretation_policy": {"no_run_or_seed_selection": True,
                   "means_and_sample_sd_require_three_seeds": True,
                   "direction_consistency_is_statistical_significance": False,
                   "seed_repeats_are": "training repeats, not biological replicates",
                   "no_pooling_of_different_task_metric_units": True,
                   "language_and_C1": "Separate channels in the complete source analysis; no values imputed here.",
                   "ELM_scope": "31-candidate function-description retrieval and strict generated-text catalog matching."},
               "recovery_context": "恢复针对ELM、M3、qwen3_4b_2507、seed17加载阶段的OOM；旧失败记录由恢复流程保留，补跑原因是技术性显存失败。数据划分、评价指标、训练预算和种子集合保持冻结。"}
    lines = ["# M3三任务、多随机种子结果解释", "", summary["overall_M3_vs_M0"]["conclusion"], "",
             "## M3与M0的固定主指标对照", "",
             "增益统一为正值有利于M3：热稳定性和功能描述取M3−M0，保留时间取M0−M3。不同任务的数值单位保持分开。", ""]
    lines += _comparison_table(comparisons["M0"])
    lines += ["", "完整三种子才计算均值和样本标准差。赢/平/输按逐种子增益的正/零/负统计；缺失位置保留null。",
              "“3/3同方向有利”沿用原独立分析的方向标记并核对逐种子方向，表示训练重复的一致性，不代表统计显著。", "",
              "## ESM参照", "", "同样以正值表示M3更好；ESM与连接器的参数量、在线语言模型计算量不同。", ""]
    lines += _comparison_table(comparisons["ESM"])
    lines += ["", "## 各组冻结指标", "", "以下逐项列出原arm_summaries中的全部指标，包括主指标和次指标。", "",
              "|任务|方法|底座|指标|有效种子|均值 ± 样本SD|", "|---|---|---|---|---:|---:|"]
    for row in secondary:
        lines.append("|{}|{}|{}|{}|{}|{} ± {}|".format(TASK_NAMES.get(row["task"], row["task"]), row["method"],
            row["backbone"] or "none", row["metric"], row["n_valid_seeds"], _number(row["mean"]), _number(row["sample_sd"])))
    lines += ["", "## 解释范围与恢复记录", "",
              "语言生成和C1通道见原完整分析中的各案例结果，独立于主预测头；本摘要不补造缺失数值。",
              "功能描述任务覆盖31候选的描述检索，以及生成文字的严格目录命中；该基准评价相对排序与目录命中。开放式功能描述的事实真伪需要另外的验证。",
              "三个随机种子是训练重复，不是生物学独立重复。结果方向或均值变化不自动给出模块改善的因果机制。",
              summary["recovery_context"], "", "## 可追溯来源", "",
              f"- 完整独立分析：`{analysis_path.name}`；SHA256 `{summary['analysis_sha256']}`。",
              f"- 冻结计划SHA256：`{summary['plan_sha256']}`。",
              f"- 冻结数据manifest SHA256：`{summary['data_manifest_sha256']}`。",
              "- 原独立分析保留45个固定位置；本报告不按分数选择种子、不修改评分、不改变验收状态。", ""]
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path, summary_path = output_dir / "REPORT.md", output_dir / "summary.json"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"report_path": str(report_path), "summary_path": str(summary_path), "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = write_summary(args.analysis, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "summary"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

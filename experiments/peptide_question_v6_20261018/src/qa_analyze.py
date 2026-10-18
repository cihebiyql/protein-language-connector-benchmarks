"""Independent raw-answer re-scoring and the frozen v6 continue/pause decision."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

if __package__:
    from .qa_metrics import QUESTIONS, score
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.qa_metrics import QUESTIONS, score


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def compare_scores(recomputed, saved, path="score", tolerance=1e-12):
    """Compare every re-scored field; saved runtime-only extras are permitted."""
    if isinstance(recomputed, dict):
        if not isinstance(saved, dict) or not set(recomputed).issubset(saved):
            raise ValueError(f"Saved score fields differ: {path}")
        for key, value in recomputed.items():
            compare_scores(value, saved[key], f"{path}.{key}", tolerance)
    elif isinstance(recomputed, list):
        if not isinstance(saved, list) or len(saved) != len(recomputed):
            raise ValueError(f"Saved score list differs: {path}")
        for index, (left, right) in enumerate(zip(recomputed, saved)):
            compare_scores(left, right, f"{path}[{index}]", tolerance)
    elif isinstance(recomputed, (int, float)) and not isinstance(recomputed, bool):
        if (isinstance(saved, bool) or not isinstance(saved, (int, float))
                or not math.isfinite(saved) or abs(recomputed - saved) > tolerance):
            raise ValueError(f"Saved score numeric mismatch: {path}")
    elif type(recomputed) is not type(saved) or recomputed != saved:
        raise ValueError(f"Saved score value mismatch: {path}")


def descriptive(values, seeds):
    if len(values) != 3 or len(seeds) != 3:
        raise ValueError("Mean/sample SD require all three frozen seeds")
    if any(value is None or not math.isfinite(value) for value in values):
        raise ValueError("Complete finite seed scores required")
    return {"seeds": list(seeds), "values": list(values), "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values), "n_training_repeats": 3}


def _get(run_scores, arm, seed, condition="normal"):
    return run_scores[f"{arm}__seed{seed}"][condition]


def paired_group_bootstrap(run_scores, seeds, policy):
    """Shared cluster resamples; each draw retains all questions and all seeds."""
    candidates = policy["candidate_memory_arms"]
    reference = "TEXT"
    base = _get(run_scores, reference, seeds[0])["per_peptide"]
    identifiers = [row["id"] for row in base]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Bootstrap requires unique peptide identities")
    group_ids = [row["group_id"] for row in base]
    if any(not isinstance(group, str) or not group for group in group_ids):
        raise ValueError("Bootstrap requires nonempty string connected-group IDs")
    groups = sorted(set(group_ids))
    group_lookup = {group: index for index, group in enumerate(groups)}
    membership = np.asarray([group_lookup[group] for group in group_ids], dtype=np.int64)
    truth = np.asarray([[row["truth"][question] == "yes" for question in ("rt", "ccs")]
                        for row in base], dtype=np.bool_)
    # One [property,class,group] denominator shared by every arm and seed.
    denominators = np.zeros((2, 2, len(groups)), dtype=np.float64)
    correct_counts = np.zeros((len(candidates) + 1, len(seeds), 2, 2, len(groups)), dtype=np.float64)
    arms = [reference] + list(candidates)
    for question_index in range(2):
        for class_index in range(2):
            positions = truth[:, question_index] == bool(class_index)
            denominators[question_index, class_index] = np.bincount(membership[positions], minlength=len(groups))
    for arm_index, arm in enumerate(arms):
        for seed_index, seed in enumerate(seeds):
            records = _get(run_scores, arm, seed)["per_peptide"]
            by_id = {row["id"]: row for row in records}
            if len(records) != len(base) or set(by_id) != set(identifiers):
                raise ValueError("Paired bootstrap peptide panels differ")
            for index, identifier in enumerate(identifiers):
                row = by_id[identifier]
                if row["group_id"] != group_ids[index] or row["truth"] != base[index]["truth"]:
                    raise ValueError("Paired bootstrap group or truth changed")
                for question_index, question in enumerate(("rt", "ccs")):
                    class_index = int(truth[index, question_index])
                    correct_counts[arm_index, seed_index, question_index, class_index, membership[index]] += int(row["correct"][question])
    replicates = policy["paired_group_bootstrap_replicates"]
    if type(replicates) is not int or replicates < 1:
        raise ValueError("Positive bootstrap replicate budget required")
    rng = np.random.default_rng(policy["bootstrap_seed"])
    resample_digest = hashlib.sha256()
    gains = np.empty((replicates, len(candidates)), dtype=np.float64)
    for replicate in range(replicates):
        drawn = rng.integers(0, len(groups), size=len(groups), dtype=np.int64)
        resample_digest.update(drawn.astype("<i8", copy=False).tobytes())
        multiplicity = np.bincount(drawn, minlength=len(groups)).astype(np.float64)
        denominator = np.einsum("pcg,g->pc", denominators, multiplicity)
        if np.any(denominator <= 0):
            raise ValueError("Undefined balanced accuracy in a frozen group-bootstrap draw")
        numerator = np.einsum("aspcg,g->aspc", correct_counts, multiplicity)
        primary = (numerator / denominator[None, None]).mean(axis=(1, 2, 3))
        gains[replicate] = primary[1:] - primary[0]
    quantile = policy["one_sided_lower_quantile"]
    return {
        "method": "paired connected-group percentile bootstrap; mean across the same three seeds in every resample",
        "sampling_unit": "connected_group", "group_count": len(groups), "peptide_count": len(base),
        "replicates": replicates, "seed": policy["bootstrap_seed"],
        "seeds_kept_together": list(seeds), "questions_kept_together": list(QUESTIONS),
        "same_resamples_for_both_candidate_arms": True, "resample_indices_sha256": resample_digest.hexdigest(),
        "lower_quantile": quantile, "quantile_method": "linear",
        "familywise_one_sided_error": policy["two_candidate_arms_familywise_one_sided_error"],
        "contrasts": {arm: {"lower_primary_gain": float(np.quantile(gains[:, index], quantile, method="linear")),
                             "bootstrap_mean_primary_gain": float(gains[:, index].mean())}
                      for index, arm in enumerate(candidates)},
        "interpretation": "training repeats share one fixed processed-data test panel; not independent biological datasets",
    }


def analyze_scores(run_scores, config):
    """Pure decision calculation from already independently re-scored runs."""
    seeds, arms, policy = config["seeds"], config["arms"], config["decision"]
    if tuple(seeds) != (17, 42, 2026) or tuple(arms) != ("TEXT", "M0", "M3"):
        raise ValueError("Fixed nine-case matrix differs")
    expected = {f"{arm}__seed{seed}" for arm in arms for seed in seeds}
    if set(run_scores) != expected:
        raise ValueError("Decision requires every frozen case, without dropped seeds")
    arm_summaries = {}
    for arm in arms:
        conditions = ("normal", "shuffled", "removed") if arm == "TEXT" else ("normal", "mean", "shuffled", "off")
        summaries = {}
        for condition in conditions:
            summaries[condition] = {
                "primary": descriptive([_get(run_scores, arm, seed, condition)["primary"] for seed in seeds], seeds),
                "joint_all_correct": descriptive([_get(run_scores, arm, seed, condition)["joint_all_correct"] for seed in seeds], seeds),
                "questions": {question: {metric: descriptive([
                    _get(run_scores, arm, seed, condition)["questions"][question][metric] for seed in seeds], seeds)
                    for metric in ("accuracy", "balanced_accuracy", "valid_rate")} for question in QUESTIONS},
            }
        arm_summaries[arm] = summaries
    bootstrap = paired_group_bootstrap(run_scores, seeds, policy)
    contrasts = {}
    for arm in policy["candidate_memory_arms"]:
        primary = [_get(run_scores, arm, seed)["primary"] - _get(run_scores, "TEXT", seed)["primary"] for seed in seeds]
        endpoint = {question: [_get(run_scores, arm, seed)["questions"][question]["balanced_accuracy"]
                              - _get(run_scores, "TEXT", seed)["questions"][question]["balanced_accuracy"]
                              for seed in seeds] for question in ("rt", "ccs")}
        joint = [_get(run_scores, arm, seed)["joint_all_correct"] - _get(run_scores, "TEXT", seed)["joint_all_correct"] for seed in seeds]
        controls = {condition: [_get(run_scores, arm, seed)["primary"]
                               - _get(run_scores, arm, seed, condition)["primary"] for seed in seeds]
                    for condition in ("mean", "shuffled", "off")}
        endpoint_gate = any(all(value > 0 for value in endpoint[question])
                            and statistics.mean(endpoint[other]) >= policy["other_endpoint_mean_gain_floor"]
                            for question, other in (("rt", "ccs"), ("ccs", "rt")))
        gates = {
            "minimum_mean_primary_gain": statistics.mean(primary) >= policy["minimum_mean_primary_gain"],
            "all_three_seed_primary_gains_positive": all(value > 0 for value in primary),
            "independent_endpoint_consistency_and_other_endpoint_floor": endpoint_gate,
            "joint_all_three_answers_mean_gain_positive": statistics.mean(joint) > 0,
            "bootstrap_lower_primary_gain_above_zero": bootstrap["contrasts"][arm]["lower_primary_gain"] > 0,
        }
        for condition in policy["molecular_controls"]:
            gates[f"{condition}_minimum_mean_drop"] = statistics.mean(controls[condition]) >= policy["minimum_control_mean_primary_drop"]
            gates[f"{condition}_all_three_seed_drops_positive"] = all(value > 0 for value in controls[condition])
        contrasts[arm] = {"primary_gain": descriptive(primary, seeds),
                          "endpoint_gains": {question: descriptive(values, seeds) for question, values in endpoint.items()},
                          "joint_all_correct_gain": descriptive(joint, seeds),
                          "control_primary_drops": {condition: descriptive(values, seeds) for condition, values in controls.items()},
                          "gates": gates, "passes_all_frozen_gates": all(gates.values())}
    passing = [arm for arm in policy["candidate_memory_arms"] if contrasts[arm]["passes_all_frozen_gates"]]
    # Fixed candidate order resolves an exact tie after comparing mean primary.
    selected = max(passing, key=lambda arm: arm_summaries[arm]["normal"]["primary"]["mean"]) if passing else None
    return {"schema": "same-peptide-final-analysis.v1",
            "status": "complete_stable_gain" if selected else "complete_direction_paused",
            "decision": "continue_selected_arm" if selected else "pause_connector_direction_preserve_all_results",
            "selected_arm": selected, "accepted_cases": 9, "expected_cases": 9,
            "seeds": list(seeds), "arms": arm_summaries, "contrasts": contrasts,
            "bootstrap": bootstrap, "frozen_decision_policy": policy,
            "per_seed": {case: {condition: {key: value for key, value in result.items() if key != "per_peptide"}
                                 for condition, result in conditions.items()} for case, conditions in run_scores.items()},
            "scope_limit": "Within the fixed ProteomeTools processed-data peptide analytical-property QA panel; this does not establish universal biological failure or success.",
            "repeat_interpretation": "Three training seeds share the same test peptides and are not three independent biological datasets."}


def _load_verifier(scope):
    path = Path(scope) / "resources" / "node_pipeline.py"
    spec = importlib.util.spec_from_file_location("v6_final_node_integrity_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyze_scope(scope, verifier=None):
    """Require all fits and freeze verification before opening test labels."""
    scope = Path(scope).resolve()
    config = read_json(scope / "config.json")
    expected = [{"id": f"{arm}__seed{seed}", "arm": arm, "seed": seed}
                for arm in config["arms"] for seed in config["seeds"]]
    receipts, problems = {}, []
    verifier = verifier or _load_verifier(scope)
    try:
        verifier.verify_freeze(scope)
    except Exception as error:
        problems.append({"stage": "frozen_inputs", "reason": type(error).__name__})
    if not problems:
        for case in expected:
            try:
                receipts[case["id"]] = verifier.verify_result(scope, case, config)
            except Exception as error:
                problems.append({"case": case["id"], "stage": "artifact_integrity", "reason": type(error).__name__})
    result = {"schema": "same-peptide-final-analysis.v1", "status": "technical_inconclusive",
              "decision": "technical_inconclusive_do_not_claim_scientific_failure", "selected_arm": None,
              "expected_cases": 9, "accepted_cases": len(receipts), "problems": problems,
              "no_scientific_stop_decision": True}
    if not problems:
        current_case, current_condition = None, None
        try:
            rows = read_jsonl(scope / config.get("data_root", "data") / "test.jsonl")
            run_scores = {}
            for case in expected:
                current_case = case["id"]
                directory = scope / "runs" / case["id"] / "formal"
                saved = read_json(directory / "test_summary.json")
                conditions = ("normal", "shuffled", "removed") if case["arm"] == "TEXT" else ("normal", "mean", "shuffled", "off")
                run_scores[case["id"]] = {}
                for condition in conditions:
                    current_condition = condition
                    recomputed = score(rows, read_jsonl(directory / f"test_{condition}.jsonl"), config["thresholds"])
                    compare_scores(recomputed, saved[condition], f"{case['id']}.{condition}")
                    run_scores[case["id"]][condition] = recomputed
            result = analyze_scores(run_scores, config)
            result["independent_rescore"] = {"all_raw_answer_metrics_match_saved": True, "absolute_tolerance": 1e-12,
                                             "test_panel_sha256": sha256(scope / config.get("data_root", "data") / "test.jsonl")}
        except Exception as error:
            problem = {"stage": "independent_rescore_or_decision", "reason": type(error).__name__,
                       "case": current_case, "condition": current_condition}
            # These messages are constructed above from relative case/metric
            # names and cannot expose a node's absolute filesystem paths.
            if str(error).startswith("Saved score "):
                problem["detail"] = str(error)
            result["problems"].append(problem)
    result["artifact_verification"] = receipts
    result["config_sha256"] = sha256(scope / "config.json")
    result["generated_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def report_text(summary):
    if summary["status"] == "technical_inconclusive":
        return ("九组实验尚未全部通过完整性及原始答案复核，当前结论为技术上未完成。\n\n"
                f"已通过基础验收 {summary['accepted_cases']}/9 组；不据此暂停科学方向，也不补删低分种子。"
                "问题阶段和记录见 summary.json。\n")
    conclusion = (f"本轮 {summary['selected_arm']} 通过全部预先冻结条件，可以继续该方案。" if summary["selected_arm"]
                  else "本轮两个连接方案均未通过全部预先冻结条件，暂停连接器方向，保留全部结果。")
    lines = [conclusion, "", "九组拟合均完成并通过原始答案复核；三个训练种子为17、42、2026。数据划分、阈值、问题、评价规则和停止条件保持冻结。", "",
             "| 方案 | 两性质平均平衡准确率（均值±样本SD） | RT平衡准确率 | CCS平衡准确率 | 同肽三问全对率 |", "|---|---:|---:|---:|---:|"]
    for arm in ("TEXT", "M0", "M3"):
        metrics = summary["arms"][arm]["normal"]
        primary = metrics["primary"]
        lines.append(f"| {arm} | {primary['mean']:.4f} ± {primary['sample_sd']:.4f} | {metrics['questions']['rt']['balanced_accuracy']['mean']:.4f} | {metrics['questions']['ccs']['balanced_accuracy']['mean']:.4f} | {metrics['joint_all_correct']['mean']:.4f} |")
    lines += ["", "主指标只包含RT与CCS两个实测性质；第三问是组合判断。所有无效生成在完整分母中计错。", ""]
    gate_labels = {
        "minimum_mean_primary_gain": "平均提升至少2个百分点",
        "all_three_seed_primary_gains_positive": "三个种子主指标均提升",
        "independent_endpoint_consistency_and_other_endpoint_floor": "独立性质的跨种子一致性及另一性质下降限制",
        "joint_all_three_answers_mean_gain_positive": "同肽三问全对率提升",
        "bootstrap_lower_primary_gain_above_zero": "分组bootstrap下界大于零",
        "mean_minimum_mean_drop": "正确记忆较均值记忆平均提升至少2个百分点",
        "mean_all_three_seed_drops_positive": "三个种子均依赖正确记忆而非均值记忆",
        "shuffled_minimum_mean_drop": "正确记忆较打乱记忆平均提升至少2个百分点",
        "shuffled_all_three_seed_drops_positive": "三个种子均依赖正确记忆而非打乱记忆",
    }
    for arm in ("M0", "M3"):
        contrast = summary["contrasts"][arm]
        failed = [gate_labels.get(name, name) for name, passed in contrast["gates"].items() if not passed]
        lines.append(f"{arm} 相对TEXT主指标平均变化 {contrast['primary_gain']['mean']:+.4f}；三种子变化为 "
                     + "、".join(f"{value:+.4f}" for value in contrast["primary_gain"]["values"])
                     + f"；配对分组bootstrap单侧校正下界 {summary['bootstrap']['contrasts'][arm]['lower_primary_gain']:+.4f}。"
                     + ("全部门槛通过。" if not failed else "未通过条件：" + "、".join(failed) + "。"))
    lines += ["", f"bootstrap以肽连通组重采样，同肽三问与三个训练种子始终一起保留；两候选使用相同{summary['bootstrap']['replicates']}次重采样，单侧下分位各2.5%。关闭支路结果单列，不替代均值记忆及打乱记忆控制条件。", "",
              "三个种子是在同一测试集上的训练重复，不是三个独立生物数据集。本结论限于这个冻结的肽分析性质问答基准，不等同于任意生物功能能力的普遍成立或否定。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True)
    arguments = parser.parse_args(argv)
    scope = Path(arguments.scope).resolve()
    summary = analyze_scope(scope)
    atomic_json(scope / "analysis" / "summary.json", summary)
    (scope / "analysis" / "REPORT.md").write_text(report_text(summary), encoding="utf-8")
    active_path = scope / "ACTIVE_EXPERIMENT.json"
    active = read_json(active_path) if active_path.exists() else {}
    active.update({"status": summary["status"], "analysis": "analysis/summary.json", "report": "analysis/REPORT.md",
                   "selected_arm": summary.get("selected_arm"), "updated_utc": summary["generated_utc"],
                   "scheduled_monitor": False, "preserve_all_previous_results": True})
    atomic_json(active_path, active)
    print(json.dumps({"status": summary["status"], "accepted_cases": summary["accepted_cases"],
                      "selected_arm": summary.get("selected_arm")}, ensure_ascii=False), flush=True)
    return 2 if summary["status"] == "technical_inconclusive" else 0


if __name__ == "__main__":
    raise SystemExit(main())

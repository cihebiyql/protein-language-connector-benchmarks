"""Pre-import device budget wrapper and independently checked V5 case evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

from .mt_supervise import GIB, PHYSICAL_GPUS, read_json, write_json, process_identity, same_process


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pins(plan):
    for category in ("source_pins", "asset_pins"):
        pins = plan.get(category)
        if not isinstance(pins, dict) or not pins:
            raise ValueError(f"Missing frozen {category}")
        for path, expected in pins.items():
            if not Path(path).is_absolute() or sha(path) != expected:
                raise ValueError(f"Frozen {category} mismatch: {path}")


def command(plan, case, output, device, *, phase=None):
    paths = plan["paths"]
    phase = phase or plan.get("phase", "formal")
    args = [paths["python"], "-m", "src.mt_train", "--config", paths["config"],
        "--task", case["task"], "--method", case["method"], "--backbone", case["backbone"],
        "--seed", str(case["seed"]), "--data-root", paths["data_root"], "--cache-root", paths["cache_root"],
        "--output", str(output), "--phase", phase, "--device", device]
    if phase == "train-canary":
        args += ["--max-updates", "4"]
    return args


def guarded_plan(spec_path, *, descendant=False):
    spec = read_json(spec_path)
    # A controller crash in Popen's first milliseconds must not leave an
    # unguarded model process. Do not import torch until the independent guard
    # and this exact standalone group have receipts.
    supervision = Path(spec_path).parent / "supervision"
    guard_receipt = supervision / "guard_started.json"
    until = time.monotonic() + 30
    while not guard_receipt.exists() and time.monotonic() < until:
        time.sleep(.02)
    if not guard_receipt.exists():
        raise RuntimeError("Independent guard was not installed before training")
    guard_identity = read_json(guard_receipt)["identity"]
    started = read_json(supervision / "started.json")
    own = process_identity(os.getpid())
    leader = started["identity"]
    valid_worker = (own and own["pgid"] == leader["pgid"] and own["sid"] == leader["sid"]
                    and own["start_ticks"] >= leader["start_ticks"]) if descendant else leader["pid"] == os.getpid()
    if not guard_identity or not same_process(guard_identity) or not valid_worker or not same_process(leader):
        raise RuntimeError("Training guard/process binding is not live and exact")
    plan = read_json(spec["plan_path"])
    if sha(spec["plan_path"]) != spec["plan_sha256"]:
        raise ValueError("Plan changed after worker admission")
    verify_pins(plan)
    config = read_json(plan["paths"]["config"])
    configured_cap = config.get("numerical_policy", {}).get("gpu_allocator_gib", {}).get(spec["case"]["backbone"])
    if spec["gpu"] is not None and configured_cap is not None and float(configured_cap) * GIB != spec["gpu_memory_budget_bytes"]:
        raise ValueError("Trainer allocator setting differs from supervised cap")
    return plan, spec


def train_process(spec_path, plan, spec, output, phase, receipt_path):
    gpu = spec["gpu"]
    if gpu is None:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("CPU baseline must have no visible GPU")
        device = "cpu"
    else:
        if gpu not in PHYSICAL_GPUS or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
            raise ValueError("Physical GPU assignment mismatch")
        device = "cuda:0"
    # This module and all imports above are model-free. Set the allocator cap
    # before importing the training/model implementation.
    import torch
    torch.set_num_threads(2)
    if gpu is not None:
        total = torch.cuda.get_device_properties(0).total_memory
        fraction = spec["gpu_memory_budget_bytes"] / total
        if not 0 < fraction <= 1:
            raise ValueError("Allocator cap exceeds physical GPU memory")
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    else:
        total, fraction = None, None
    receipt = {"gpu": gpu, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_memory_budget_bytes": spec["gpu_memory_budget_bytes"], "allocator_fraction": fraction,
        "physical_total_bytes": total, "cpu_threads": 2, "plan_sha256": spec["plan_sha256"]}
    receipt.update(phase=phase, pid=os.getpid(), pgid=os.getpgrp() if os.name == "posix" else None)
    write_json(receipt_path, receipt)
    argv = command(plan, spec["case"], output, device, phase=phase)
    sys.argv = argv[2:]
    runpy.run_module("src.mt_train", run_name="__main__", alter_sys=True)


def qualification_acceptance(plan, attempt, case):
    """TRAIN-only technical qualification; predictive accuracy is never gated."""
    root = Path(attempt)
    spec = read_json(root / "spec.json")
    if spec["case"] != case or spec["plan_sha256"] != sha(spec["plan_path"]):
        raise ValueError("Qualification case/plan binding changed")
    verify_pins(plan)
    output = root / "qualification"
    manifest = read_json(output / "run_manifest.json")
    if (manifest.get("status") != "complete" or manifest.get("phase") != "train-canary"
            or manifest.get("scope_id") != plan["scope_id"]
            or manifest.get("config_sha256") != sha(plan["paths"]["config"])
            or manifest.get("test_labels_used_for_fit") is not False):
        raise ValueError("Qualification is not a complete frozen TRAIN-only run")
    for key in ("task", "method", "backbone", "seed"):
        if manifest.get(key) != case[key]:
            raise ValueError("Qualification case identity mismatch")
    train = manifest.get("training", {})
    if train.get("optimizer_steps") != 4 or train.get("examples_seen") != 16 or train.get("epochs_completed") != 0:
        raise ValueError("Qualification requires exactly four updates and sixteen TRAIN examples")
    qualification = manifest.get("canary_qualification", {})
    if (qualification.get("train_rows_scored") != 8 or qualification.get("development_only") is not True
            or any(qualification.get(key) is not False for key in ("validation_opened", "calibration_opened", "test_opened"))):
        raise ValueError("Qualification opened an evaluation split or lacks TRAIN-only evidence")
    updates = [r for r in prediction_rows(output / "progress.jsonl") if r.get("event") == "optimizer_step"]
    if [r.get("optimizer_step") for r in updates] != [1, 2, 3, 4] or updates[-1].get("examples_seen") != 16:
        raise ValueError("Qualification optimizer evidence is missing or incomplete")
    from .mt_data import load_task, load_catalog
    rows = load_task(plan["paths"]["data_root"], case["task"], "train")[:8]
    readout = prediction_rows(output / "train_readout_predictions.jsonl")
    exact_ids(readout, [r["sample_id"] for r in rows], "TRAIN qualification readout")
    if case["task"] == "elm":
        ids = {r["candidate_id"] for r in load_catalog(plan["paths"]["data_root"])}
        if any(len(r.get("ranking", [])) != len(ids) or set(r.get("ranking", [])) != ids for r in readout):
            raise ValueError("Qualification readout lacks complete candidate rankings")
    elif any(type(r.get("prediction")) not in (int, float) or not math.isfinite(r["prediction"]) for r in readout):
        raise ValueError("Qualification readout has nonfinite values")
    budget = read_json(root / "qualification_device_budget.json")
    if (budget.get("gpu") != spec["gpu"] or budget.get("gpu_memory_budget_bytes") != spec["gpu_memory_budget_bytes"]
            or budget.get("plan_sha256") != spec["plan_sha256"] or budget.get("phase") != "train-canary"):
        raise ValueError("Qualification allocator receipt differs from the supervised budget")
    runtime = manifest.get("runtime", {})
    if spec["gpu"] is not None:
        for key in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
            value = runtime.get(key)
            if type(value) is not int or not 0 < value <= spec["gpu_memory_budget_bytes"]:
                raise ValueError("Qualification actual allocator peak exceeds its bound")
    required = {"run_manifest.json", "progress.jsonl", "training_history.json", "checkpoint.pt", "selection.json",
                "train_readout_predictions.jsonl"}
    for entry in manifest.get("artifacts", {}).values():
        path = (output / entry["path"]).resolve(strict=True)
        if not path.is_relative_to(output.resolve()) or sha(path) != entry["sha256"]:
            raise ValueError("Qualification artifact SHA mismatch")
        required.add(path.relative_to(output.resolve()).as_posix())
    if case["method"] != "ESM":
        generated = prediction_rows(output / "train_LM_predictions.jsonl")
        exact_ids(generated, [r["sample_id"] for r in rows], "TRAIN qualification language")
        limit = 256 if case["task"] == "elm" else 24
        if any(not isinstance(r.get("text"), str) or r.get("max_new_tokens") != limit
               or type(r.get("generated_tokens")) is not int or not 0 <= r["generated_tokens"] <= limit for r in generated):
            raise ValueError("Qualification did not use the frozen generation shape")
        if read_json(output / "train_anchor_qualification.json").get("passed") is not True:
            raise ValueError("Qualification future-token anchor check failed")
        required.update(("train_LM_predictions.jsonl", "train_anchor_qualification.json"))
    elapsed = manifest.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Qualification elapsed-time evidence is missing")
    return {"schema": "mt-train-qualification.v1", "case_id": case["case_id"],
        "plan_sha256": spec["plan_sha256"], "training_updates": 4, "training_examples": 16,
        "excluded_from_formal_fit_counts": True, "accuracy_used_for_qualification": False,
        "elapsed_seconds": elapsed, "peak_cuda_allocated_bytes": runtime.get("peak_cuda_allocated_bytes"),
        "peak_cuda_reserved_bytes": runtime.get("peak_cuda_reserved_bytes"),
        "artifact_sha256": {name: sha(output / name) for name in sorted(required)},
        "device_budget_sha256": sha(root / "qualification_device_budget.json")}


def run_training_child(spec_path, phase, output, receipt_path):
    spec = read_json(spec_path)
    plan = read_json(spec["plan_path"])
    argv = [plan["paths"]["python"], "-m", "src.mt_case", "--train-child", "--spec", str(spec_path),
            "--phase", phase, "--output", str(output), "--receipt", str(receipt_path)]
    started = time.time()
    process = subprocess.Popen(argv, cwd=plan["paths"]["scope_root"])
    identity = process_identity(process.pid) if os.name == "posix" else {"pid": process.pid}
    code = process.wait()
    result = {"phase": phase, "pid": process.pid, "identity": identity, "command": argv,
              "exit_code": code, "started_unix": started, "finished_unix": time.time(),
              "fresh_process": True, "inherits_owned_process_group": True}
    write_json(Path(spec_path).parent / f"{phase}_process.json", result)
    if code:
        raise RuntimeError(f"{phase} child failed with exit code {code}; no implicit retry")
    return result


def preflight_then_formal(spec_path, plan, spec, *, run_child=None, qualify=None):
    root = Path(spec_path).parent
    run_child = run_child or run_training_child
    qualify = qualify or qualification_acceptance
    canary = run_child(spec_path, "train-canary", root / "qualification", root / "qualification_device_budget.json")
    receipt = qualify(plan, root, spec["case"])
    receipt["process_receipt_sha256"] = sha(root / "train-canary_process.json")
    write_json(root / "qualification_acceptance.json", receipt)
    formal = run_child(spec_path, "formal", root / "artifacts", root / "device_budget.json")
    if canary["pid"] == formal["pid"]:
        raise RuntimeError("Qualification/formal fits must use distinct fresh processes")


def worker(spec_path):
    plan, spec = guarded_plan(spec_path)
    if plan.get("require_train_canary") is True and plan.get("phase", "formal") == "formal":
        preflight_then_formal(spec_path, plan, spec)
    else:
        train_process(spec_path, plan, spec, Path(spec_path).parent / "artifacts",
                      plan.get("phase", "formal"), Path(spec_path).parent / "device_budget.json")


def prediction_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def exact_ids(rows, expected, filename):
    ids = [row.get("sample_id") for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError(f"Exact sample coverage mismatch: {filename}")


def accepted_attempt(plan, attempt, case):
    """Re-read actual evidence. File existence or producer 'complete' is insufficient."""
    verify_pins(plan)
    root = Path(attempt)
    spec = read_json(root / "spec.json")
    if spec["case"] != case or spec["plan_sha256"] != sha(spec["plan_path"]):
        raise ValueError("Attempt plan/case binding mismatch")
    qualification_binding = None
    if plan.get("require_train_canary") is True and plan.get("phase", "formal") == "formal":
        expected = qualification_acceptance(plan, root, case)
        expected["process_receipt_sha256"] = sha(root / "train-canary_process.json")
        if read_json(root / "qualification_acceptance.json") != expected:
            raise ValueError("Qualification acceptance is missing, stale or forged")
        canary_process = read_json(root / "train-canary_process.json")
        formal_process = read_json(root / "formal_process.json")
        if (canary_process.get("exit_code") != 0 or formal_process.get("exit_code") != 0
                or canary_process.get("pid") == formal_process.get("pid")
                or canary_process.get("fresh_process") is not True or formal_process.get("fresh_process") is not True
                or canary_process.get("phase") != "train-canary" or formal_process.get("phase") != "formal"):
            raise ValueError("Qualification and formal fits lack distinct successful process evidence")
        if (read_json(root / "qualification_device_budget.json").get("pid") != canary_process["pid"]
                or read_json(root / "device_budget.json").get("pid") != formal_process["pid"]):
            raise ValueError("Actual phase process and allocator receipts disagree")
        qualification_binding = {"acceptance_sha256": sha(root / "qualification_acceptance.json"),
            "train_canary_process_sha256": sha(root / "train-canary_process.json"),
            "formal_process_sha256": sha(root / "formal_process.json"), **expected}
    completed = read_json(root / "supervision/finished.json")
    if (completed.get("status") != "completed" or completed.get("exit_code") != 0
            or completed.get("remaining_group_members") or completed.get("guard", {}).get("reason") != "process_group_exited"):
        raise ValueError("Actual supervision did not complete cleanly")
    device = read_json(root / "device_budget.json")
    if (device["gpu"] != spec["gpu"] or device["gpu_memory_budget_bytes"] != spec["gpu_memory_budget_bytes"]
            or device["plan_sha256"] != spec["plan_sha256"]):
        raise ValueError("Actual device budget differs from admission")
    artifacts = root / "artifacts"
    manifest = read_json(artifacts / "run_manifest.json")
    if manifest.get("scope_id") != plan["scope_id"] or manifest.get("config_sha256") != sha(plan["paths"]["config"]):
        raise ValueError("Producer scope/config binding mismatch")
    if manifest.get("test_labels_used_for_fit") is not False:
        raise ValueError("Producer does not retain the frozen test boundary")
    for path, expected in plan["source_pins"].items():
        source = Path(path)
        if source.parent.name == "src" and source.suffix == ".py" and manifest.get("code_sha256", {}).get(source.name) != expected:
            raise ValueError("Producer did not execute the frozen source closure")
    if manifest.get("data_manifest_sha256") != sha(Path(plan["paths"]["data_root"]) / "manifest.json"):
        raise ValueError("Producer data view closure mismatch")
    for key in ("task", "method", "backbone", "seed"):
        if manifest.get(key) != case[key]:
            raise ValueError(f"Producer case metadata mismatch: {key}")
    if manifest.get("status") != "complete" or manifest.get("phase") != plan.get("phase", "formal"):
        raise ValueError("Producer not complete in frozen phase")
    # Checkpoint must contain actual finite trainable CPU tensors.
    import torch
    checkpoint = torch.load(artifacts / "checkpoint.pt", map_location="cpu", weights_only=True)
    state = checkpoint.get("trainable_state", checkpoint.get("state_dict", checkpoint.get("trainable_state_dict", checkpoint)))
    tensors = [v for v in state.values() if isinstance(v, torch.Tensor)]
    if not tensors or any(v.device.type != "cpu" or not torch.isfinite(v).all() for v in tensors):
        raise ValueError("Missing/nonfinite actual trainable checkpoint tensors")
    if any(name.startswith("base_lm.") for name in state) or any(v.is_floating_point() and v.dtype != torch.float32 for v in tensors):
        raise ValueError("Checkpoint contains frozen LM weights or non-FP32 trainable tensors")
    for key in ("task", "method", "backbone", "seed", "config_sha256"):
        if checkpoint.get("identity", {}).get(key) != manifest.get(key):
            raise ValueError("Checkpoint identity differs from actual run")
    config = read_json(plan["paths"]["config"])
    from .mt_data import load_task
    train = load_task(plan["paths"]["data_root"], case["task"], "train")
    epochs = int(config["training"]["epochs"])
    effective = int(config["training"].get("effective_batch_size", 4))
    if plan.get("phase") == "train-canary":
        expected_steps, expected_examples = 4, 16
    else:
        expected_steps, expected_examples = epochs * math.ceil(len(train) / effective), epochs * len(train)
    progress = prediction_rows(artifacts / "progress.jsonl")
    # Exact schema is bound to the trainer; progress contains optimizer updates.
    updates = [r for r in progress if r.get("event") == "optimizer_step"]
    if len(updates) != expected_steps:
        raise ValueError(f"Optimizer evidence count {len(updates)} != {expected_steps}")
    if [r.get("optimizer_step") for r in updates] != list(range(1, expected_steps + 1)):
        raise ValueError("Actual optimizer-step sequence is not contiguous")
    if updates[-1].get("examples_seen") != expected_examples:
        raise ValueError("Actual TRAIN example count differs from frozen budget")
    if (manifest.get("training", {}).get("optimizer_steps") != expected_steps
            or manifest.get("training", {}).get("examples_seen") != expected_examples):
        raise ValueError("Producer summary and actual optimizer evidence disagree")
    required = ["run_manifest.json", "training_history.json", "progress.jsonl", "checkpoint.pt"]
    if plan.get("phase", "formal") == "formal":
        test = load_task(plan["paths"]["data_root"], case["task"], "test")
        ids = [r["sample_id"] for r in test]
        readout = prediction_rows(artifacts / "readout_predictions.jsonl")
        exact_ids(readout, ids, "readout_predictions.jsonl")
        if case["task"] == "elm":
            from .mt_data import load_catalog
            candidate_ids = {r["candidate_id"] for r in load_catalog(plan["paths"]["data_root"])}
            if any(len(r.get("ranking", [])) != len(candidate_ids) or set(r.get("ranking", [])) != candidate_ids for r in readout):
                raise ValueError("Readout ranking is not a complete candidate permutation")
        elif any(type(r.get("prediction")) not in (int, float) or not math.isfinite(r["prediction"]) for r in readout):
            raise ValueError("Readout contains missing or nonfinite numeric predictions")
        if case["method"] != "ESM":
            exact_ids(prediction_rows(artifacts / "LM_predictions.jsonl"), ids, "LM_predictions.jsonl")
            exact_ids(prediction_rows(artifacts / "C1_predictions.jsonl"), ids, "C1_predictions.jsonl")
            required += ["LM_predictions.jsonl", "C1_predictions.jsonl"]
        if case["task"] == "rt":
            calibration = load_task(plan["paths"]["data_root"], "rt", "calibration")
            exact_ids(prediction_rows(artifacts / "calibration_predictions.jsonl"),
                [r["sample_id"] for r in calibration], "calibration_predictions.jsonl")
            required.append("calibration_predictions.jsonl")
        required += ["selection.json", "model_frozen_before_test.json", "readout_predictions.jsonl",
                     "metrics.json"]
        selection = read_json(artifacts / "selection.json")
        if selection.get("checkpoint_sha256") != sha(artifacts / "checkpoint.pt"):
            raise ValueError("Selection does not bind the actual checkpoint")
        frozen = read_json(artifacts / "model_frozen_before_test.json")
        if frozen.get("checkpoint_sha256") != sha(artifacts / "checkpoint.pt"):
            raise ValueError("Test prediction boundary differs from selected checkpoint")
        if (frozen.get("test_opened") is not False or frozen.get("test_labels_used_for_fit") is not False
                or frozen.get("selection_sha256") != sha(artifacts / "selection.json")
                or selection.get("test_labels_used_for_selection") is not False
                or selection.get("calibration_labels_used_for_selection") is not False):
            raise ValueError("Selected model/test/calibration boundary is not frozen")
        validation = load_task(plan["paths"]["data_root"], case["task"], "selection" if case["task"] == "rt" else "validation")
        selected_predictions = read_json(artifacts / "selection_predictions.json")
        exact_ids(selected_predictions, [r["sample_id"] for r in validation], "selection_predictions.json")
        from .mt_metrics import score_regression, score_ranking
        if case["task"] == "elm":
            metric = score_ranking(validation, selected_predictions,
                [r["candidate_id"] for r in load_catalog(plan["paths"]["data_root"])])
        else:
            metric = score_regression(validation, selected_predictions)
        name = {"meltome": "spearman", "elm": "class_macro_mrr", "rt": "mae"}[case["task"]]
        if selection.get("name") != name or selection.get("value") != metric[name]:
            raise ValueError("Independent validation scoring differs from selected checkpoint criterion")
        history = read_json(artifacts / "training_history.json")
        if len(history) != epochs:
            raise ValueError("Actual epoch history differs from the frozen budget")
        key = lambda r: r["selection"]["value"] if case["task"] == "rt" else -r["selection"]["value"]
        best = min(history, key=key)
        if selection.get("epoch") != best["epoch"] or checkpoint.get("epoch") != best["epoch"]:
            raise ValueError("Checkpoint did not use the earliest best full-validation epoch")
        required.append("selection_predictions.json")
    return {"schema": "v5.accepted-attempt.v1", "case_id": case["case_id"],
        "expected_optimizer_steps": expected_steps, "expected_training_examples": expected_examples,
        "artifact_sha256": {name: sha(artifacts / name) for name in required},
        "run_manifest_sha256": sha(artifacts / "run_manifest.json"),
        "metrics_sha256": sha(artifacts / "metrics.json") if (artifacts / "metrics.json").is_file() else None,
        "plan_sha256": spec["plan_sha256"], "TRAIN_qualification": qualification_binding}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--train-child", action="store_true")
    parser.add_argument("--phase", choices=("train-canary", "formal"))
    parser.add_argument("--output")
    parser.add_argument("--receipt")
    args = parser.parse_args()
    if args.train_child:
        if not all((args.phase, args.output, args.receipt)):
            parser.error("Training child requires explicit phase, output and allocator receipt")
        plan, spec = guarded_plan(args.spec, descendant=True)
        train_process(args.spec, plan, spec, args.output, args.phase, args.receipt)
    else:
        worker(args.spec)


if __name__ == "__main__":
    main()

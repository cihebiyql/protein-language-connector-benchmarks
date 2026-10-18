"""Explicit, case-bound fourth-GPU resource override; frozen science stays intact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ADAPTER = Path(__file__).resolve()
SCOPE = ADAPTER.parent.parent
if str(SCOPE) not in sys.path:
    sys.path.insert(0, str(SCOPE))

from src import mt_case, mt_supervise

TARGET = "elm__M3__qwen3_4b_2507__seed17"
AUTHORIZED = (3, 4, 5, 6)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def absolute(path):
    value = Path(path)
    if not value.is_absolute():
        raise ValueError("Recovery paths must be absolute")
    return value.resolve(strict=True)


def validate(spec_path):
    spec_path = absolute(spec_path)
    spec = mt_supervise.read_json(spec_path)
    policy_path = absolute(spec["recovery_policy_path"])
    if sha(policy_path) != spec["recovery_policy_sha256"]:
        raise ValueError("Recovery policy SHA mismatch")
    if sha(ADAPTER) != spec["recovery_adapter_sha256"]:
        raise ValueError("Recovery adapter SHA mismatch")
    policy = mt_supervise.read_json(policy_path)
    if (policy.get("schema") != "v5-recovery-resource-policy.v1"
            or policy.get("case_id") != TARGET
            or policy.get("authorized_gpu_devices") != list(AUTHORIZED)
            or policy.get("original_gpu_devices") != [3, 4, 5]
            or type(policy.get("maximum_distinct_owned_gpus")) is not int
            or policy["maximum_distinct_owned_gpus"] != 4):
        raise ValueError("Recovery authorization must name this case and exactly four GPUs")
    scope = absolute(policy["scope_root"])
    if absolute(spec["cwd"]) != scope:
        raise ValueError("Recovery working directory differs from original scope")
    case_root = scope / "runs" / "recovery_retry_v1" / "cases" / TARGET
    if spec_path.name != "spec.json" or spec_path.parent.parent != case_root:
        raise ValueError("Recovery attempt must be one new directory below the exact case")
    plan_path = absolute(spec["plan_path"])
    if (plan_path != absolute(policy["original_plan_path"])
            or sha(plan_path) != policy["original_plan_sha256"]
            or spec["plan_sha256"] != policy["original_plan_sha256"]):
        raise ValueError("Original frozen plan binding mismatch")
    plan = mt_supervise.read_json(plan_path)
    if (absolute(plan["paths"]["scope_root"]) != scope
            or plan.get("gpu_devices") != [3, 4, 5]
            or plan.get("phase") != "formal" or plan.get("require_train_canary") is not True):
        raise ValueError("Recovery requires the original three-GPU qualified formal plan")
    matching = [case for case in plan["cases"] if case.get("case_id") == TARGET]
    if (len(matching) != 1 or spec.get("case") != matching[0]
            or matching[0] != {"case_id": TARGET, "task": "elm", "method": "M3",
                               "backbone": "qwen3_4b_2507", "seed": 17}):
        raise ValueError("Recovery is restricted to the exact frozen target case")
    if type(spec.get("gpu")) is not int or spec["gpu"] not in AUTHORIZED:
        raise ValueError("Recovery physical GPU is outside the four-card authorization")
    resources = plan["resource_policy"]
    expected = {"gpu_memory_budget_bytes": resources["gpu_memory_budget_bytes"]["qwen3_4b_2507"],
                "rss_limit_bytes": resources["rss_limit_bytes"],
                "storage_reserve_bytes": resources["storage_reserve_bytes"],
                "expected_host": resources["expected_host"],
                "timeout_seconds": plan["per_case_timeout_seconds"]}
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError("Recovery must retain the original memory and time budgets")
    expected_command = [plan["paths"]["python"], str(ADAPTER), "--mode", "worker",
                        "--spec", str(spec_path)]
    if spec.get("command") != expected_command:
        raise ValueError("Recovery worker command is not this exact bound adapter")
    # The original guard also verifies all asset pins before importing torch.
    pins = plan.get("source_pins")
    if not isinstance(pins, dict) or not pins:
        raise ValueError("Frozen source pins are missing")
    for path, digest in pins.items():
        if sha(absolute(path)) != digest:
            raise ValueError("Frozen scientific source SHA mismatch")
    return plan, spec, policy


def activate(spec_path):
    plan, spec, policy = validate(spec_path)
    mt_case.PHYSICAL_GPUS = AUTHORIZED
    mt_supervise.PHYSICAL_GPUS = AUTHORIZED
    receipt = {"schema": "v5-recovery-resource-override.v1", "case_id": TARGET,
        "original_gpu_devices": policy["original_gpu_devices"],
        "authorized_gpu_devices": list(AUTHORIZED), "maximum_distinct_owned_gpus": 4,
        "execution_entrypoint_changed": True, "scientific_source_files_modified": False,
        "recovery_policy_path": spec["recovery_policy_path"],
        "recovery_policy_sha256": spec["recovery_policy_sha256"],
        "recovery_adapter_sha256": spec["recovery_adapter_sha256"],
        "original_plan_sha256": spec["plan_sha256"], "selected_gpu": spec["gpu"]}
    path = Path(spec_path).parent / "recovery_resource_override.json"
    if path.exists():
        if mt_supervise.read_json(path) != receipt:
            raise ValueError("Existing recovery resource receipt differs")
    else:
        mt_supervise.write_json(path, receipt)
    return plan, spec


def run_training_child(spec_path, phase, output, receipt_path):
    plan, spec, _ = validate(spec_path)
    if phase not in ("train-canary", "formal"):
        raise ValueError("Unexpected recovery training phase")
    argv = [plan["paths"]["python"], str(ADAPTER), "--mode", "train-child", "--spec", str(spec_path),
            "--phase", phase, "--output", str(output), "--receipt", str(receipt_path)]
    started = time.time()
    # No new session/group: the original independent guard owns both fresh fits.
    process = subprocess.Popen(argv, cwd=plan["paths"]["scope_root"])
    identity = mt_supervise.process_identity(process.pid) if os.name == "posix" else {"pid": process.pid}
    code = process.wait()
    result = {"phase": phase, "pid": process.pid, "identity": identity, "command": argv,
              "exit_code": code, "started_unix": started, "finished_unix": time.time(),
              "fresh_process": True, "inherits_owned_process_group": True,
              "recovery_policy_sha256": spec["recovery_policy_sha256"],
              "recovery_adapter_sha256": spec["recovery_adapter_sha256"]}
    mt_supervise.write_json(Path(spec_path).parent / f"{phase}_process.json", result)
    if code:
        raise RuntimeError(f"{phase} child failed with exit code {code}; no implicit retry")
    return result


def execute(mode, spec_path, *, output=None, phase=None, receipt=None):
    spec_path = absolute(spec_path)
    plan, spec = activate(spec_path)
    root = spec_path.parent
    if mode == "supervise":
        if output is None or Path(output).resolve() != root / "supervision" or phase or receipt:
            raise ValueError("Supervision output must be the exact attempt supervision directory")
        return mt_supervise.supervise(spec_path, output)
    if mode == "worker":
        if output or phase or receipt:
            raise ValueError("Worker uses only the bound spec")
        checked_plan, checked_spec = mt_case.guarded_plan(spec_path)
        mt_case.preflight_then_formal(spec_path, checked_plan, checked_spec, run_child=run_training_child)
        return 0
    if mode != "train-child" or phase not in ("train-canary", "formal"):
        raise ValueError("Invalid recovery mode or phase")
    expected_output = root / ("qualification" if phase == "train-canary" else "artifacts")
    expected_receipt = root / ("qualification_device_budget.json" if phase == "train-canary" else "device_budget.json")
    if (output is None or receipt is None or Path(output).resolve() != expected_output
            or Path(receipt).resolve() != expected_receipt):
        raise ValueError("Recovery phase output or allocator receipt binding mismatch")
    checked_plan, checked_spec = mt_case.guarded_plan(spec_path, descendant=True)
    mt_case.train_process(spec_path, checked_plan, checked_spec, output, phase, receipt)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("supervise", "worker", "train-child"))
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output")
    parser.add_argument("--phase", choices=("train-canary", "formal"))
    parser.add_argument("--receipt")
    args = parser.parse_args(argv)
    return execute(args.mode, args.spec, output=args.output, phase=args.phase, receipt=args.receipt)


if __name__ == "__main__":
    raise SystemExit(main())

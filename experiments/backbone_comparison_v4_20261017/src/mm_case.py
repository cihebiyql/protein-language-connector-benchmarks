"""One fixed fit -> selected-checkpoint evaluation -> model-free acceptance case."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def watchdog_binding(contract_path):
    """Bind this exact independent process group to its fresh frozen contract."""
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Independent case watchdog requires the POSIX main thread")
    path = Path(contract_path).resolve(strict=True)
    contract = read_json(path)
    pid, pgid = os.getpid(), os.getpgrp()
    parent_pgid = os.getpgid(os.getppid())
    if pid <= 1 or pgid != pid or os.getsid(0) != pid or parent_pgid == pgid:
        raise RuntimeError("Watchdog refuses a shared or foreign process group")
    expected = contract["execution"]["command"]
    actual = [x.decode() for x in Path("/proc/self/cmdline").read_bytes().split(b"\0") if x]
    if (not actual or len(actual) != len(expected) or actual[1:] != expected[1:]
            or Path(actual[0]).resolve() != Path(expected[0]).resolve()
            or Path.cwd().resolve() != Path(contract["execution"]["working_directory"]).resolve()
            or contract.get("is_template") is not False or contract.get("contract_status") != "frozen"):
        raise RuntimeError("Watchdog process argv/cwd is not the frozen contract command")
    contract_sha = sha(path)
    started_path = path.parent.parent / "supervision/started.json"
    # Popen can run the child a few milliseconds before the parent writes its receipt.
    wait_until = time.monotonic() + .5
    while not started_path.exists() and time.monotonic() < wait_until:
        time.sleep(.01)
    started = read_json(started_path)
    if (started.get("pid") != pid or started.get("pgid") != pgid
            or started.get("contract_sha256") != contract_sha):
        raise RuntimeError("Watchdog contract SHA/PID differs from actual-start receipt")
    timeout = contract["execution"]["timeout_seconds"]
    frozen = datetime.fromisoformat(contract["frozen_at"].replace("Z", "+00:00"))
    deadline = datetime.fromisoformat(contract["timing"]["effective_deadline_at"].replace("Z", "+00:00"))
    budget_clock = time.monotonic()
    now = datetime.now(timezone.utc)
    if (type(timeout) not in (int, float) or isinstance(timeout, bool) or timeout <= 0
            or not math.isfinite(timeout) or frozen.tzinfo is None or deadline.tzinfo is None or frozen > now):
        raise RuntimeError("Invalid watchdog timing contract")
    # Freeze precedes supervisor begin/Popen. Charging its intervening preparation
    # is conservative and avoids silently granting a new full timeout in the child.
    remaining = min(timeout - (now - frozen).total_seconds(), (deadline - now).total_seconds())
    if remaining <= 0:
        raise RuntimeError("Case watchdog budget already exhausted")
    return {"pid": pid, "pgid": pgid, "parent_pgid_at_install": parent_pgid,
            "contract_path": str(path), "contract_sha256": contract_sha,
            "actual_started_receipt_sha256": sha(started_path), "contract_timeout_seconds": timeout,
            "frozen_at": frozen.isoformat(), "absolute_deadline": deadline.isoformat(),
            "monotonic_at_budget_calculation": budget_clock,
            "timer_seconds": remaining, "installed_at": now.isoformat(),
            "timeout_anchor": "conservative_frozen_at", "termination_signal": "SIGKILL",
            "scope": "this exact standalone case process group and inherited fit/evaluate children"}


def expire_owned_group(binding, killpg=None):
    """Alarm handler may kill only this pre-bound independent leader and its group."""
    pid, pgid = binding["pid"], binding["pgid"]
    if (pid != os.getpid() or pgid != pid or os.getpgrp() != pgid or os.getsid(0) != pid
            or binding["parent_pgid_at_install"] == pgid):
        raise RuntimeError("Watchdog refuses to signal any foreign process group")
    (killpg or os.killpg)(pgid, signal.SIGKILL)


@contextmanager
def case_watchdog(contract_path):
    binding = watchdog_binding(contract_path)
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    begin = time.monotonic()
    remaining = min(binding["timer_seconds"] - (begin - binding["monotonic_at_budget_calculation"]),
                    (datetime.fromisoformat(binding["absolute_deadline"]) - datetime.now(timezone.utc)).total_seconds()) - .01
    if remaining <= 0:
        raise RuntimeError("Watchdog budget exhausted before timer installation")
    # Preserve an already earlier bound rather than extending it.
    seconds = min(remaining, previous_timer[0]) if previous_timer[0] > 0 else remaining
    binding["timer_seconds"] = seconds
    def alarm(_number, _frame):
        expire_owned_group(binding)
    signal.signal(signal.SIGALRM, alarm)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield binding
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            remaining = max(previous_timer[0] - (time.monotonic() - begin), 1e-6)
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


def verify_pins(plan):
    for category in ("source_pins", "asset_pins"):
        for path, expected in plan[category].items():
            if sha(path) != expected:
                raise ValueError(f"Fixed {category} mismatch: {path}")


def training_commands(plan, scenario, variant, seed, output, *, backbone):
    paths = plan["paths"]
    base = ["--config", paths["config"],
            "--project-root", paths["project_root"], "--cache-root", paths["cache_root"],
            "--view-root", paths["view_root"], "--scenario", scenario, "--variant", variant, "--backbone", backbone,
            "--seed", str(seed), "--device", "cuda:0"]
    root = Path(output).resolve()
    def wrapped(phase, arguments):
        return [paths["python"], "-m", "src.mm_worker_limit", "--gpu-memory-budget-bytes",
                str(plan["gpu_memory_budget_bytes"]), "--receipt", str(root / f"{phase}_gpu_limit.json"),
                "--module", "src.mm_train", "--"] + base + arguments
    return [wrapped("fit", ["--phase", "fit", "--output", str(root / "fit")]),
            wrapped("evaluate", ["--phase", "evaluate", "--checkpoint", str(root / "fit/selected.pt"),
                                  "--output", str(root / "evaluate")])]


def run_case(plan, scenario, variant, seed, output, *, execute=subprocess.run, evaluate=None, physical_gpu=None,
             watchdog=None, backbone=None):
    """Child processes inherit this process group; only mm_supervise owns a session."""
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    try:
        if watchdog is not None:
            write_new(root / "independent_watchdog.json", watchdog)
        verify_pins(plan)
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in {str(g) for g in plan.get("candidate_gpu_devices", [x["gpu"] for x in plan["ready_assignments"]])}:
            raise ValueError("Case must inherit an explicitly assigned physical GPU from supervisor")
        if physical_gpu is not None and os.environ["CUDA_VISIBLE_DEVICES"] != str(physical_gpu):
            raise ValueError("Visible GPU differs from this case's exact contracted assignment")
        if backbone not in plan.get("backbones", ("qwen25_1p5b", "qwen3_4b_2507")):
            raise ValueError("Explicit registered backbone required")
        commands = training_commands(plan, scenario, variant, seed, root, backbone=backbone)
        write_new(root / "case_request.json", {
            "scenario": scenario, "variant": variant, "seed": seed, "backbone": backbone,
            "started_at": started, "pid": os.getpid(), "pgid": os.getpgrp(),
            "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"], "device": "cuda:0",
            "commands": commands, "new_session": False})
        for command in commands:
            verify_pins(plan)
            # No shell, setsid, preexec_fn or start_new_session: descendants stay bounded.
            completed = execute(command, cwd=plan["paths"]["scope_root"], check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"Training phase {command[command.index('--phase') + 1]} exited {completed.returncode}")
            cap = read_json(command[command.index("--receipt") + 1])
            budget = plan["gpu_memory_budget_bytes"]
            if (cap.get("allocator_budget_bytes") != budget
                    or cap.get("visible_devices") != os.environ["CUDA_VISIBLE_DEVICES"]
                    or cap.get("module") != "src.mm_train"
                    or any(type(cap.get(key)) is not int or not 0 <= cap[key] <= budget
                           for key in ("peak_allocated_bytes", "peak_reserved_bytes"))):
                raise ValueError("Actual allocator receipt differs from this case budget/assignment")
        verify_pins(plan)
        if evaluate is None:
            from .mm_evaluate import evaluate_run
            evaluate = evaluate_run
        result = evaluate(root / "evaluate/run_manifest.json", plan["paths"]["evaluation_protocol"],
                          plan["paths"]["data_root"])
        if (result.get("status") != "complete" or result.get("formal_eligible") is not True
                or (result.get("scenario"), result.get("arm"), result.get("seed"), result.get("backbone")) != (scenario, variant, seed, backbone)):
            raise ValueError("Model-free evaluator did not accept this exact formal case")
        write_new(root / "result.json", result)
        write_new(root / "case_receipt.json", {
            "status": "complete", "scientific_acceptance": "controller_pending",
            "result_sha256": sha(root / "result.json"), "started_at": started,
            "gpu_limit_receipts": {phase: sha(root / f"{phase}_gpu_limit.json") for phase in ("fit", "evaluate")},
            "ended_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "pgid": os.getpgrp()})
        return 0
    except BaseException as exc:
        write_new(root / "case_failed.json", {"status": "failed", "scientific_acceptance": False,
                  "started_at": started, "ended_at": datetime.now(timezone.utc).isoformat(),
                  "exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    with case_watchdog(args.contract) as watchdog:
        if sha(args.plan) != args.plan_sha256:
            raise ValueError("Frozen plan changed")
        plan = read_json(args.plan)
        # Same matrix and readiness contract as the parent queue, without launching it.
        from .mm_queue import fixed_cases, validate_plan
        validate_plan(plan)
        if (args.scenario, args.variant, args.seed, args.backbone) not in {
                (x["scenario"], x["variant"], x["seed"], x["backbone"]) for x in fixed_cases()}:
            raise ValueError("Case is outside the fixed matrix")
        return run_case(plan, args.scenario, args.variant, args.seed, args.output,
                        physical_gpu=args.physical_gpu, watchdog=watchdog, backbone=args.backbone)


if __name__ == "__main__":
    raise SystemExit(main())

"""Finite V5 job queue: three existing physical GPUs and one CPU baseline lane."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .mt_case import accepted_attempt, sha, verify_pins
from .mt_supervise import GIB, PHYSICAL_GPUS, probe, process_identity, same_process, read_json, write_json

TERMINAL = {"accepted", "failed", "not_started_resource_timeout"}


def validate_plan(plan):
    if set(plan["gpu_devices"]) != set(PHYSICAL_GPUS) or len(plan["gpu_devices"]) != 3:
        raise ValueError("Use exactly existing physical GPUs 3,4,5; never a fourth card")
    expected = {(task, method, backbone, seed)
        for task in ("meltome", "elm", "rt") for seed in (17, 42, 2026)
        for method, backbone in (("ESM", "none"), ("M0", "qwen25_1p5b"), ("M3", "qwen25_1p5b"),
                                 ("M0", "qwen3_4b_2507"), ("M3", "qwen3_4b_2507"))}
    actual = [(c["task"], c["method"], c["backbone"], c["seed"]) for c in plan["cases"]]
    if len(actual) != 45 or set(actual) != expected or len({c["case_id"] for c in plan["cases"]}) != 45:
        raise ValueError("Frozen plan must contain exactly the 45 fresh case identities")
    for case in plan["cases"]:
        if Path(case["case_id"]).name != case["case_id"] or case["case_id"] in (".", ".."):
            raise ValueError("Unsafe case ID")
    for key in ("python", "scope_root", "data_root", "cache_root", "config"):
        if not Path(plan["paths"][key]).is_absolute():
            raise ValueError(f"Absolute runtime path required: {key}")
    if not 0 < plan["per_case_timeout_seconds"] <= 86400:
        raise ValueError("Explicit finite per-case timeout required")
    if not 0 < plan.get("resource_wait_timeout_seconds", 86400) <= 86400:
        raise ValueError("Resource wait must be positive and at most one day")
    policy = plan["resource_policy"]
    if policy.get("cpu_threads", 2) != 2:
        raise ValueError("Two CPU threads per worker required")
    if (not 0 < policy.get("rss_limit_bytes", 64 * GIB) <= 64 * GIB
            or not 0 < policy.get("cpu_rss_limit_bytes", 16 * GIB) <= 16 * GIB):
        raise ValueError("RSS budget exceeds frozen user policy")
    for backbone in ("qwen25_1p5b", "qwen3_4b_2507"):
        cap = policy["gpu_memory_budget_bytes"][backbone]
        if type(cap) is not int or not 0 < cap <= 24 * GIB:
            raise ValueError("Invalid explicitly frozen allocator cap")
    config = plan["paths"]["config"]
    if config not in plan["asset_pins"] and config not in plan["source_pins"]:
        raise ValueError("Training configuration must be hashed in the frozen plan")
    if plan.get("status") != "frozen":
        raise ValueError("Queue requires a frozen plan")
    verify_pins(plan)


@contextmanager
def lease(path):
    import fcntl
    with Path(path).open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another V5 owner holds {Path(path).name}") from exc
        yield stream


def append_event(root, event):
    event = {"at_unix": time.time(), **event}
    with (Path(root) / "queue_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return event


def events(root):
    path = Path(root) / "queue_events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_spec(plan_path, plan, case, attempt, gpu):
    policy = plan["resource_policy"]
    cap = 0 if gpu is None else policy["gpu_memory_budget_bytes"][case["backbone"]]
    rss = policy.get("cpu_rss_limit_bytes", 16 * GIB) if gpu is None else policy.get("rss_limit_bytes", 64 * GIB)
    return {"schema": "mt-case-spec.v1", "plan_path": str(Path(plan_path).resolve()),
        "plan_sha256": sha(plan_path), "case": case, "gpu": gpu,
        "gpu_memory_budget_bytes": cap, "rss_limit_bytes": rss,
        "expected_host": policy.get("expected_host"),
        "storage_reserve_bytes": policy.get("storage_reserve_bytes", 10 * GIB),
        "timeout_seconds": plan["per_case_timeout_seconds"], "cwd": plan["paths"]["scope_root"],
        "command": [plan["paths"]["python"], "-m", "src.mt_case", "--spec", str(attempt / "spec.json")]}


def run(plan_path, output, *, resume=False, poll_seconds=10):
    if os.name != "posix":
        raise RuntimeError("V5 training queue is Linux-only")
    plan_path = Path(plan_path).resolve(strict=True)
    plan = read_json(plan_path)
    validate_plan(plan)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=resume)
    # Scope-wide leases also reject a second controller with a different output
    # directory; output-local locks alone would permit duplicate GPU workers.
    lock_root = Path(plan["paths"]["scope_root"]) / ".runtime_locks"
    lock_root.mkdir(exist_ok=True)
    with lease(lock_root / "controller.lock"):
        if resume:
            binding = read_json(root / "queue_binding.json")
            if binding["plan_sha256"] != sha(plan_path):
                raise ValueError("Resume refuses a changed plan")
        else:
            write_json(root / "queue_binding.json", {"plan_path": str(plan_path), "plan_sha256": sha(plan_path),
                "started_unix": time.time(), "case_count": 45})
            for case in plan["cases"]:
                append_event(root, {"case_id": case["case_id"], "status": "pending"})
        write_json(root / "controller.json", {"identity": process_identity(os.getpid()), "resumed": resume,
            "plan_sha256": sha(plan_path), "at_unix": time.time()})
        return _loop(plan_path, plan, root, poll_seconds, lock_root)


def _loop(plan_path, plan, root, poll_seconds, lock_root):
    import fcntl
    history = events(root)
    latest = {e["case_id"]: e for e in history if "case_id" in e}
    cases = {c["case_id"]: c for c in plan["cases"]}
    if set(latest) != set(cases):
        raise ValueError("Ledger case identities differ from the frozen matrix")
    active = {}
    lane_locks = {}
    handles = {}
    first_wait = {cid: min(e["at_unix"] for e in history if e.get("case_id") == cid and e["status"] == "resource_wait")
        for cid in cases if any(e.get("case_id") == cid and e["status"] == "resource_wait" for e in history)}

    def lock_lane(lane):
        stream = (lock_root / f"lane_{lane}.lock").open("a+")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            return None
        return stream

    for cid, state in latest.items():
        if state["status"] in ("starting", "running"):
            lane = state["lane"]
            if lane in active:
                raise ValueError("Ledger contains overlapping V5 workers on one lane")
            stream = lock_lane(lane)
            if stream is None:
                raise RuntimeError("An active V5 lane is owned by another controller")
            active[lane], lane_locks[lane] = cid, stream

    try:
        while any(e["status"] not in TERMINAL for e in latest.values()):
            if sha(plan_path) != read_json(root / "queue_binding.json")["plan_sha256"]:
                raise ValueError("Frozen plan changed during execution")
            for lane, cid in list(active.items()):
                state = latest[cid]
                attempt = Path(state["attempt"])
                finished = attempt / "supervision/finished.json"
                if not finished.exists():
                    supervisor_identity = state.get("supervisor_identity")
                    owner_receipt = attempt / "supervisor_owner.json"
                    if supervisor_identity is None and owner_receipt.exists():
                        supervisor_identity = read_json(owner_receipt)["identity"]
                    supervisor_alive = bool(supervisor_identity and same_process(supervisor_identity))
                    started = attempt / "supervision/started.json"
                    worker_alive = started.exists() and same_process(read_json(started)["identity"])
                    # A new supervisor may need a moment to create the child receipt.
                    grace = plan["per_case_timeout_seconds"] + 35 if state["status"] == "starting" else 15
                    if supervisor_alive or worker_alive or time.time() - state["at_unix"] < grace:
                        continue
                    outcome = {"status": "failed", "error": "supervisor_lost_without_terminal_receipt"}
                else:
                    receipt = read_json(finished)
                    if receipt["status"] == "not_started_resource_wait":
                        outcome = {"status": "resource_wait", "reasons": receipt.get("reasons", [])}
                    else:
                        try:
                            binding = accepted_attempt(plan, attempt, cases[cid])
                            outcome = {"status": "accepted", "binding": binding}
                        except Exception as exc:
                            outcome = {"status": "failed", "error": repr(exc)}
                latest[cid] = append_event(root, {"case_id": cid, "attempt": str(attempt), **outcome})
                if outcome["status"] == "resource_wait":
                    first_wait.setdefault(cid, time.time())
                if lane in handles:
                    handles.pop(lane).wait(timeout=10)
                lane_locks.pop(lane).close()
                del active[lane]
            for lane in ["cpu", *plan["gpu_devices"]]:
                if lane in active:
                    continue
                waiting = [c for c in plan["cases"] if latest[c["case_id"]]["status"] in ("pending", "resource_wait")
                    and (c["method"] == "ESM") == (lane == "cpu")]
                # Execute lower-memory work while larger jobs await actual VRAM.
                if lane != "cpu":
                    waiting.sort(key=lambda c: plan["resource_policy"]["gpu_memory_budget_bytes"][c["backbone"]])
                candidate = next(iter(waiting), None)
                if candidate is None:
                    continue
                cid = candidate["case_id"]
                if cid in first_wait and time.time() - first_wait[cid] >= plan.get("resource_wait_timeout_seconds", 86400):
                    latest[cid] = append_event(root, {"case_id": cid, "status": "not_started_resource_timeout"})
                    continue
                gpu = None if lane == "cpu" else lane
                case_root = root / "cases" / cid
                case_root.mkdir(parents=True, exist_ok=True)
                number = 1 + len(list(case_root.glob("attempt_*")))
                attempt = case_root / f"attempt_{number:03d}"
                spec = build_spec(plan_path, plan, candidate, attempt, gpu)
                observation = probe(root=root, gpu=gpu, rss_limit_bytes=spec["rss_limit_bytes"],
                    gpu_memory_budget_bytes=spec["gpu_memory_budget_bytes"], expected_host=spec["expected_host"],
                    storage_reserve_bytes=spec["storage_reserve_bytes"])
                if not observation["ready"]:
                    first_wait.setdefault(cid, time.time())
                    previous = latest[cid]
                    if previous["status"] != "resource_wait" or previous.get("reasons") != observation["reasons"]:
                        latest[cid] = append_event(root, {"case_id": cid, "status": "resource_wait",
                            "reasons": observation["reasons"], "last_resource_observation": observation})
                    continue
                stream = lock_lane(lane)
                if stream is None:
                    continue
                try:
                    verify_pins(plan)
                    attempt.mkdir(exist_ok=False)
                    write_json(attempt / "spec.json", spec)
                    # Reservation precedes Popen, so --resume never starts a
                    # second worker if interruption hits the launch window.
                    latest[cid] = append_event(root, {"case_id": cid, "status": "starting", "lane": lane,
                        "attempt": str(attempt)})
                    with (attempt / "controller_stdout.log").open("wb") as stdout, (attempt / "controller_stderr.log").open("wb") as stderr:
                        process = subprocess.Popen([plan["paths"]["python"], "-m", "src.mt_supervise",
                            "--spec", str(attempt / "spec.json"), "--output", str(attempt / "supervision")],
                            cwd=plan["paths"]["scope_root"], stdout=stdout, stderr=stderr,
                            stdin=subprocess.DEVNULL, start_new_session=True)
                    identity = process_identity(process.pid)
                    if not identity:
                        raise RuntimeError("Supervisor identity unavailable")
                    write_json(attempt / "supervisor_owner.json", {"identity": identity})
                    latest[cid] = append_event(root, {"case_id": cid, "status": "running", "lane": lane,
                        "attempt": str(attempt), "supervisor_identity": identity})
                    handles[lane], active[lane], lane_locks[lane] = process, cid, stream
                except BaseException:
                    stream.close()
                    raise
            write_json(root / "queue_state.json", {"schema": "mt-queue-state.v1", "plan_sha256": sha(plan_path),
                "at_unix": time.time(), "case_count": 45,
                "counts": {status: sum(v["status"] == status for v in latest.values()) for status in sorted({v["status"] for v in latest.values()})},
                "cases": latest})
            if any(e["status"] not in TERMINAL for e in latest.values()):
                time.sleep(poll_seconds)
        summary = {"schema": "mt-queue-complete.v1", "plan_sha256": sha(plan_path), "finished_unix": time.time(),
            "case_count": 45, "accepted": sum(v["status"] == "accepted" for v in latest.values()), "cases": latest}
        write_json(root / "complete.json", summary)
        if plan.get("analysis_command"):
            # The parent freezes this exact CLI. Never execute shell text.
            argv = plan["analysis_command"]
            if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
                raise ValueError("Analysis command must be an explicit argv list")
            with (root / "analysis_stdout.log").open("wb") as stdout, (root / "analysis_stderr.log").open("wb") as stderr:
                result = subprocess.run(argv, cwd=plan["paths"]["scope_root"], stdout=stdout, stderr=stderr, timeout=3600)
            write_json(root / "analysis_exit.json", {"exit_code": result.returncode})
        return 0 if summary["accepted"] == 45 else 1
    finally:
        # Workers and their independent guards survive a controller interruption.
        for stream in lane_locks.values():
            stream.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args.plan, args.output, resume=args.resume))


if __name__ == "__main__":
    main()

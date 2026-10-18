"""Finite node1-owned formal queue, integrity acceptance, analysis, publication.

Prerequisite feature extraction, TRAIN-only specialists, technical smokes, and
freezing are performed before this entrypoint. No timer or client callback is
installed. Existing fits and failures are never deleted or silently retried.
"""
import argparse
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


ARMS = ("TEXT", "M0", "M3")
SEEDS = (17, 42, 2026)
CASE_TIMEOUT = 18 * 3600
TOTAL_TIMEOUT = 72 * 3600
RESOURCE_TIMEOUT = 6 * 3600
PROJECT_MARKERS = ("multitask_connector_v5_20261017",
                   "backbone_comparison_v4_20261017/project",
                   "molecular_memory_v3_20261017/project",
                   "peptide_question_v6_20261018")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def checked_path(scope, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Artifact path must stay within its scope")
    path = (Path(scope) / relative).resolve()
    path.relative_to(Path(scope).resolve())
    return path


def verify_freeze(scope):
    scope = Path(scope)
    manifest_path = scope / "frozen_manifest.json"
    manifest = read_json(manifest_path)
    entries = manifest.get("files")
    if isinstance(entries, dict):
        entries = [{"path": name, "sha256": value if isinstance(value, str)
                    else value["sha256"]} for name, value in entries.items()]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Frozen manifest must enumerate nonempty files")
    seen = set()
    for item in entries:
        name = item["path"]
        if name in seen or sha256(checked_path(scope, name)) != item["sha256"]:
            raise ValueError(f"Frozen artifact changed or duplicated: {name}")
        seen.add(name)
    if "config.json" not in seen or "src/qa_train.py" not in seen:
        raise ValueError("Frozen manifest must bind config.json and src/qa_train.py")
    return sha256(manifest_path)


def validate_config(config):
    if tuple(config.get("arms", ())) != ARMS or tuple(config.get("seeds", ())) != SEEDS:
        raise ValueError("Formal queue requires all three fixed arms and seeds")
    devices = config.get("gpu_devices")
    if (not isinstance(devices, list) or not 1 <= len(devices) <= 4
            or len(set(devices)) != len(devices)
            or any(not isinstance(device, int) or device < 0 for device in devices)):
        raise ValueError("gpu_devices must contain one to four unique physical GPU indices")
    minimum = config.get("min_free_memory_mib")
    if not isinstance(minimum, (int, float)) or minimum <= 0:
        raise ValueError("Positive smoke-measured min_free_memory_mib required")
    if socket.gethostname().split(".")[0] != "node1":
        raise ValueError("Formal queue may run only on node1")


def process_identity(pid, proc_root=Path("/proc")):
    """Linux PID + start ticks identifies a process despite PID reuse."""
    try:
        root = Path(proc_root) / str(int(pid))
        stat = (root / "stat").read_text()
        fields = stat.rsplit(")", 1)[1].split()
        return {"pid": int(pid), "start_ticks": int(fields[19]),
                "state": fields[0], "session": int(fields[3])}
    except (OSError, ValueError, IndexError):
        return None


def same_live_process(identity):
    actual = process_identity(identity["pid"])
    return (actual is not None and actual["state"] != "Z"
            and actual["start_ticks"] == identity["start_ticks"]
            and actual["session"] == identity["session"])


@contextmanager
def exclusive_lock(path):
    import fcntl  # node1 POSIX only; analysis helpers remain importable on Windows.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another pipeline owns {path.name}") from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def run_capture(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=30,
                            check=True, stdin=subprocess.DEVNULL)
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def gpu_snapshot():
    rows = run_capture(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
                        "--format=csv,noheader,nounits"])
    devices = [{"index": int(row[0]), "uuid": row[1], "name": row[2],
                "total_mib": int(row[3]), "used_mib": int(row[4]),
                "free_mib": int(row[5]), "utilization_percent": int(row[6])}
               for row in rows if row]
    apps = run_capture(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name,used_memory",
                        "--format=csv,noheader,nounits"])
    uuid_to_index = {item["uuid"]: item["index"] for item in devices}
    processes = []
    project_devices = set()
    for row in apps:
        if not row or not row[0].strip().isdigit():
            continue
        pid, uuid = int(row[0]), row[1]
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
        except OSError:
            command = ""
        owned_project = any(marker in command for marker in PROJECT_MARKERS)
        index = uuid_to_index.get(uuid)
        if owned_project and index is not None:
            project_devices.add(index)
        processes.append({"pid": pid, "gpu_index": index, "process_name": row[2],
                          "used_memory_mib": row[3], "project_process": owned_project})
    return {"at": utc_now(), "devices": devices, "processes": processes,
            "project_gpu_devices": sorted(project_devices)}


def select_devices(snapshot, permitted, occupied, minimum, cap=4):
    """Use free memory as feasibility only; never impose utilization limits."""
    observed = set(snapshot["project_gpu_devices"]) | set(occupied)
    selected = []
    by_index = {item["index"]: item for item in snapshot["devices"]}
    for index in permitted:
        if index in occupied or index not in by_index:
            continue
        if by_index[index]["free_mib"] < minimum or len(observed | {index}) > cap:
            continue
        selected.append(index)
        observed.add(index)
    return selected


def verify_result(scope, case, config):
    directory = Path(scope) / "runs" / case["id"] / "formal"
    if (directory / "failure.json").exists():
        raise ValueError("Trainer failure record exists")
    receipt = read_json(directory / "completion.json")
    for name, expected in (("status", "complete"), ("arm", case["arm"]),
                           ("seed", case["seed"]), ("phase", "formal")):
        if receipt.get(name) != expected:
            raise ValueError(f"Completion identity differs: {name}")
    epochs = int(config.get("training", {}).get("epochs", 6))
    if receipt.get("epochs_completed") != epochs or receipt.get("optimizer_updates", 0) <= 0:
        raise ValueError("Formal epoch/update budget incomplete")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Completion must bind its artifacts by SHA256")
    conditions = ("normal", "shuffled", "removed") if case["arm"] == "TEXT" else (
                  "normal", "mean", "shuffled", "off")
    data_root = checked_path(scope, config.get("data_root", "data"))
    panels = {split: [json.loads(line) for line in (data_root / f"{split}.jsonl").read_text(
                         encoding="utf-8").splitlines() if line.strip()]
              for split in ("train", "validation", "test")}
    expected_keys = {split: {(row["id"], question) for row in rows
                             for question in ("rt", "ccs", "both")}
                     for split, rows in panels.items()}
    accumulation = int(config.get("training", {}).get("gradient_accumulation_steps", 8))
    expected_updates = epochs * math.ceil(len(expected_keys["train"]) / accumulation)
    if (receipt["optimizer_updates"] != expected_updates
            or receipt.get("training_examples_seen") != epochs * len(expected_keys["train"])):
        raise ValueError("Formal optimizer/example counts differ from the complete frozen TRAIN budget")
    frozen = receipt.get("frozen", {})
    if (frozen.get("manifest_sha256") != sha256(Path(scope) / "frozen_manifest.json")
            or frozen.get("config_sha256") != sha256(Path(scope) / "config.json")):
        raise ValueError("Completion frozen inputs differ from the current frozen scope")
    required = {"selected_checkpoint.json", "best_adapter.pt", "training_history.json", "test_summary.json", "run_contract.json"}
    required.update(f"test_{condition}.jsonl" for condition in conditions)
    required.update(f"validation_epoch_{epoch:02d}.jsonl" for epoch in range(1, epochs + 1))
    if not required.issubset(artifacts):
        raise ValueError("Completion lacks checkpoint/history/prediction artifact bindings")
    hashes = {}
    for relative, item in artifacts.items():
        expected_sha = item if isinstance(item, str) else item["sha256"]
        path = checked_path(directory, relative)
        actual_sha = sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(f"Result SHA256 mismatch: {relative}")
        hashes[relative] = actual_sha
        if relative.endswith(".jsonl"):
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not rows or (isinstance(item, dict) and "rows" in item and len(rows) != item["rows"]):
                raise ValueError(f"Prediction row count mismatch: {relative}")
            keys = [(row["id"], row["question"]) for row in rows]
            if len(keys) != len(set(keys)) or len(rows) % 3:
                raise ValueError("Predictions require three unique question rows per peptide")
            groups = {}
            split = "test" if relative.startswith("test_") else "validation"
            if set(keys) != expected_keys[split]:
                raise ValueError("Prediction identities/questions differ from the full frozen panel")
            for row in rows:
                groups.setdefault(row["id"], set()).add(row["question"])
                expected_condition = (relative.removeprefix("test_").removesuffix(".jsonl")
                                      if split == "test" else "normal")
                if row.get("condition") != expected_condition:
                    raise ValueError("Prediction condition differs from artifact name")
                if not isinstance(row.get("raw"), str):
                    raise ValueError("Prediction raw answer missing")
            if any(len(questions) != 3 for questions in groups.values()):
                raise ValueError("Each test peptide must retain all three questions")
    selected = read_json(directory / "selected_checkpoint.json")
    if (selected.get("test_opened") is not False
            or not 1 <= selected.get("selected_epoch", 0) <= epochs
            or selected.get("epochs_completed") != epochs
            or selected.get("checkpoint_sha256") != hashes["best_adapter.pt"]
            or selected.get("frozen") != frozen
            or selected.get("optimizer_updates") != expected_updates
            or selected.get("training_examples_seen") != receipt["training_examples_seen"]):
        raise ValueError("Checkpoint selection receipt inconsistent")
    contract = read_json(directory / "run_contract.json")
    if (contract.get("arm") != case["arm"] or contract.get("seed") != case["seed"]
            or contract.get("phase") != "formal" or contract.get("backbone_frozen") is not True
            or contract.get("frozen") != frozen):
        raise ValueError("Training contract identity/backbone/frozen bindings differ")
    history = read_json(directory / "training_history.json")
    if len(history) != epochs or [item.get("epoch_one_based") for item in history] != list(range(1, epochs + 1)):
        raise ValueError("Training history must preserve every fixed epoch")
    return {"verified_at": utc_now(), "completion_sha256": sha256(directory / "completion.json"),
            "artifact_sha256": hashes, "optimizer_updates": receipt["optimizer_updates"]}


def child_environment(gpu=None):
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu),
                        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                        "OPENBLAS_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
                        "PYTHONUNBUFFERED": "1"})
    return environment


def start_child(arguments, log, gpu=None):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab", buffering=0) as stream:
        process = subprocess.Popen(arguments, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, env=child_environment(gpu),
            start_new_session=True, close_fds=True)
    identity = process_identity(process.pid)
    if identity is None or identity["session"] != process.pid:
        raise RuntimeError("Could not establish detached owned child identity")
    return process, identity


def stop_owned(identity):
    if not same_live_process(identity):
        return
    if identity["session"] != identity["pid"] or os.getpgid(identity["pid"]) != identity["session"]:
        raise RuntimeError("Refusing to terminate a process outside its owned new session")
    os.killpg(identity["session"], signal.SIGTERM)
    deadline = time.monotonic() + 20
    while same_live_process(identity) and time.monotonic() < deadline:
        time.sleep(0.2)
    if same_live_process(identity):
        os.killpg(identity["session"], signal.SIGKILL)


def event(scope, value):
    with (Path(scope) / "queue_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": utc_now(), **value}, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def save_status(scope, status):
    status["updated_at"] = utc_now()
    status["counts"] = {name: sum(case["status"] == name for case in status["cases"])
                        for name in ("pending", "running", "accepted", "failed")}
    atomic_json(Path(scope) / "STATUS.json", status)


def run_postprocess(scope, status):
    for name, relative in (("analysis", "src/qa_analyze.py"), ("publication", "resources/publish_v6.py")):
        arguments = [sys.executable, str(Path(scope) / relative), "--scope", str(scope)]
        try:
            process, identity = start_child(arguments, Path(scope) / "logs" / f"{name}.log")
        except Exception as error:
            status[name] = {"status": "failed", "reason": str(error), "finished_at": utc_now()}
            event(scope, {"event": f"{name}_launch_failed", "reason": str(error)})
            save_status(scope, status)
            continue
        status[name] = {"status": "running", "process": identity, "started_at": utc_now()}
        save_status(scope, status)
        try:
            code = process.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            stop_owned(identity)
            code = -signal.SIGTERM
        status[name].update(status="complete" if code == 0 else "failed", exit_code=code,
                            finished_at=utc_now())
        save_status(scope, status)
        if code:
            event(scope, {"event": f"{name}_failed", "exit_code": code})
            # Publication still gets the preserved queue/failure state if analysis failed.


def run_pipeline(scope):
    scope = Path(scope).resolve()
    with exclusive_lock(scope / "private" / "pipeline.run.lock"):
        manifest_sha = verify_freeze(scope)
        config = read_json(scope / "config.json")
        validate_config(config)
        config_sha = sha256(scope / "config.json")
        if (scope / "STATUS.json").exists():
            status = read_json(scope / "STATUS.json")
            if status.get("config_sha256") != config_sha or status.get("frozen_manifest_sha256") != manifest_sha:
                raise ValueError("Existing queue is bound to different frozen inputs")
            if status.get("status") == "complete":
                for case in status["cases"]:
                    if case["status"] == "accepted":
                        verify_result(scope, case, config)
                return status
        else:
            status = {"status": "running", "host": socket.gethostname(), "started_at": utc_now(),
                      "config_sha256": config_sha, "frozen_manifest_sha256": manifest_sha,
                      "cases": [{"id": f"{arm}__seed{seed}", "arm": arm, "seed": seed,
                                 "status": "pending"} for seed in SEEDS for arm in ARMS]}
        processes = {}
        for case in status["cases"]:
            if case["status"] == "accepted":
                case["acceptance"] = verify_result(scope, case, config)
            elif case["status"] == "running":
                if same_live_process(case["process"]):
                    processes[case["id"]] = None  # Adopt only this exact owned identity.
                else:
                    try:
                        case["acceptance"] = verify_result(scope, case, config)
                        case["status"] = "accepted"
                    except Exception as error:
                        case.update(status="failed", reason=f"Previous child stopped: {error}")
        save_status(scope, status)
        started = time.monotonic()
        wait_seconds = 0.0
        previous = started
        while any(case["status"] in ("pending", "running") for case in status["cases"]):
            now = time.monotonic()
            if now - started > TOTAL_TIMEOUT:
                for case in status["cases"]:
                    if case["status"] == "running":
                        stop_owned(case["process"])
                    if case["status"] in ("pending", "running"):
                        case.update(status="failed", reason="Overall finite queue deadline exceeded")
                break
            for case in status["cases"]:
                if case["status"] != "running":
                    continue
                process = processes.get(case["id"])
                code = process.poll() if process is not None else None
                alive = same_live_process(case["process"])
                age = time.time() - case["started_unix"]
                if alive and age <= CASE_TIMEOUT:
                    continue
                if alive:
                    stop_owned(case["process"])
                    code = -signal.SIGTERM
                case["exit_code"] = code
                case["finished_at"] = utc_now()
                try:
                    if code not in (0, None):
                        raise ValueError(f"Trainer exit code {code}")
                    case["acceptance"] = verify_result(scope, case, config)
                    case["status"] = "accepted"
                except Exception as error:
                    case.update(status="failed", reason=str(error))
                event(scope, {"event": "case_finished", "case_id": case["id"],
                              "status": case["status"], "exit_code": code})
                processes.pop(case["id"], None)
                save_status(scope, status)
            pending = [case for case in status["cases"] if case["status"] == "pending"]
            occupied = {case["gpu"] for case in status["cases"] if case["status"] == "running"}
            if pending and len(occupied) < len(config["gpu_devices"]):
                try:
                    snapshot = gpu_snapshot()
                except Exception as error:
                    # A failed GPU inventory cannot authorize a launch. Preserve
                    # owned running children and charge the finite wait budget.
                    snapshot = {"at": utc_now(), "devices": [], "processes": [],
                                "project_gpu_devices": [], "inventory_error": str(error)}
                available = select_devices(snapshot, config["gpu_devices"], occupied,
                                           config["min_free_memory_mib"])
                launched = False
                for gpu, case in zip(available, pending):
                    directory = scope / "runs" / case["id"] / "formal"
                    if directory.exists():
                        case.update(status="failed", reason="Existing formal directory preserved; refusing repeat fit")
                        event(scope, {"event": "existing_fit_preserved", "case_id": case["id"]})
                        continue
                    verify_freeze(scope)
                    arguments = [sys.executable, str(scope / "src" / "qa_train.py"), "--scope", str(scope),
                                 "--arm", case["arm"], "--seed", str(case["seed"]), "--phase", "formal"]
                    try:
                        process, identity = start_child(arguments, scope / "logs" / f"{case['id']}.log", gpu)
                    except Exception as error:
                        case.update(status="failed", reason=f"Child launch failed: {error}")
                        event(scope, {"event": "case_launch_failed", "case_id": case["id"],
                                      "reason": str(error)})
                        save_status(scope, status)
                        continue
                    case.update(status="running", gpu=gpu, process=identity, started_at=utc_now(),
                                started_unix=time.time(), launch_gpu_snapshot=snapshot)
                    processes[case["id"]] = process
                    event(scope, {"event": "case_started", "case_id": case["id"], "gpu": gpu,
                                  "process": identity, "arguments": arguments})
                    launched = True
                    save_status(scope, status)
                if not available and not launched:
                    wait_seconds += now - previous
                    status["resource_wait_seconds"] = wait_seconds
                    status["last_resource_snapshot"] = snapshot
                    if wait_seconds > RESOURCE_TIMEOUT:
                        for case in pending:
                            case.update(status="failed", reason="Six-hour resource wait budget exhausted")
                    save_status(scope, status)
            previous = now
            time.sleep(5)
        status["status"] = "formal_terminal"
        save_status(scope, status)
        verify_freeze(scope)
        run_postprocess(scope, status)
        success = (all(case["status"] == "accepted" for case in status["cases"])
                   and status.get("analysis", {}).get("status") == "complete"
                   and status.get("publication", {}).get("status") == "complete")
        status.update(status="complete" if success else "terminal_with_failures", finished_at=utc_now(),
                      local_computer_callback_required=False, scheduled_monitor_created=False)
        save_status(scope, status)
        event(scope, {"event": "pipeline_terminal", "status": status["status"], "counts": status["counts"]})
        return status


def launch_pipeline(scope):
    scope = Path(scope).resolve()
    with exclusive_lock(scope / "private" / "pipeline.launch.lock"):
        config = read_json(scope / "config.json")
        validate_config(config)
        manifest_sha = verify_freeze(scope)
        receipt_path = scope / "private" / "pipeline_launch.json"
        if receipt_path.exists():
            previous = read_json(receipt_path)
            if same_live_process(previous["process"]):
                return {**previous, "already_running": True}
        arguments = [sys.executable, str(Path(__file__).resolve()), "--scope", str(scope), "--stage", "run"]
        process, identity = start_child(arguments, scope / "logs" / "pipeline.log")
        receipt = {"started_at": utc_now(), "host": socket.gethostname(), "process": identity,
                   "arguments": arguments, "frozen_manifest_sha256": manifest_sha,
                   "stdin": "DEVNULL", "new_session": True,
                   "local_computer_callback_required": False, "scheduled_monitor_created": False}
        atomic_json(receipt_path, receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", required=True)
    parser.add_argument("--stage", choices=("run", "launch"), required=True)
    args = parser.parse_args()
    result = launch_pipeline(args.scope) if args.stage == "launch" else run_pipeline(args.scope)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if args.stage == "launch" or result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

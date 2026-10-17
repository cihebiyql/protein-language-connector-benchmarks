"""Fixed, resumable multi-GPU queue; only canonical supervisors launch GPU jobs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time
import traceback

from .mm_case import read_json, sha, verify_pins, write_new

DEADLINE = "2026-10-18T03:26:57+08:00"
SCENARIOS = ("new_peptide_known_target", "new_peptide_new_target")
SEEDS = (17, 42, 2026)
VARIANTS = ("M0", "M3")
BACKBONES = ("qwen25_1p5b", "qwen3_4b_2507")
SOURCE_NAMES = ("mm_case.py", "mm_queue.py", "mm_contract.py", "mm_supervise.py", "mm_train.py",
                "mm_model.py", "mm_data.py", "mm_extract.py", "mm_evaluate.py", "mm_worker_limit.py", "mm_archive.py")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def fixed_cases():
    return [{"case_id": f"{backbone}__{scenario}__{variant}__seed{seed}",
             "scenario": scenario, "backbone": backbone, "variant": variant, "seed": seed}
            for seed in SEEDS for scenario in SCENARIOS for backbone in BACKBONES for variant in VARIANTS]


def validate_plan(plan):
    if plan.get("schema_version") != "mm-v4.queue.v1" or plan.get("status") != "FROZEN":
        raise ValueError("An explicitly FROZEN mm-v4.queue.v1 plan is required")
    if plan.get("deadline") != DEADLINE:
        raise ValueError("The queue cannot reset the authorized deadline")
    paths = plan["paths"]
    required_paths = ("python", "project_root", "scope_root", "config", "cache_root", "view_root",
                      "evaluation_protocol", "data_root", "standard_root")
    if set(paths) != set(required_paths) or any(not Path(paths[k]).is_absolute() for k in required_paths):
        raise ValueError("Queue paths must be exactly the registered absolute paths")
    scope, project = Path(paths["scope_root"]).resolve(), Path(paths["project_root"]).resolve()
    if scope == project or project not in scope.parents:
        raise ValueError("New scope must be inside its project")
    for category in ("source_pins", "asset_pins"):
        pins = plan.get(category)
        if not isinstance(pins, dict) or not pins:
            raise ValueError(f"Missing {category}")
        for path, checksum in pins.items():
            if (not Path(path).is_absolute() or not isinstance(checksum, str) or len(checksum) != 64
                    or any(c not in "0123456789abcdef" for c in checksum)):
                raise ValueError(f"Invalid {category} entry")
    required_sources = {str(scope / "src" / name) for name in SOURCE_NAMES}
    if not required_sources <= set(plan["source_pins"]):
        raise ValueError("All execution and scientific sources must be pinned")
    required_assets = {paths["config"], paths["evaluation_protocol"],
                       str(Path(paths["cache_root"]) / "manifest.json"),
                       str(Path(paths["view_root"]) / "manifest.json"), plan["contract_request"]["environment_lock"],
                       str(Path(paths["standard_root"]) / "validate_reproduction.py"),
                       str(Path(paths["standard_root"]) / "task-contract.template.json"),
                       str(Path(paths["standard_root"]) / "result-record.template.json")}
    if not required_assets <= set(plan["asset_pins"]):
        raise ValueError("Config, evaluation, cache, view and environment pins are required")
    assignments = plan.get("ready_assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("Explicit ready GPU assignments required")
    devices = []
    for assignment in assignments:
        gpu = assignment.get("gpu")
        if (type(gpu) is not int or gpu < 0 or assignment.get("readiness") != "ready"
                or not isinstance(assignment.get("host"), str) or not assignment["host"]):
            raise ValueError("Only explicit ready physical GPU assignments can run")
        evidence = assignment.get("evidence", {})
        if plan["asset_pins"].get(evidence.get("path")) != evidence.get("sha256") or not evidence.get("sha256"):
            raise ValueError("Ready assignment needs a pinned evidence file")
        devices.append(gpu)
    if len(set(devices)) != len(devices):
        raise ValueError("GPU assignments must be exclusive")
    occupied = plan.get("occupied_gpu_devices")
    if (not isinstance(occupied, list) or any(type(g) is not int or g < 0 for g in occupied)
            or set(occupied) & set(devices)):
        raise ValueError("Selected GPUs overlap declared occupied devices")
    candidates = plan.get("candidate_gpu_devices", devices)
    if (not isinstance(candidates, list) or not candidates or any(type(g) is not int or not 0 <= g <= 7 for g in candidates)
            or len(set(candidates)) != len(candidates) or not set(devices) <= set(candidates)):
        raise ValueError("Candidate devices are authorization only, not a ready assignment")
    interval = plan.get("candidate_probe_interval_seconds", 20)
    if type(interval) not in (int, float) or isinstance(interval, bool) or not 0 < interval <= 60:
        raise ValueError("Candidate observations need a bounded poll interval")
    if len({a["host"] for a in assignments}) != 1:
        raise ValueError("One queue executes on one explicitly assigned host")
    request = plan["contract_request"]
    if (request["project_root"] != paths["project_root"] or request["scope_root"] != paths["scope_root"]
            or request["resources"]["host"] != assignments[0]["host"]
            or not set(plan["source_pins"]) <= set(request["code_paths"])):
        raise ValueError("Common contract request differs from queue asset/host binding")
    resources = request["resources"]
    if any(type(resources.get(k)) is not int or resources[k] < 1 for k in ("cpu_threads", "memory_limit_bytes")):
        raise ValueError("Explicit per-worker CPU and RAM limits required")
    if (type(plan.get("gpu_memory_budget_bytes")) is not int or plan["gpu_memory_budget_bytes"] < 1
            or resources.get("gpu_memory_budget_bytes") != plan["gpu_memory_budget_bytes"]):
        raise ValueError("One positive allocator budget must bind plan and common contract")
    retries = plan["resource_retry"]
    if (type(retries.get("max_attempts")) is not int or not 1 <= retries["max_attempts"] <= 100
            or not isinstance(retries.get("interval_seconds"), (int, float))
            or isinstance(retries["interval_seconds"], bool) or not 0 <= retries["interval_seconds"] <= 60):
        raise ValueError("Resource retry must be bounded with intervals <=60 seconds")
    if type(plan.get("timeout_seconds_per_case")) is not int or plan["timeout_seconds_per_case"] < 1:
        raise ValueError("Positive fixed maximum wallclock timeout required")
    config = read_json(paths["config"])
    from .mm_evaluate import validate_backbones
    validate_backbones(config.get("backbones"), formal=True)
    if (config.get("status") != "frozen" or config.get("deadline") != DEADLINE
            or config.get("scenarios") != list(SCENARIOS) or config.get("variants") != list(VARIANTS)
            or config.get("seeds") != list(SEEDS) or tuple(config.get("backbones", {})) != BACKBONES
            or config.get("backbone_signatures") != {b: digest(config["backbones"][b]) for b in BACKBONES}
            or config.get("budget_signature") != digest(config.get("training"))):
        raise ValueError("Common frozen training budget or matrix mismatch")
    protocol = read_json(paths["evaluation_protocol"])
    budget = protocol.get("training_budget", {})
    if (budget.get("status") != "FROZEN" or budget.get("config_sha256") != plan["asset_pins"][paths["config"]]
            or budget.get("training_budget_sha256") != config["budget_signature"]
            or budget.get("training") != config["training"]
            or protocol.get("backbones") != config["backbones"]
            or protocol.get("producer_source_sha256") != plan["source_pins"][str(scope / "src/mm_train.py")]
            or protocol.get("evaluator_source_sha256") != plan["source_pins"][str(scope / "src/mm_evaluate.py")]):
        raise ValueError("Evaluation protocol must bind the exact common training budget")
    return True


@contextmanager
def exclusive_lock(path):
    """Local Linux advisory lock retained for the whole worker/controller lifetime."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Already exclusively owned: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class Ledger:
    def __init__(self, path, plan_sha):
        self.path, self.plan_sha = Path(path), plan_sha
        self.lock = threading.RLock()
        self.events, self.states = [], {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                expected = dict(event)
                checksum = expected.pop("event_sha256")
                previous = self.events[-1]["event_sha256"] if self.events else "0" * 64
                if (digest(expected) != checksum or event["previous_event_sha256"] != previous
                        or event["sequence"] != len(self.events) or event["plan_sha256"] != plan_sha):
                    raise ValueError("Ledger integrity or plan binding mismatch")
                self.events.append(event)
                if event.get("case_id"):
                    self.states[event["case_id"]] = event

    def append(self, case_id, status, **details):
        if status not in ("not_started", "running", "succeeded", "failed", "queue"):
            raise ValueError("Unknown ledger status")
        with self.lock:
            event = {"sequence": len(self.events), "previous_event_sha256": self.events[-1]["event_sha256"] if self.events else "0" * 64,
                     "plan_sha256": self.plan_sha, "utc": datetime.now(timezone.utc).isoformat(),
                     "case_id": case_id, "status": status, **details}
            event["event_sha256"] = digest(event)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(canonical(event) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.events.append(event)
            if case_id:
                self.states[case_id] = event
            return event


def default_dependencies():
    # Parent canonical helpers currently use sibling absolute imports.
    source = str(Path(__file__).resolve().parent)
    if source not in sys.path:
        sys.path.insert(0, source)
    import mm_contract
    import mm_supervise
    return mm_contract.create, mm_supervise.run, mm_supervise.group_members


def completion_spec(attempt, case):
    expected = 152 if case["scenario"] == SCENARIOS[0] else 442
    assertions = [("/status", "complete"), ("/formal_eligible", True),
                  ("/backbone", case["backbone"]), ("/scenario", case["scenario"]),
                  ("/arm", case["variant"]), ("/seed", case["seed"]),
                  ("/n_expected_test", expected), ("/LM/n_total", expected),
                  ("/numeric/n_total", expected), ("/numeric/n_finite", expected),
                  ("/numeric/n_failed", 0), ("/numeric/coverage", 1.0),
                  ("/numeric/status", "defined"), ("/numeric/full_population/n", expected)]
    return {"artifact_root": str((attempt / "case").resolve()),
            "assertions": [{"file": "result.json", "pointer": pointer, "equals": value}
                           for pointer, value in assertions],
            "target_counts": {"registered_endpoint": {"file": "result.json", "pointer": "/n_expected_test"}}}


def accepted_attempt(plan, attempt, case, *, archive_fn=None, create_archive=False):
    receipt = read_json(attempt / "supervision/receipt.json")
    contract_path = attempt / "contract/contract.json"
    result_path = attempt / "case/result.json"
    if (receipt.get("status") != "succeeded" or receipt.get("exit_code") != 0
            or receipt.get("remaining_group_members") != []
            or receipt.get("contract_sha256") != sha(contract_path)):
        raise ValueError("Supervisor receipt did not accept this attempt")
    result = read_json(result_path)
    if (result.get("status") != "complete" or result.get("formal_eligible") is not True
            or (result.get("scenario"), result.get("arm"), result.get("seed"), result.get("backbone")) != (case["scenario"], case["variant"], case["seed"], case["backbone"])
            or result.get("protocol_sha256") != plan["asset_pins"][plan["paths"]["config"]]
            or result.get("evaluation_protocol_sha256") != plan["asset_pins"][plan["paths"]["evaluation_protocol"]]):
        raise ValueError("Formal scientific result binding failed")
    spec = completion_spec(attempt, case)
    # Recheck the registered scientific denominator before any claimed terminal acceptance.
    from .mm_archive import _generic_complete
    _generic_complete([(attempt / "case").resolve()], spec,
                      [{"target_id": "registered_endpoint", "required": True,
                        "denominator": 152 if case["scenario"] == SCENARIOS[0] else 442}])
    archive_root = attempt / "archive"
    if create_archive:
        if archive_fn is None:
            from .mm_archive import archive
            archive_fn = archive
        write_new(attempt / "completion_spec.json", spec)
        archived = archive_fn(contract_path, attempt / "supervision", [attempt / "case"],
                              plan["paths"]["standard_root"], archive_root, completion_spec=spec)
        if archived.get("status") != "accepted":
            raise ValueError("Independent terminal archive rejected this attempt")
    archived = read_json(archive_root / "archive_receipt.json")
    terminal = read_json(archive_root / "terminal_validation.json")
    record = read_json(archive_root / "result-record.json")
    if (archived.get("status") != "accepted" or archived.get("contract_sha256") != sha(contract_path)
            or archived.get("record_sha256") != sha(archive_root / "result-record.json")
            or archived.get("validation_sha256") != sha(archive_root / "terminal_validation.json")
            or terminal.get("valid_for_acceptance") is not True or terminal.get("valid") is not True
            or terminal.get("files_skipped")
            or record.get("review", {}).get("status") != "accepted"
            or record.get("lifecycle_status") != "succeeded"
            or record.get("contract_sha256") != sha(contract_path)
            or record.get("archive_provenance", {}).get("completion_spec") != spec):
        raise ValueError("Canonical terminal acceptance or archive byte binding failed")
    return {"receipt_sha256": sha(attempt / "supervision/receipt.json"),
            "result_sha256": sha(result_path), "contract_sha256": sha(contract_path),
            "archive_receipt_sha256": sha(archive_root / "archive_receipt.json"),
            "terminal_validation_sha256": sha(archive_root / "terminal_validation.json"),
            "result_record_sha256": sha(archive_root / "result-record.json")}


def stop_owned_attempt(attempt, terminate_fn=None):
    """Kill only an active leader whose receipt, PGID and exact argv bind this contract."""
    attempt = Path(attempt)
    started_path, contract_path = attempt / "supervision/started.json", attempt / "contract/contract.json"
    if not started_path.exists():
        return "waiting_for_started_receipt"
    started, contract = read_json(started_path), read_json(contract_path)
    pid, pgid = started["pid"], started["pgid"]
    if (type(pid) is not int or pid <= 1 or pid != pgid or pgid == os.getpgrp()
            or started["contract_sha256"] != sha(contract_path)):
        raise RuntimeError("Refuse unbound process-group termination")
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return "leader_absent_supervisor_cleans_descendants"
    expected = contract["execution"]["command"]
    actual = [x.decode() for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
    if (not actual or len(actual) != len(expected) or actual[1:] != expected[1:]
            or Path(actual[0]).resolve() != Path(expected[0]).resolve() or os.getpgid(pid) != pgid):
        raise RuntimeError("Refuse foreign/reused process-group termination")
    if terminate_fn is None:
        source = str(Path(__file__).resolve().parent)
        if source not in sys.path:
            sys.path.insert(0, source)
        from mm_supervise import terminate_group
        terminate_fn = terminate_group
    terminate_fn(pgid)
    return "terminated_bound_owned_group"


def run_queue(plan_path, output, resume=False, *, create_fn=None, run_fn=None,
              members_fn=None, archive_fn=None, now_fn=None, check_hostname=True, probe_fn=None):
    """Dependency injection exists solely for no-GPU CPU fixtures."""
    plan_path, root = Path(plan_path).resolve(), Path(output).resolve()
    plan = read_json(plan_path)
    validate_plan(plan)
    verify_pins(plan)
    if Path(plan["paths"]["scope_root"]).resolve() not in root.parents:
        raise ValueError("Queue output must be inside the new scope")
    if check_hostname and socket.gethostname() != plan["ready_assignments"][0]["host"]:
        raise ValueError("Queue hostname differs from ready assignment")
    root.mkdir(parents=True, exist_ok=resume)
    if create_fn is None or run_fn is None or members_fn is None:
        defaults = default_dependencies()
        create_fn, run_fn, members_fn = create_fn or defaults[0], run_fn or defaults[1], members_fn or defaults[2]
    if plan.get("candidate_gpu_devices") is not None and probe_fn is None:
        source = str(Path(__file__).resolve().parent)
        if source not in sys.path: sys.path.insert(0, source)
        from mm_supervise import probe
        probe_fn = probe
    now_fn = now_fn or (lambda: datetime.now(timezone.utc))
    deadline = datetime.fromisoformat(DEADLINE)
    cases = fixed_cases()
    halt, claim_lock, claimed = threading.Event(), threading.Lock(), set()
    cancellation, watcher_done, active_lock = threading.Event(), threading.Event(), threading.Lock()
    active_attempts = set()
    with exclusive_lock(root / "controller.lock"):
        snapshot = root / "plan.json"
        if resume:
            if not snapshot.is_file() or sha(snapshot) != sha(plan_path):
                raise ValueError("Resume requires the same exact frozen plan bytes")
        else:
            with snapshot.open("xb") as stream:
                stream.write(plan_path.read_bytes())
        plan_sha = sha(snapshot)
        ledger = Ledger(root / "queue_events.jsonl", plan_sha)
        if not ledger.events:
            for case in cases:
                ledger.append(case["case_id"], "not_started", reason="fixed_preregistered_matrix")
        elif set(ledger.states) != {x["case_id"] for x in cases}:
            raise ValueError("Resume ledger does not contain exactly the fixed24 matrix")
        ledger.append(None, "queue", event="controller_start", pid=os.getpid(), host=socket.gethostname(),
                      resume=resume, order="seed_scenario_backbone_variant",
                      gpu_devices=plan.get("candidate_gpu_devices", [x["gpu"] for x in plan["ready_assignments"]]),
                      ready_gpu_devices_at_freeze=[x["gpu"] for x in plan["ready_assignments"]],
                      aggregate_cpu_threads=len(plan.get("candidate_gpu_devices", plan["ready_assignments"])) * plan["contract_request"]["resources"]["cpu_threads"],
                      aggregate_worker_memory_limit_bytes=len(plan.get("candidate_gpu_devices", plan["ready_assignments"])) * plan["contract_request"]["resources"]["memory_limit_bytes"])
        for case in cases:
            old = ledger.states[case["case_id"]]
            if old["status"] != "running":
                continue
            attempt = root / old["attempt"]
            started_path = attempt / "supervision/started.json"
            if started_path.exists() and members_fn(read_json(started_path)["pgid"]):
                raise RuntimeError("Resume refuses a still-active owned process group")
            try:
                binding = accepted_attempt(plan, attempt, case)
                ledger.append(case["case_id"], "succeeded", attempt=old["attempt"], recovered=True, **binding)
            except (OSError, ValueError, KeyError):
                ledger.append(case["case_id"], "failed", attempt=old["attempt"],
                              reason="interrupted_without_accepted_terminal_receipt", recovered=True)

        def worker(assignment):
            gpu = assignment["gpu"]
            case_id, relative = None, None
            try:
                lock_path = Path(plan["paths"]["scope_root"]) / "resources" / f"queue_gpu{gpu}.lock"
                with exclusive_lock(lock_path):
                    observation_number = 0
                    while not halt.is_set():
                        if plan.get("candidate_gpu_devices") is not None:
                            with claim_lock:
                                pending = any(ledger.states[x["case_id"]]["status"] == "not_started"
                                              and not ledger.states[x["case_id"]].get("exhausted") for x in cases)
                            if not pending or (deadline - now_fn()).total_seconds() <= 125:
                                return
                            observation = probe_fn(assignment["host"], gpu,
                                plan["contract_request"]["resources"]["memory_limit_bytes"], Path(plan["paths"]["scope_root"]))
                            observation_number += 1
                            observations = root / "resource_observations" / f"gpu{gpu}"
                            observations.mkdir(parents=True, exist_ok=True)
                            # A resumed controller never overwrites prior observations.
                            observed_path = observations / f"controller{os.getpid()}_{time.time_ns()}_{observation_number}.json"
                            write_new(observed_path, observation)
                            ledger.append(None, "queue", event="candidate_resource_observation", gpu=gpu,
                                readiness=observation.get("readiness"), reasons=observation.get("reasons"),
                                observation=str(observed_path.relative_to(root)), observation_sha256=sha(observed_path))
                            if observation.get("readiness") != "ready":
                                halt.wait(plan.get("candidate_probe_interval_seconds", 20))
                                continue
                        with claim_lock:
                            if now_fn() >= deadline:
                                return
                            case = next((x for x in cases if x["case_id"] not in claimed
                                         and ledger.states[x["case_id"]]["status"] == "not_started"
                                         and not ledger.states[x["case_id"]].get("exhausted")), None)
                            if case is None:
                                return
                            claimed.add(case["case_id"])
                        case_id = case["case_id"]
                        try:
                            for retry in range(plan["resource_retry"]["max_attempts"]):
                                if halt.is_set() or (deadline - now_fn()).total_seconds() <= 125:
                                    ledger.append(case_id, "not_started", reason="deadline_or_common_halt", exhausted=True)
                                    break
                                try:
                                    verify_pins(plan)
                                except (OSError, ValueError) as exc:
                                    ledger.append(case_id, "not_started", reason="common_pin_mismatch", error=str(exc), exhausted=True)
                                    halt.set()
                                    break
                                case_root = root / "cases" / case_id
                                case_root.mkdir(parents=True, exist_ok=True)
                                # Existing attempts are never overwritten, including resource-busy admissions.
                                attempt_number = 1
                                while (case_root / f"attempt_{attempt_number:04d}").exists():
                                    attempt_number += 1
                                attempt = case_root / f"attempt_{attempt_number:04d}"
                                attempt.mkdir()
                                relative = str(attempt.relative_to(root))
                                ledger.append(case_id, "running", attempt=relative, gpu=gpu, retry=retry)
                                (attempt / "contract").mkdir()
                                request = deepcopy(plan["contract_request"])
                                request.update(run_id=f"{case_id}_attempt{attempt_number:04d}",
                                    seeds=[case["seed"]], timeout_seconds=min(plan["timeout_seconds_per_case"],
                                      int((deadline - now_fn()).total_seconds() - 120)),
                                    phase="formal", denominator=152 if case["scenario"] == SCENARIOS[0] else 442)
                                request["resources"]["gpu"] = gpu
                                request["command"] = [plan["paths"]["python"], "-m", "src.mm_case", "--plan", str(snapshot),
                                    "--plan-sha256", plan_sha, "--scenario", case["scenario"], "--variant", case["variant"],
                                    "--backbone", case["backbone"], "--seed", str(case["seed"]), "--physical-gpu", str(gpu),
                                    "--contract", str(attempt / "contract/contract.json"), "--output", str(attempt / "case")]
                                write_new(attempt / "request.json", request)
                                try:
                                    contract = create_fn(request, plan["paths"]["standard_root"], attempt / "contract/contract.json")
                                except Exception as exc:
                                    write_new(attempt / "controller_failure.json", {"stage": "contract_create", "error": str(exc),
                                              "type": type(exc).__name__, "traceback": traceback.format_exc()})
                                    ledger.append(case_id, "failed", attempt=relative, reason="contract_admission_error", error=str(exc))
                                    halt.set()  # Common standard/asset/config failures cannot be papered over.
                                    break
                                if contract is None:
                                    ledger.append(case_id, "not_started", attempt=relative, reason="resource_busy", gpu=gpu,
                                                  exhausted=retry + 1 == plan["resource_retry"]["max_attempts"])
                                    halt.wait(plan["resource_retry"]["interval_seconds"])
                                    continue
                                try:
                                    if halt.is_set():
                                        ledger.append(case_id, "not_started", attempt=relative, reason="common_halt_before_spawn", exhausted=True)
                                        break
                                    with active_lock:
                                        active_attempts.add(attempt)
                                    code = run_fn(attempt / "contract/contract.json",
                                        Path(plan["paths"]["standard_root"]) / "validate_reproduction.py", attempt / "supervision", gpu=gpu)
                                    if code == 3:
                                        receipt = read_json(attempt / "supervision/receipt.json")
                                        if receipt.get("reason") == "fresh_resource_gate":
                                            ledger.append(case_id, "not_started", attempt=relative, reason="resource_busy_at_boundary", gpu=gpu,
                                                          exhausted=retry + 1 == plan["resource_retry"]["max_attempts"])
                                            halt.wait(plan["resource_retry"]["interval_seconds"])
                                            continue
                                        ledger.append(case_id, "not_started", attempt=relative, reason="remaining_window_at_start", exhausted=True)
                                        halt.set()
                                        break
                                    if code != 0:
                                        raise RuntimeError(f"Supervisor exited {code}")
                                    binding = accepted_attempt(plan, attempt, case, archive_fn=archive_fn, create_archive=True)
                                    verify_pins(plan)
                                    ledger.append(case_id, "succeeded", attempt=relative, gpu=gpu, **binding)
                                except Exception as exc:
                                    write_new(attempt / "controller_failure.json", {"stage": "supervision_or_acceptance", "error": str(exc),
                                              "type": type(exc).__name__, "traceback": traceback.format_exc()})
                                    ledger.append(case_id, "failed", attempt=relative, gpu=gpu, reason="execution_or_acceptance_failure", error=str(exc))
                                    try:
                                        verify_pins(plan)
                                    except (OSError, ValueError):
                                        halt.set()
                                finally:
                                    with active_lock:
                                        active_attempts.discard(attempt)
                                break
                        finally:
                            with claim_lock:
                                claimed.discard(case_id)
            except Exception as exc:
                if case_id is not None and ledger.states[case_id]["status"] == "running":
                    ledger.append(case_id, "failed", attempt=relative, reason="unhandled_controller_error", error=str(exc))
                ledger.append(None, "queue", event="worker_halt", gpu=gpu, error=str(exc), traceback=traceback.format_exc())
                halt.set()

        def cancel_watcher():
            last_outcomes = {}
            while not watcher_done.is_set():
                if cancellation.is_set():
                    with active_lock:
                        attempts = list(active_attempts)
                    for attempt in attempts:
                        try:
                            outcome = stop_owned_attempt(attempt)
                        except (OSError, ValueError, KeyError, RuntimeError) as exc:
                            outcome = "termination_refused: " + str(exc)
                        if last_outcomes.get(attempt) != outcome:
                            ledger.append(None, "queue", event="signal_owned_group_cleanup", attempt=str(attempt.relative_to(root)), outcome=outcome)
                            last_outcomes[attempt] = outcome
                watcher_done.wait(.2)

        previous_handlers = {}
        def handle_signal(number, _frame):
            halt.set()
            cancellation.set()
        if threading.current_thread() is threading.main_thread():
            for number in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[number] = signal.signal(number, handle_signal)
        watcher = threading.Thread(target=cancel_watcher, name="own_process_group_cleanup")
        watcher.start()
        worker_assignments = ([{"gpu": gpu, "host": plan["ready_assignments"][0]["host"]}
                               for gpu in plan["candidate_gpu_devices"]]
                              if "candidate_gpu_devices" in plan else plan["ready_assignments"])
        workers = [threading.Thread(target=worker, args=(a,), name=f"physical_gpu_{a['gpu']}")
                   for a in worker_assignments]
        for thread in workers:
            thread.start()
        try:
            for thread in workers:
                thread.join()
        except BaseException:
            halt.set()
            cancellation.set()
            # Each supervisor still owns and bounds its independent process group.
            for thread in workers:
                thread.join()
            raise
        finally:
            watcher_done.set()
            watcher.join()
            for number, handler in previous_handlers.items():
                signal.signal(number, handler)
        counts = {status: sum(x["status"] == status for x in ledger.states.values())
                  for status in ("not_started", "running", "succeeded", "failed")}
        ledger.append(None, "queue", event="controller_terminal", counts=counts,
                      halted=halt.is_set(), deadline_reached=now_fn() >= deadline)
        return {"counts": counts, "halted": halt.is_set(), "ledger_path": str(ledger.path),
                "scientific_matrix_size": 24, "plan_sha256": plan_sha}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = run_queue(args.plan, args.output, args.resume)
    print(canonical(result))
    return 0 if result["counts"]["succeeded"] == 24 else 2


if __name__ == "__main__":
    raise SystemExit(main())

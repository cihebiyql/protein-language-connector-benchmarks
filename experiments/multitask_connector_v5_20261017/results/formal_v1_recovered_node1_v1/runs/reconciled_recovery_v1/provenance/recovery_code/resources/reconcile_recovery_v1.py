"""Build a byte-preserving, independently accepted view of one technical retry.

This adds no training, selection rule, or analysis. Original ledgers and attempts
remain untouched; only the derived ledger's attempt locations are remapped.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

TARGET = "elm__M3__qwen3_4b_2507__seed17"
TERMINAL = {"accepted", "failed", "not_started_resource_timeout"}
SELECTION_REASON = "Recover original model-loading OOM; technical success only, without inspecting scores"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def unique(pairs):
        value = dict(pairs)
        if len(value) != len(pairs):
            raise ValueError("Duplicate JSON metadata key")
        return value
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def safe_path(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Symlinks are forbidden in recovery inputs or outputs")
    return path.resolve(strict=True)


def tree(path):
    """Hash every regular byte stream; reject indirection and unusual files."""
    path = safe_path(path)
    result = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("Symlink inside attempt")
        if item.is_dir():
            continue
        if not item.is_file() or item.stat().st_nlink != 1:
            raise ValueError("Only independently copied regular files are permitted")
        result[item.relative_to(path).as_posix()] = {"sha256": sha(item), "bytes": item.stat().st_size}
    digest = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"tree_sha256": digest, "files": result}


def accepted_attempt(plan, attempt, case):
    scope = Path(plan["paths"]["scope_root"])
    if str(scope) not in sys.path:
        sys.path.insert(0, str(scope))
    from src.mt_case import accepted_attempt as check
    return check(plan, attempt, case)


def validate_retry(scope, retry, plan, binding):
    if retry != scope / "runs/recovery_retry_v1/cases" / TARGET / "attempt_002":
        raise ValueError("Only the authorized new attempt_002 may repair the fixed case")
    spec_source = retry / "spec.json"
    spec = read(spec_source)
    case = next(c for c in plan["cases"] if c["case_id"] == TARGET)
    if (spec.get("case") != case or spec.get("plan_path") != binding["plan_path"]
            or spec.get("plan_sha256") != binding["plan_sha256"]):
        raise ValueError("Retry case or original frozen plan binding differs")
    policy_path = safe_path(spec["recovery_policy_path"])
    adapter_path = safe_path(scope / "resources/recovery_worker_v1.py")
    if not policy_path.is_relative_to(scope):
        raise ValueError("Recovery policy must remain inside this scope")
    if sha(policy_path) != spec.get("recovery_policy_sha256") or sha(adapter_path) != spec.get("recovery_adapter_sha256"):
        raise ValueError("Recovery resource policy or entrypoint SHA differs")
    policy = read(policy_path)
    if (policy.get("schema") != "v5-recovery-resource-policy.v1" or policy.get("case_id") != TARGET
            or policy.get("authorized_gpu_devices") != [3, 4, 5, 6]
            or policy.get("original_gpu_devices") != [3, 4, 5]
            or type(policy.get("maximum_distinct_owned_gpus")) is not int
            or policy["maximum_distinct_owned_gpus"] != 4
            or policy.get("scope_root") != str(scope)
            or policy.get("original_plan_path") != binding["plan_path"]
            or policy.get("original_plan_sha256") != binding["plan_sha256"]
            or type(spec.get("gpu")) is not int or spec["gpu"] not in [3, 4, 5, 6]):
        raise ValueError("Retry must retain exact four-card, case-bound authorization")
    expected_worker = [plan["paths"]["python"], str(adapter_path), "--mode", "worker", "--spec", str(spec_source)]
    resources = plan["resource_policy"]
    expected_resources = {"gpu_memory_budget_bytes": resources["gpu_memory_budget_bytes"]["qwen3_4b_2507"],
                          "rss_limit_bytes": resources["rss_limit_bytes"],
                          "storage_reserve_bytes": resources["storage_reserve_bytes"],
                          "expected_host": resources["expected_host"],
                          "timeout_seconds": plan["per_case_timeout_seconds"], "cwd": str(scope)}
    if spec.get("command") != expected_worker or any(spec.get(k) != v for k, v in expected_resources.items()):
        raise ValueError("Retry worker must use the disclosed recovery adapter")
    finished = read(retry / "supervision/finished.json")
    if finished.get("status") not in {"completed", "failed", "supervisor_error"}:
        raise ValueError("Retry supervision is not terminal")
    receipts = {}
    for phase, out, budget in (("train-canary", "qualification", "qualification_device_budget.json"),
                                ("formal", "artifacts", "device_budget.json")):
        path = retry / f"{phase}_process.json"
        if not path.exists() and finished.get("status") != "completed":
            continue
        receipt = read(path)
        expected = [plan["paths"]["python"], str(adapter_path), "--mode", "train-child", "--spec", str(spec_source),
                    "--phase", phase, "--output", str(retry / out), "--receipt", str(retry / budget)]
        if (receipt.get("command") != expected or receipt.get("phase") != phase
                or receipt.get("recovery_policy_sha256") != spec["recovery_policy_sha256"]
                or receipt.get("recovery_adapter_sha256") != spec["recovery_adapter_sha256"]):
            raise ValueError("Actual fresh phase process lacks exact adapter, source spec, phase and SHA binding")
        receipts[phase] = {"path": str(path), "sha256": sha(path), "source_spec_path": str(spec_source)}
    return {"spec_source_path": str(spec_source), "spec_sha256": sha(spec_source),
            "recovery_policy_path": str(policy_path), "recovery_policy_sha256": sha(policy_path),
            "recovery_adapter_path": str(adapter_path), "recovery_adapter_sha256": sha(adapter_path),
            "execution_entrypoint_changed": True, "part_of_original_frozen_source_closure": False,
            "scientific_source_files_modified": False, "authorized_gpu_devices": [3, 4, 5, 6],
            "maximum_distinct_owned_gpus": 4, "phase_receipts": receipts}


def build_view(scope, original_queue, retry_attempt, output):
    scope, original, retry = map(safe_path, (scope, original_queue, retry_attempt))
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError("Fresh output only; interrupted or completed recovery views are preserved")
    if original != scope / "runs/formal_v1" or output != scope / "runs/reconciled_recovery_v1":
        raise ValueError("Recovery is restricted to the original queue and one explicitly named derived view")
    if any(p.is_symlink() for p in (output.parent, *output.parent.parents)):
        raise ValueError("Symlink output parent forbidden")
    binding = read(original / "queue_binding.json")
    plan_path = safe_path(binding["plan_path"])
    if sha(plan_path) != binding["plan_sha256"]:
        raise ValueError("Original frozen plan SHA changed")
    plan = read(plan_path)
    cases = {c["case_id"]: c for c in plan["cases"]}
    expected = {(t, m, b, s) for t in ("meltome", "elm", "rt") for s in (17, 42, 2026)
                for m, b in (("ESM", "none"), ("M0", "qwen25_1p5b"), ("M3", "qwen25_1p5b"),
                             ("M0", "qwen3_4b_2507"), ("M3", "qwen3_4b_2507"))}
    if (len(cases) != 45 or len(plan["cases"]) != 45
            or {(c["task"], c["method"], c["backbone"], c["seed"]) for c in cases.values()} != expected
            or plan["paths"]["scope_root"] != str(scope)):
        raise ValueError("Original complete 45-case matrix is required")
    if any(cid != f"{c['task']}__{c['method']}__{c['backbone']}__seed{c['seed']}" for cid, c in cases.items()):
        raise ValueError("Unsafe or noncanonical case ID")
    complete = read(original / "complete.json")
    read(original / "analysis_exit.json")  # Must have reached the original queue's analysis terminal boundary.
    ledger = original / "queue_events.jsonl"
    events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest = {}
    for event in events:
        if event.get("case_id") not in cases:
            raise ValueError("Ledger contains unknown case")
        latest[event["case_id"]] = event
    if (set(latest) != set(cases) or complete.get("case_count") != 45
            or complete.get("plan_sha256") != binding["plan_sha256"]
            or complete.get("cases") != latest or any(e.get("status") not in TERMINAL for e in latest.values())):
        raise ValueError("Original queue must have a matching complete terminal 45-case ledger")
    failed = original / "cases" / TARGET / "attempt_001"
    if latest[TARGET].get("status") != "failed" or safe_path(latest[TARGET]["attempt"]) != failed:
        raise ValueError("Retry only repairs the preserved original failed attempt_001")
    extra = validate_retry(scope, retry, plan, binding)
    sources = []
    mapping = {}
    for cid in cases:
        case_root = original / "cases" / cid
        if not case_root.exists():
            continue
        if safe_path(case_root).parent != original / "cases":
            raise ValueError("Attempt outside original queue")
        for attempt in sorted(case_root.iterdir()):
            if not attempt.is_dir() or not attempt.name.startswith("attempt_"):
                raise ValueError("Unexpected entry in original case directory")
            if safe_path(attempt).parent != case_root:
                raise ValueError("External attempt forbidden")
            destination = output / "cases" / cid / attempt.name
            mapping[str(attempt)] = str(destination)
            sources.append((attempt, destination, "original_failed" if attempt == failed else "original"))
    retry_destination = output / "cases" / TARGET / "attempt_002"
    if str(retry_destination) in mapping.values():
        raise ValueError("New retry must not overwrite any original attempt")
    sources.append((retry, retry_destination, "authorized_technical_retry"))
    before = {str(src): tree(src) for src, _, _ in sources}
    for event in events:
        if "attempt" in event and str(safe_path(event["attempt"])) not in mapping:
            raise ValueError("Original ledger references an external or missing attempt")
    output.mkdir(exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    original_metadata = {}
    for src in sorted(original.iterdir()):
        if src.is_symlink():
            raise ValueError("Symlink queue metadata forbidden")
        if src.is_file():
            if src.stat().st_nlink != 1:
                raise ValueError("Hardlinked queue metadata forbidden")
            destination = provenance / ("original_" + src.name)
            shutil.copy2(src, destination)
            if sha(src) != sha(destination):
                raise ValueError("Original queue provenance copy changed")
            original_metadata[src.name] = {"sha256": sha(src), "copy": str(destination)}
    extra_provenance = {}
    for name, src in (("original_frozen_plan.json", plan_path),
                      ("recovery_resource_policy.json", Path(extra["recovery_policy_path"])),
                      ("recovery_worker_v1.py", Path(extra["recovery_adapter_path"]))):
        destination = provenance / name
        shutil.copy2(src, destination)
        if sha(src) != sha(destination):
            raise ValueError("Recovery authorization provenance copy changed")
        extra_provenance[name] = {"source": str(src), "copy": str(destination), "sha256": sha(destination)}
    attempts = []
    for src, destination, role in sources:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, destination, copy_function=shutil.copy2)
        after = tree(destination)
        if tree(src) != before[str(src)] or after != before[str(src)]:
            raise ValueError("Attempt changed while copying or copy bytes differ")
        attempts.append({"source": str(src), "copy": str(destination), "role": role,
                         "source_tree": before[str(src)], "copy_tree_sha256": after["tree_sha256"]})
    new_events = []
    for event in events:
        remapped = dict(event)
        if "attempt" in remapped:
            remapped["attempt"] = mapping[remapped["attempt"]]
        new_events.append(remapped)
    terminal = {}
    for cid, case in cases.items():
        old = latest[cid]
        target = retry_destination if cid == TARGET else Path(mapping[old["attempt"]]) if old.get("attempt") else None
        event = {"at_unix": time.time(), "case_id": cid, "status": old["status"],
                 "derived_reconciliation": True}
        if target is not None:
            event["attempt"] = str(target)
        if cid == TARGET or old["status"] == "accepted":
            try:
                event.update(status="accepted", binding=accepted_attempt(plan, target, case))
            except Exception as exc:
                event.update(status="failed", error=repr(exc))
        elif old.get("error"):
            event["error"] = old["error"]
        terminal[cid] = event
        new_events.append(event)
    manifest = {"schema": "v5-reconciliation-manifest.v1", "created_unix": time.time(),
                "original_queue": str(original), "output": str(output), "case_id": TARGET,
                "original_plan_path": str(plan_path), "original_plan_sha256": binding["plan_sha256"],
                "selection_reason": SELECTION_REASON, "scores_used_to_choose_attempt": False,
                "metadata_derivation": "Only copied ledger attempt paths and new terminal acceptance events are derived; attempt bytes are unchanged",
                "original_ledger_sha256": sha(ledger), "original_metadata": original_metadata,
                "extra_provenance": extra_provenance,
                "original_failed_attempt_tree_sha256": before[str(failed)]["tree_sha256"],
                "attempts": attempts, "retry_execution_disclosure": extra,
                "retry_accepted": terminal[TARGET]["status"] == "accepted"}
    write(output / "RECONCILIATION_MANIFEST.json", manifest)
    write(output / "queue_binding.json", {**binding, "derived_recovery_view": True,
                                         "reconciliation_manifest_sha256": sha(output / "RECONCILIATION_MANIFEST.json")})
    (output / "queue_events.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n" for e in new_events), encoding="utf-8")
    counts = dict(Counter(e["status"] for e in terminal.values()))
    write(output / "queue_state.json", {"schema": "mt-queue-state.v1", "plan_sha256": binding["plan_sha256"],
         "at_unix": time.time(), "case_count": 45, "counts": counts, "cases": terminal})
    write(output / "complete.json", {"schema": "mt-queue-complete.v1", "plan_sha256": binding["plan_sha256"],
         "finished_unix": time.time(), "case_count": 45, "accepted": counts.get("accepted", 0), "cases": terminal})
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("scope", "original-queue", "retry-attempt", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    result = build_view(args.scope, args.original_queue, args.retry_attempt, args.output)
    print(json.dumps({"output": args.output, "retry_accepted": result["retry_accepted"]}))


if __name__ == "__main__":
    main()

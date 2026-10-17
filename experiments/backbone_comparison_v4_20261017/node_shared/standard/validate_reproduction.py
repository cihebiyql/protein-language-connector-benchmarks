"""Validate administrative reproduction contracts and records; execute no workloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

LEVELS = {"registered", "assets_only", "static_audit", "smoke", "component", "published_output_reanalysis", "independent_inference", "training", "full_paper"}
COVERAGES = {"full", "partial_subset", "component_only", "published_outputs_only", "not_evaluated"}
INTEGRITY = {"complete", "partial", "missing", "unknown", "not_applicable"}
STATUSES = {"registered", "ready", "running", "succeeded", "partial", "failed", "blocked", "not_run", "expired", "restricted"}
SHA = re.compile(r"^[0-9a-fA-F]{64}$")
COMMIT = re.compile(r"^[0-9a-fA-F]{40,64}$")


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("must be an ISO timestamp with timezone")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone is required")
    return parsed.astimezone(timezone.utc)


def present(value):
    return value is not None and value != "" and value != [] and value != {}


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_json(path):
    def reject_constant(value):
        raise ValueError("non-finite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject_constant)


def file_sha256(path):
    """Hash bounded chunks; compatible with Python 3.10 and large artifacts."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(contract, result=None, *, contract_bytes=None, at=None, allow_template=False,
             check_files=False, contract_directory=None, result_directory=None):
    errors, blockers, checked, skipped = [], [], [], []
    now = timestamp(at) if isinstance(at, str) else (at or datetime.now(timezone.utc))

    def error(path, message):
        errors.append({"field": path, "message": message})

    def need(obj, key, path):
        value = obj.get(key) if isinstance(obj, dict) else None
        if not present(value):
            error(path + "." + key, "required value is missing")
        return value

    def enum(value, choices, path):
        if value not in choices:
            error(path, "must be one of: " + ", ".join(sorted(choices)))

    def date(value, path):
        try:
            return timestamp(value)
        except (ValueError, TypeError) as exc:
            error(path, str(exc))
            return None

    def shape(document, kind, required):
        if not isinstance(document, dict):
            error(kind, "must be a JSON object")
            return False
        if document.get("record_type") != kind:
            error(kind + ".record_type", "expected " + kind)
        if document.get("schema_version") != "1.0":
            error(kind + ".schema_version", "expected 1.0")
        if not isinstance(document.get("is_template"), bool):
            error(kind + ".is_template", "must be boolean")
        for key in required:
            if key not in document:
                error(kind + "." + key, "required field is missing")
        return True

    def numeric_scan(value, path):
        if isinstance(value, float) and not math.isfinite(value):
            error(path, "numeric values must be finite")
        elif isinstance(value, dict):
            for key, child in value.items():
                numeric_scan(child, path + "." + key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                numeric_scan(child, f"{path}[{index}]")

    def file_scan(value, base, trail):
        if isinstance(value, list):
            for index, child in enumerate(value):
                file_scan(child, base, f"{trail}[{index}]")
            return
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            label = trail + "." + key
            if key == "evidence_paths" and isinstance(child, list):
                for index, path in enumerate(child):
                    inspect_file(path, None, base, f"{label}[{index}]", value.get("location"))
            elif (key == "path" or key.endswith("_path")) and isinstance(child, str) and child:
                hash_key = "sha256" if key == "path" else key[:-5] + "_sha256"
                inspect_file(child, value.get(hash_key), base, label, value.get("location"))
            elif isinstance(child, (dict, list)):
                file_scan(child, base, label)

    def inspect_file(raw, expected, base, label, location):
        if not isinstance(raw, str) or not raw:
            error(label, "path must be a nonempty string")
            return
        remote = location == "remote" or "://" in raw or (re.match(r"^[^/\\]+@[^:]+:", raw) is not None)
        if os.name == "nt" and raw.startswith("/"):
            remote = True  # Windows: POSIX paths refer to the remote research host.
        if remote:
            skipped.append({"field": label, "path": raw, "reason": "remote reference; local integrity not verified"})
            return
        path = Path(raw)
        if not path.is_absolute():
            path = Path(base or ".") / path
        if not path.exists():
            error(label, "local reference does not exist: " + str(path))
            return
        receipt = {"field": label, "path": str(path), "hash_verified": False}
        if expected is not None:
            if not isinstance(expected, str) or not SHA.fullmatch(expected):
                error(label, "paired SHA256 must contain 64 hexadecimal characters")
            elif not path.is_file():
                error(label, "cannot verify a file SHA256 against a directory")
            else:
                digest = file_sha256(path)
                if digest.lower() != expected.lower():
                    error(label, "SHA256 mismatch")
                else:
                    receipt["hash_verified"] = True
        checked.append(receipt)

    if not shape(contract, "task_contract", ["task_id", "run_id", "contract_status", "identity", "execution_scope", "timing", "assets", "targets", "environment", "resources", "execution", "stop_conditions", "review"]):
        return {"valid": False, "valid_for_execution": False, "valid_for_acceptance": False, "errors": errors, "execution_blockers": [], "files_checked": [], "files_skipped": []}
    numeric_scan(contract, "contract")
    template = contract.get("is_template") is True
    if template and not allow_template:
        error("contract.is_template", "template requires --allow-template for structure-only validation")
    if template:
        blockers.append("template is never executable")
    for key in ("identity", "execution_scope", "timing", "assets", "environment", "resources", "execution", "review"):
        if not isinstance(contract.get(key), dict):
            error("contract." + key, "must be an object")
    if errors:
        return {"valid": False, "valid_for_execution": False, "valid_for_acceptance": False, "errors": errors, "execution_blockers": blockers, "files_checked": checked, "files_skipped": skipped}
    scope, timing = contract["execution_scope"], contract["timing"]
    enum(scope.get("intended_level"), LEVELS, "contract.execution_scope.intended_level")
    enum(scope.get("scope_coverage"), COVERAGES, "contract.execution_scope.scope_coverage")
    enum(contract.get("contract_status"), {"draft", "frozen", "expired", "closed"}, "contract.contract_status")
    targets = contract.get("targets")
    if not isinstance(targets, list):
        error("contract.targets", "must be an array")
        targets = []
    target_map = {}
    for index, target in enumerate(targets):
        label = f"contract.targets[{index}]"
        if not isinstance(target, dict):
            error(label, "must be an object")
            continue
        target_id = need(target, "target_id", label)
        if target_id in target_map:
            error(label + ".target_id", "duplicate endpoint")
        target_map[target_id] = target
        if not isinstance(target.get("required"), bool):
            error(label + ".required", "must be boolean")
        if not template and target.get("required"):
            for key in ("paper_endpoint", "metric", "metric_definition", "statistical_unit", "denominator", "unit"):
                need(target, key, label)
            tolerance = target.get("tolerance", {})
            enum(tolerance.get("kind"), {"absolute", "relative", "both", "exact", "not_comparable"}, label + ".tolerance.kind")
            if tolerance.get("kind") == "not_comparable":
                need(tolerance, "reason", label + ".tolerance")
            else:
                need(target, "paper_value_source", label)
                if target.get("paper_value") is None:
                    error(label + ".paper_value", "comparison requires a known paper value")
                for name in ("absolute", "relative"):
                    if tolerance.get("kind") in {name, "both"} and (not finite_number(tolerance.get(name)) or tolerance[name] < 0):
                        error(label + ".tolerance." + name, "must be a finite nonnegative number")
    effective = frozen = None
    if not template:
        for key in ("task_id", "run_id"):
            need(contract, key, "contract")
        if contract.get("contract_status") != "frozen":
            blockers.append("contract is not frozen")
        frozen = date(need(contract, "frozen_at", "contract"), "contract.frozen_at")
        for key in ("title", "doi", "publication_status"):
            need(contract["identity"], key, "contract.identity")
        if contract["identity"].get("publication_status") == "unknown":
            error("contract.identity.publication_status", "real contract requires publication version identity")
        pdf = contract["identity"].get("pdf", {})
        need(pdf, "path", "contract.identity.pdf")
        if not isinstance(pdf.get("sha256"), str) or not SHA.fullmatch(pdf["sha256"]):
            error("contract.identity.pdf.sha256", "valid PDF SHA256 is required")
        for key in ("objective", "authorization_reference"):
            need(scope, key, "contract.execution_scope")
        if scope.get("safety_assessment") != "allowed":
            blockers.append("execution scope has not been assessed as allowed")
        restriction = scope.get("restriction_check", {})
        if restriction.get("prior_platform_restriction") is not False:
            blockers.append("prior platform restriction is present or unverified")
        if timing.get("prior_window_expired") is not False:
            blockers.append("prior execution window expired or has not been verified")
        if timing.get("retry_resets_clock") is not False:
            error("contract.timing.retry_resets_clock", "retries must never reset the clock")
        cap = timing.get("wall_clock_cap_hours")
        if not finite_number(cap) or not 0 < cap <= 10:
            error("contract.timing.wall_clock_cap_hours", "must be greater than zero and at most 10")
            cap = 10
        first = date(need(timing, "first_substantive_started_at", "contract.timing"), "contract.timing.first_substantive_started_at")
        original = date(need(timing, "original_deadline_at", "contract.timing"), "contract.timing.original_deadline_at")
        preparation = date(timing["earliest_associated_preparation_at"], "contract.timing.earliest_associated_preparation_at") if timing.get("earliest_associated_preparation_at") else None
        effective = date(need(timing, "effective_deadline_at", "contract.timing"), "contract.timing.effective_deadline_at")
        deadlines = [value for value in (original, first + timedelta(hours=cap) if first else None, preparation + timedelta(hours=cap) if preparation else None) if value]
        for index, raw in enumerate(timing.get("additional_deadlines_at", [])):
            value = date(raw, f"contract.timing.additional_deadlines_at[{index}]")
            if value:
                deadlines.append(value)
        if first and original and original > first + timedelta(hours=cap):
            error("contract.timing.original_deadline_at", "exceeds the first-start wall-clock cap")
        if effective and deadlines and effective != min(deadlines):
            error("contract.timing.effective_deadline_at", "must equal the earliest applicable deadline")
        if effective and now >= effective:
            blockers.append("effective execution deadline has passed")
        timeout = contract["execution"].get("timeout_seconds")
        if not finite_number(timeout) or timeout <= 0:
            error("contract.execution.timeout_seconds", "must be a positive finite number")
        elif effective and now < effective and timeout > (effective - now).total_seconds():
            blockers.append("timeout exceeds remaining execution window at validation time")
        resources = contract["resources"]
        for key in ("cpu_threads", "memory_limit_bytes"):
            value = resources.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                error("contract.resources." + key, "positive integer limit is required")
        for key in ("expected_hostname", "resource_probe_path", "resource_probe_at", "start_gate", "budget_basis"):
            need(resources, key, "contract.resources")
        if resources.get("no_preemption") is not True:
            error("contract.resources.no_preemption", "must be true")
        for key in ("command", "working_directory", "seed_policy", "stdout_path", "stderr_path"):
            need(contract["execution"], key, "contract.execution")
        if contract["execution"].get("process_control", {}).get("whole_tree_bounded") is not True:
            error("contract.execution.process_control.whole_tree_bounded", "bounded process-tree control is required")
        for key in ("isolation", "environment_path", "lockfile_path"):
            need(contract["environment"], key, "contract.environment")
        if contract["environment"].get("shared_environment_modified") is not False:
            error("contract.environment.shared_environment_modified", "shared environment must remain unchanged")
        if not targets:
            error("contract.targets", "at least one explicit endpoint is required")
    assets = contract["assets"]
    for category in ("datasets", "code", "models"):
        entries = assets.get(category)
        if not isinstance(entries, list) or not entries:
            error("contract.assets." + category, "nonempty explicit asset assessment is required")
            continue
        for index, entry in enumerate(entries):
            label = f"contract.assets.{category}[{index}]"
            if not isinstance(entry, dict):
                error(label, "must be an object")
                continue
            enum(entry.get("integrity"), INTEGRITY, label + ".integrity")
            if not template and entry.get("integrity") in {"unknown", "missing", "partial", "not_applicable"}:
                need(entry, "notes", label)
            if not template and entry.get("integrity") == "complete" and category in {"datasets", "models"}:
                for key in ("name", "path"):
                    need(entry, key, label)
                hash_key = "manifest_sha256" if category == "datasets" else "sha256"
                if not isinstance(entry.get(hash_key), str) or not SHA.fullmatch(entry[hash_key]):
                    error(label + "." + hash_key, "complete assets require a frozen SHA256")
            if not template and category == "code" and entry.get("integrity") != "not_applicable":
                if not (isinstance(entry.get("commit"), str) and COMMIT.fullmatch(entry["commit"])) and not (isinstance(entry.get("archive_sha256"), str) and SHA.fullmatch(entry["archive_sha256"])):
                    error(label, "frozen code commit or archive SHA256 is required")
    accepted = False
    if result is not None and shape(result, "result_record", ["task_id", "run_id", "contract_sha256", "lifecycle_status", "reproduction_level", "scope_coverage", "claims", "assets_integrity", "execution", "metric_comparisons", "artifacts", "differences", "incomplete_reasons", "review"]):
        numeric_scan(result, "result")
        result_template = result.get("is_template") is True
        if result_template and not allow_template:
            error("result.is_template", "template requires --allow-template")
        enum(result.get("lifecycle_status"), STATUSES, "result.lifecycle_status")
        enum(result.get("reproduction_level"), LEVELS, "result.reproduction_level")
        enum(result.get("scope_coverage"), COVERAGES, "result.scope_coverage")
        if not result_template:
            if template:
                error("result", "real result cannot bind a template contract")
            for key in ("task_id", "run_id"):
                if result.get(key) != contract.get(key):
                    error("result." + key, "must match contract")
            expected_sha = hashlib.sha256(contract_bytes).hexdigest() if contract_bytes is not None else None
            if expected_sha is None or result.get("contract_sha256") != expected_sha:
                error("result.contract_sha256", "must match the exact contract file bytes")
            execution = result.get("execution", {})
            performed = execution.get("performed")
            if not isinstance(performed, bool):
                error("result.execution.performed", "must be boolean")
            if performed:
                for key in ("hostname", "command", "working_directory", "environment_path", "stdout_path", "stderr_path", "receipt_path", "exit_verification_path", "handles_absent_verified_at"):
                    need(execution, key, "result.execution")
                if execution.get("hostname") != contract["resources"].get("expected_hostname"):
                    error("result.execution.hostname", "must match contracted hostname")
                if not isinstance(execution.get("pid"), int) or isinstance(execution.get("pid"), bool) or execution["pid"] <= 0:
                    error("result.execution.pid", "positive actual PID is required")
                if not isinstance(execution.get("exit_code"), int) or isinstance(execution.get("exit_code"), bool):
                    error("result.execution.exit_code", "actual terminal exit code is required")
                start = date(execution.get("started_at"), "result.execution.started_at")
                end = date(execution.get("ended_at"), "result.execution.ended_at")
                if start and frozen and start < frozen:
                    error("result.execution.started_at", "execution preceded contract freezing")
                if start and end and end < start:
                    error("result.execution.ended_at", "end precedes start")
                if effective and end and end > effective:
                    error("result.execution.ended_at", "execution exceeded effective deadline")
                if effective and start and finite_number(timeout) and start + timedelta(seconds=timeout) > effective:
                    error("result.execution.started_at", "contract timeout exceeds the execution window at actual start")
                if execution.get("effective_deadline_at") != timing.get("effective_deadline_at"):
                    error("result.execution.effective_deadline_at", "must preserve the contracted effective deadline")
                if execution.get("exit_code") != 0 and result.get("lifecycle_status") == "succeeded":
                    error("result.lifecycle_status", "nonzero terminal exit cannot support a succeeded run")
            elif result.get("reproduction_level") in {"smoke", "component", "published_output_reanalysis", "independent_inference", "training", "full_paper"}:
                error("result.reproduction_level", "scientific result level requires actual execution")
            if (scope.get("intended_level") == "published_output_reanalysis" or result.get("scope_coverage") == "published_outputs_only" or scope.get("scope_coverage") == "published_outputs_only") and result.get("reproduction_level") in {"independent_inference", "training", "full_paper"}:
                error("result.reproduction_level", "published-output verification cannot claim inference, training or full paper")
            comparisons = result.get("metric_comparisons", [])
            observed = {}
            for index, comparison in enumerate(comparisons):
                label = f"result.metric_comparisons[{index}]"
                target_id = comparison.get("target_id")
                if target_id not in target_map:
                    error(label + ".target_id", "unknown contract endpoint")
                    continue
                if target_id in observed:
                    error(label + ".target_id", "duplicate result endpoint")
                observed[target_id] = comparison
                target = target_map[target_id]
                for key in ("metric", "metric_definition", "statistical_unit", "denominator", "unit"):
                    if comparison.get(key) != target.get(key):
                        error(label + "." + key, "must match contract comparison definition")
                contract_paper = target.get("paper_value")
                paper = comparison.get("paper_value")
                actual = comparison.get("actual_value")
                if paper != contract_paper or (isinstance(paper, bool) != isinstance(contract_paper, bool)):
                    error(label + ".paper_value", "must preserve the contracted paper value")
                enum(comparison.get("comparability"), {"comparable", "not_comparable", "not_evaluated"}, label + ".comparability")
                boolean_endpoint = any(isinstance(value, bool) for value in (contract_paper, paper, actual))
                exact_boolean = all(isinstance(value, bool) for value in (contract_paper, paper, actual)) and target.get("tolerance", {}).get("kind") == "exact"
                known_values = [value for value in (contract_paper, paper, actual) if value is not None]
                if boolean_endpoint and (any(not isinstance(value, bool) for value in known_values) or target.get("tolerance", {}).get("kind") != "exact"):
                    error(label, "Boolean comparison requires exact tolerance and Boolean contract, paper and actual values")
                if exact_boolean:
                    if comparison.get("absolute_difference") is not None and (not finite_number(comparison["absolute_difference"]) or comparison["absolute_difference"] != 0):
                        error(label + ".absolute_difference", "Boolean equality marker may only be zero or null")
                    if comparison.get("relative_difference") is not None:
                        error(label + ".relative_difference", "Boolean comparisons do not have arithmetic relative differences")
                if comparison.get("passed") is True:
                    numeric_endpoint = finite_number(actual) and finite_number(paper) and not boolean_endpoint
                    if comparison.get("comparability") != "comparable" or not (numeric_endpoint or exact_boolean):
                        error(label + ".passed", "pass requires comparable known values")
                    if target.get("tolerance", {}).get("kind") == "not_comparable":
                        error(label + ".passed", "contract endpoint is not comparable")
                    if exact_boolean and actual is not paper:
                        error(label + ".passed", "Boolean values do not exactly match")
                    if numeric_endpoint:
                        delta = abs(actual - paper)
                        tolerance = target.get("tolerance", {})
                        kind = tolerance.get("kind")
                        absolute_pass = finite_number(tolerance.get("absolute")) and delta <= tolerance["absolute"]
                        relative_pass = paper != 0 and finite_number(tolerance.get("relative")) and delta / abs(paper) <= tolerance["relative"]
                        if (kind == "exact" and delta != 0 or kind == "absolute" and not absolute_pass or kind == "relative" and not relative_pass or kind == "both" and not (absolute_pass and relative_pass)):
                            error(label + ".passed", "actual value is outside the frozen tolerance")
                elif comparison.get("passed") not in (False, None):
                    error(label + ".passed", "must be boolean or null")
                if comparison.get("comparability") != "comparable":
                    need(comparison, "reason", label)
            for category in ("datasets", "code", "models", "environment"):
                assessment = result.get("assets_integrity", {}).get(category, {})
                enum(assessment.get("status"), INTEGRITY, "result.assets_integrity." + category + ".status")
            enum(result.get("review", {}).get("status"), {"pending", "accepted", "needs_revision"}, "result.review.status")
            if result.get("lifecycle_status") in {"failed", "blocked", "not_run", "expired", "restricted", "partial"} and not result.get("incomplete_reasons"):
                error("result.incomplete_reasons", "non-successful result requires a specific unfinished reason")
            if result.get("reproduction_level") == "full_paper":
                if result.get("scope_coverage") != "full" or result.get("lifecycle_status") != "succeeded":
                    error("result.reproduction_level", "full paper requires full scope and succeeded status")
                if scope.get("scope_coverage") != "full" or scope.get("intended_level") != "full_paper":
                    error("result.reproduction_level", "a component contract cannot be promoted to a full-paper claim")
                if result.get("review", {}).get("status") != "accepted" or result.get("incomplete_reasons"):
                    error("result.review", "full paper requires accepted review and no unfinished reasons")
                required = [target_id for target_id, target in target_map.items() if target.get("required")]
                if not required or any(observed.get(target_id, {}).get("passed") is not True for target_id in required):
                    error("result.metric_comparisons", "every required endpoint must pass")
                if any(result.get("assets_integrity", {}).get(category, {}).get("status") not in {"complete", "not_applicable"} for category in ("datasets", "code", "models", "environment")):
                    error("result.assets_integrity", "unknown or incomplete assets cannot support full-paper acceptance")
                if any(entry.get("integrity") not in {"complete", "not_applicable"} for category in ("datasets", "code", "models") for entry in assets.get(category, [])):
                    error("contract.assets", "full-paper acceptance requires frozen sufficient assets")
                checks = result.get("review", {}).get("checks", [])
                verified_checks = {check.get("item") for check in checks if isinstance(check, dict) and check.get("passed") is True}
                if not {"identity", "assets", "contract", "execution", "comparison", "claims", "preservation"}.issubset(verified_checks):
                    error("result.review.checks", "full-paper acceptance requires every standard review check to pass")
            accepted = result.get("review", {}).get("status") == "accepted"
            if accepted and contract.get("contract_status") != "frozen":
                error("result.review.status", "accepted record requires a frozen contract")
    if check_files:
        file_scan(contract, contract_directory, "contract")
        if result is not None:
            file_scan(result, result_directory, "result")
    valid = not errors
    # With a result, an expired/blocked task may still be validly archived.
    archive_mode = result is not None
    return {"valid": valid and (archive_mode or template and allow_template or not blockers),
            "valid_for_execution": valid and not blockers and not template and result is None,
            "valid_for_acceptance": valid and accepted and not template and result is not None and not result.get("is_template"),
            "errors": errors, "execution_blockers": blockers,
            "files_checked": checked, "files_skipped": skipped,
            "validation_mode": "archive" if archive_mode else "template_structure" if template else "execution"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--result")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--allow-template", action="store_true")
    parser.add_argument("--at", help="ISO timestamp with timezone; defaults to current UTC")
    args = parser.parse_args(argv)
    try:
        raw = Path(args.contract).read_bytes()
        contract = load_json(args.contract)
        result = load_json(args.result) if args.result else None
        report = validate(contract, result, contract_bytes=raw, at=args.at, allow_template=args.allow_template,
                          check_files=args.check_files, contract_directory=Path(args.contract).parent,
                          result_directory=Path(args.result).parent if args.result else None)
    except (ValueError, TypeError, OSError, KeyError, AttributeError) as exc:
        report = {"valid": False, "valid_for_execution": False, "valid_for_acceptance": False,
                  "errors": [{"field": "input", "message": str(exc)}], "execution_blockers": [], "files_checked": [], "files_skipped": []}
    print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

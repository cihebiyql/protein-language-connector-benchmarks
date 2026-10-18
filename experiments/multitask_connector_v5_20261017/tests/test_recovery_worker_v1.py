"""CPU-only authorization and fresh-process contracts for the recovery adapter."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCOPE = Path(__file__).resolve().parents[1]
if str(SCOPE) not in sys.path:
    sys.path.insert(0, str(SCOPE))
MODULE_SPEC = importlib.util.spec_from_file_location("recovery_worker_v1", SCOPE / "resources" / "recovery_worker_v1.py")
adapter = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(adapter)


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def bound(tmp_path, monkeypatch):
    source = tmp_path / "science.py"
    source.write_text("unchanged = True\n", encoding="utf-8")
    case = {"case_id": adapter.TARGET, "task": "elm", "method": "M3", "backbone": "qwen3_4b_2507", "seed": 17}
    plan = {"phase": "formal", "require_train_canary": True, "gpu_devices": [3, 4, 5],
        "paths": {"scope_root": str(tmp_path), "python": sys.executable}, "cases": [case],
        "source_pins": {str(source): adapter.sha(source)}, "per_case_timeout_seconds": 28800,
        "resource_policy": {"gpu_memory_budget_bytes": {"qwen3_4b_2507": 18 * 1024**3},
            "rss_limit_bytes": 64 * 1024**3, "storage_reserve_bytes": 10 * 1024**3, "expected_host": "node1"}}
    plan_path = tmp_path / "original_plan.json"
    write(plan_path, plan)
    policy = {"schema": "v5-recovery-resource-policy.v1", "case_id": adapter.TARGET,
        "original_plan_path": str(plan_path), "original_plan_sha256": adapter.sha(plan_path),
        "authorized_gpu_devices": [3, 4, 5, 6], "original_gpu_devices": [3, 4, 5],
        "maximum_distinct_owned_gpus": 4, "scope_root": str(tmp_path)}
    policy_path = tmp_path / "policy.json"
    write(policy_path, policy)
    attempt = tmp_path / "runs" / "recovery_retry_v1" / "cases" / adapter.TARGET / "attempt_001"
    attempt.mkdir(parents=True)
    spec_path = attempt / "spec.json"
    spec = {"case": case, "gpu": 6, "cwd": str(tmp_path), "plan_path": str(plan_path),
        "plan_sha256": adapter.sha(plan_path), "recovery_policy_path": str(policy_path),
        "recovery_policy_sha256": adapter.sha(policy_path), "recovery_adapter_sha256": adapter.sha(adapter.ADAPTER),
        "gpu_memory_budget_bytes": 18 * 1024**3, "rss_limit_bytes": 64 * 1024**3,
        "storage_reserve_bytes": 10 * 1024**3, "expected_host": "node1", "timeout_seconds": 28800,
        "command": [sys.executable, str(adapter.ADAPTER), "--mode", "worker", "--spec", str(spec_path)]}
    write(spec_path, spec)
    monkeypatch.setattr(adapter.mt_case, "PHYSICAL_GPUS", (3, 4, 5))
    monkeypatch.setattr(adapter.mt_supervise, "PHYSICAL_GPUS", (3, 4, 5))
    return SimpleNamespace(spec_path=spec_path, spec=spec, plan=plan, plan_path=plan_path,
                           policy=policy, policy_path=policy_path, source=source, attempt=attempt)


@pytest.mark.parametrize("gpu", [3, 4, 5, 6])
def test_exact_four_cards_permitted_and_no_source_mutation(bound, gpu):
    before = {p: adapter.sha(p) for p in SCOPE.joinpath("src").glob("*.py")}
    bound.spec["gpu"] = gpu
    write(bound.spec_path, bound.spec)
    adapter.activate(bound.spec_path)
    assert adapter.mt_case.PHYSICAL_GPUS == adapter.mt_supervise.PHYSICAL_GPUS == (3, 4, 5, 6)
    assert before == {p: adapter.sha(p) for p in before}
    receipt = json.loads((bound.attempt / "recovery_resource_override.json").read_text(encoding="utf-8"))
    assert receipt["execution_entrypoint_changed"] is True
    assert receipt["scientific_source_files_modified"] is False


@pytest.mark.parametrize("gpu", [None, 0, 2, 7, True, "6"])
def test_other_cards_and_implicit_cpu_rejected(bound, gpu):
    bound.spec["gpu"] = gpu
    write(bound.spec_path, bound.spec)
    with pytest.raises(ValueError, match="physical GPU"):
        adapter.validate(bound.spec_path)


@pytest.mark.parametrize("field", ["recovery_policy_sha256", "recovery_adapter_sha256", "plan_sha256"])
def test_pinned_extra_digests_reject_tampering(bound, field):
    bound.spec[field] = "0" * 64
    write(bound.spec_path, bound.spec)
    with pytest.raises(ValueError):
        adapter.validate(bound.spec_path)


def test_frozen_source_tamper_rejected(bound):
    bound.source.write_text("changed = True", encoding="utf-8")
    with pytest.raises(ValueError, match="scientific source"):
        adapter.validate(bound.spec_path)


def test_frozen_case_and_budget_cannot_change(bound):
    bound.spec["case"] = dict(bound.spec["case"], seed=42)
    write(bound.spec_path, bound.spec)
    with pytest.raises(ValueError, match="exact frozen target"):
        adapter.validate(bound.spec_path)
    bound.spec["case"] = bound.plan["cases"][0]
    bound.spec["gpu_memory_budget_bytes"] += 1
    write(bound.spec_path, bound.spec)
    with pytest.raises(ValueError, match="memory and time budgets"):
        adapter.validate(bound.spec_path)


@pytest.mark.parametrize("field,value", [("maximum_distinct_owned_gpus", 5), ("authorized_gpu_devices", [3, 4, 5, 7]),
                                         ("case_id", "other"), ("original_gpu_devices", [0, 1, 2])])
def test_even_repinned_policy_cannot_widen_authority(bound, field, value):
    bound.policy[field] = value
    write(bound.policy_path, bound.policy)
    bound.spec["recovery_policy_sha256"] = adapter.sha(bound.policy_path)
    write(bound.spec_path, bound.spec)
    with pytest.raises(ValueError, match="exactly four GPUs"):
        adapter.validate(bound.spec_path)


def test_attempt_and_execution_command_are_bound(bound):
    bound.spec["command"] = [sys.executable, "other.py"]
    write(bound.spec_path, bound.spec)
    with pytest.raises(ValueError, match="worker command"):
        adapter.validate(bound.spec_path)
    alternate = bound.attempt.parent / "spec.json"
    write(alternate, bound.spec)
    with pytest.raises(ValueError, match="one new directory"):
        adapter.validate(alternate)


def test_supervision_calls_original_without_changing_other_globals(bound, monkeypatch):
    calls = []
    monkeypatch.setattr(adapter.mt_supervise, "supervise", lambda *args: calls.append(args) or 75)
    assert adapter.execute("supervise", bound.spec_path, output=bound.attempt / "supervision") == 75
    assert calls == [(bound.spec_path, bound.attempt / "supervision")]
    with pytest.raises(ValueError, match="Supervision output"):
        adapter.execute("supervise", bound.spec_path, output=bound.attempt / "elsewhere")


def test_fresh_qualification_and_formal_argv_and_receipts(bound, monkeypatch):
    calls = []
    pids = iter([3101, 3102])

    class Process:
        def __init__(self, argv, **kwargs):
            self.pid = next(pids)
            calls.append((argv, kwargs))
        def wait(self):
            return 0

    monkeypatch.setattr(adapter.subprocess, "Popen", Process)
    results = []
    for phase in ("train-canary", "formal"):
        results.append(adapter.run_training_child(bound.spec_path, phase, bound.attempt / phase, bound.attempt / "budget.json"))
    assert results[0]["pid"] != results[1]["pid"]
    for index, phase in enumerate(("train-canary", "formal")):
        argv, kwargs = calls[index]
        assert argv[:4] == [sys.executable, str(adapter.ADAPTER), "--mode", "train-child"]
        assert argv[argv.index("--phase") + 1] == phase
        assert "start_new_session" not in kwargs and "preexec_fn" not in kwargs
        assert results[index]["fresh_process"] and results[index]["inherits_owned_process_group"]
        assert json.loads((bound.attempt / f"{phase}_process.json").read_text(encoding="utf-8"))["recovery_policy_sha256"] == bound.spec["recovery_policy_sha256"]


def test_child_failure_preserved_without_hidden_retry(bound, monkeypatch):
    calls = []
    monkeypatch.setattr(adapter.subprocess, "Popen", lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(pid=5001, wait=lambda: 1))
    with pytest.raises(RuntimeError, match="no implicit retry"):
        adapter.run_training_child(bound.spec_path, "train-canary", bound.attempt / "qualification", bound.attempt / "budget.json")
    assert len(calls) == 1
    assert json.loads((bound.attempt / "train-canary_process.json").read_text(encoding="utf-8"))["exit_code"] == 1


@pytest.mark.parametrize("phase", ["train-canary", "formal"])
def test_child_uses_original_guard_and_train_function(bound, monkeypatch, phase):
    calls = []
    monkeypatch.setattr(adapter.mt_case, "guarded_plan", lambda path, **kwargs: calls.append(("guard", kwargs)) or (bound.plan, bound.spec))
    monkeypatch.setattr(adapter.mt_case, "train_process", lambda *args: calls.append(("train", args)))
    output = bound.attempt / ("qualification" if phase == "train-canary" else "artifacts")
    receipt = bound.attempt / ("qualification_device_budget.json" if phase == "train-canary" else "device_budget.json")
    assert adapter.execute("train-child", bound.spec_path, output=output, phase=phase, receipt=receipt) == 0
    assert calls[0] == ("guard", {"descendant": True})
    assert calls[1][1] == (bound.spec_path, bound.plan, bound.spec, output, phase, receipt)


def test_worker_uses_original_preflight_and_fresh_child_adapter(bound, monkeypatch):
    monkeypatch.setattr(adapter.mt_case, "guarded_plan", lambda path: (bound.plan, bound.spec))
    calls = []
    monkeypatch.setattr(adapter.mt_case, "preflight_then_formal", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert adapter.execute("worker", bound.spec_path) == 0
    assert calls == [((bound.spec_path, bound.plan, bound.spec), {"run_child": adapter.run_training_child})]

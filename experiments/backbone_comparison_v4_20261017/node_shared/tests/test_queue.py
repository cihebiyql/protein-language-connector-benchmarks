"""No-GPU, stdlib-only queue and case orchestration fixtures."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
import signal
from pathlib import Path
import sys
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mm_case import run_case, sha, training_commands, write_new, case_watchdog, expire_owned_group, watchdog_binding
from src.mm_queue import (DEADLINE, Ledger, SCENARIOS, SEEDS, SOURCE_NAMES, VARIANTS, BACKBONES,
                          canonical, digest, exclusive_lock, fixed_cases, run_queue, validate_plan, stop_owned_attempt)


def fixture(root, gpus=(2, 3)):
    project = root / "project"
    scope = project / "experiments/new_scope"
    for name in ("src", "configs", "cache", "views", "resources", "standard", "data"):
        (scope / name).mkdir(parents=True)
    sources = {}
    for name in SOURCE_NAMES:
        path = scope / "src" / name
        path.write_text("# immutable fixture\n")
        sources[str(path)] = sha(path)
    backbone_specs = json.loads((Path(__file__).resolve().parents[1] / "configs/development_protocol.json").read_text())["backbones"]
    for spec in backbone_specs.values():
        spec["revision"] = "d"*40; spec["model_manifest_path"] = "fixture_manifest.json"
    training = {"epochs": 3, "batch_size": 4, "numeric_loss_weight": 1.0}
    config = {"status": "frozen", "deadline": DEADLINE, "seeds": list(SEEDS),
              "scenarios": list(SCENARIOS), "variants": list(VARIANTS),
              "training": training, "budget_signature": digest(training), "backbones": backbone_specs, "backbone_signatures": {b: digest(backbone_specs[b]) for b in BACKBONES}}
    config_path = scope / "configs/frozen.json"
    write_new(config_path, config)
    evaluation = scope / "configs/evaluation.json"
    write_new(evaluation, {"backbones": config["backbones"], "producer_source_sha256": sources[str(scope / "src/mm_train.py")], "evaluator_source_sha256": sources[str(scope / "src/mm_evaluate.py")], "training_budget": {"status": "FROZEN", "config_sha256": sha(config_path),
              "training_budget_sha256": digest(training), "training": training}})
    environment = scope / "resources/environment.json"
    write_new(environment, {"fixture": True})
    for name in ("cache", "views"):
        write_new(scope / name / "manifest.json", {"fixture": True})
    (scope / "standard/validate_reproduction.py").write_text("# standard fixture\n")
    write_new(scope / "standard/task-contract.template.json", {"fixture": True})
    write_new(scope / "standard/result-record.template.json", {"fixture": True})
    paths = {"python": sys.executable, "project_root": str(project), "scope_root": str(scope),
             "config": str(config_path), "cache_root": str(scope / "cache"), "view_root": str(scope / "views"),
             "evaluation_protocol": str(evaluation), "data_root": str(scope / "data"), "standard_root": str(scope / "standard")}
    assets = {str(path): sha(path) for path in (config_path, evaluation, environment,
               scope / "cache/manifest.json", scope / "views/manifest.json",
               scope / "standard/validate_reproduction.py", scope / "standard/task-contract.template.json",
               scope / "standard/result-record.template.json")}
    assignments = []
    for gpu in gpus:
        path = scope / f"resources/ready_gpu{gpu}.json"
        write_new(path, {"readiness": "ready", "gpu": gpu})
        assets[str(path)] = sha(path)
        assignments.append({"gpu": gpu, "host": "fixture-host", "readiness": "ready",
                            "evidence": {"path": str(path), "sha256": sha(path)}})
    plan = {"schema_version": "mm-v4.queue.v1", "status": "FROZEN", "deadline": DEADLINE,
            "paths": paths, "source_pins": sources, "asset_pins": assets,
            "ready_assignments": assignments, "occupied_gpu_devices": [0, 1, 6],
            "gpu_memory_budget_bytes": 20 * 1024**3,
            "timeout_seconds_per_case": 1800, "resource_retry": {"max_attempts": 2, "interval_seconds": 0},
            "contract_request": {"project_root": str(project), "scope_root": str(scope),
                "environment_lock": str(environment), "environment_path": "/fixture/env",
                "code_paths": list(sources), "objective": "Fixed controlled scalar benchmark",
                "resources": {"host": "fixture-host", "cpu_threads": 2, "memory_limit_bytes": 1024**3,
                              "gpu_memory_budget_bytes": 20 * 1024**3}}}
    plan_path = scope / "configs/queue.json"
    write_new(plan_path, plan)
    return plan_path, plan, scope


class FakeRuntime:
    def __init__(self, plan, busy_first=False, fail_case=None):
        self.plan, self.busy_first, self.fail_case = plan, busy_first, fail_case
        self.created, self.ran, self.active = [], [], set()
        self.lock = threading.Lock()
        self.overlap = False

    def create(self, request, standard, output):
        with self.lock:
            self.created.append(deepcopy(request))
            if self.busy_first and len(self.created) == 1:
                return None
        write_new(output, {"execution": {"command": request["command"]}})
        return {"fixture": True}

    def run(self, contract, validator, output, gpu=None):
        with self.lock:
            if gpu in self.active:
                self.overlap = True
            self.active.add(gpu)
        try:
            args = json.loads(Path(contract).read_text())["execution"]["command"]
            value = lambda flag: args[args.index(flag) + 1]
            case = (value("--scenario"), value("--variant"), int(value("--seed")), value("--backbone"))
            with self.lock:
                self.ran.append(case)
            time.sleep(.001)
            output = Path(output)
            output.mkdir()
            succeeded = case != self.fail_case
            write_new(output / "receipt.json", {"status": "succeeded" if succeeded else "failed",
                      "exit_code": 0 if succeeded else 1, "remaining_group_members": [], "contract_sha256": sha(contract)})
            if succeeded:
                root = Path(value("--output"))
                root.mkdir()
                write_new(root / "result.json", {"status": "complete", "formal_eligible": True,
                          "scenario": case[0], "arm": case[1], "seed": case[2], "backbone": case[3],
                          "n_expected_test": 152 if case[0] == SCENARIOS[0] else 442,
                          "LM": {"n_total": 152 if case[0] == SCENARIOS[0] else 442},
                          "numeric": {"n_total": 152 if case[0] == SCENARIOS[0] else 442,
                              "n_finite": 152 if case[0] == SCENARIOS[0] else 442, "n_failed": 0,
                              "coverage": 1.0, "status": "defined", "full_population": {"n": 152 if case[0] == SCENARIOS[0] else 442}},
                          "protocol_sha256": self.plan["asset_pins"][self.plan["paths"]["config"]],
                          "evaluation_protocol_sha256": self.plan["asset_pins"][self.plan["paths"]["evaluation_protocol"]]})
            return 0 if succeeded else 2
        finally:
            with self.lock:
                self.active.remove(gpu)

    def archive(self, contract, supervision, artifact_roots, standard, output, *, completion_spec):
        """Explicit MOCK: not a canonical validator or a scientific acceptance test."""
        output = Path(output)
        output.mkdir()
        write_new(output / "terminal_validation.json", {"valid": True, "valid_for_acceptance": True,
                  "files_skipped": [], "fixture_only": True})
        write_new(output / "result-record.json", {"review": {"status": "accepted"}, "lifecycle_status": "succeeded",
                  "contract_sha256": sha(contract), "archive_provenance": {"completion_spec": completion_spec}})
        receipt = {"status": "accepted", "contract_sha256": sha(contract),
                   "record_sha256": sha(output / "result-record.json"),
                   "validation_sha256": sha(output / "terminal_validation.json")}
        write_new(output / "archive_receipt.json", receipt)
        return receipt


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.plan_path, self.plan, self.scope = fixture(Path(self.temporary.name))
        self.output = self.scope / "queue"

    def tearDown(self):
        self.temporary.cleanup()

    def run_fixture(self, runtime, **kwargs):
        return run_queue(self.plan_path, self.output, create_fn=runtime.create, run_fn=runtime.run,
                         archive_fn=runtime.archive,
                         members_fn=lambda _: [], check_hostname=False,
                         now_fn=lambda: datetime.fromisoformat(DEADLINE) - timedelta(hours=2), **kwargs)

    def test_fixed_order_all24_seed_scenario_backbone_arm(self):
        cases = fixed_cases()
        self.assertEqual(len(cases), 24)
        self.assertEqual(len({c['case_id'] for c in cases}), 24)
        self.assertEqual([(c['seed'],c['scenario'],c['backbone'],c['variant']) for c in cases],
            [(seed,sc,b,a) for seed in SEEDS for sc in SCENARIOS for b in BACKBONES for a in VARIANTS])

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required; execute unchanged on node1")
    def test_queue_exclusive_gpu_and_complete_ledger(self):
        runtime = FakeRuntime(self.plan)
        result = self.run_fixture(runtime)
        self.assertEqual(result["counts"], {"not_started": 0, "running": 0, "succeeded": 24, "failed": 0})
        self.assertEqual(len(runtime.ran), 24)
        self.assertFalse(runtime.overlap)
        ledger = Ledger(self.output / "queue_events.jsonl", sha(self.plan_path))
        self.assertEqual(len(ledger.states), 24)
        self.assertEqual(ledger.events[24]["aggregate_cpu_threads"], 4)
        self.assertTrue(all(request["resources"]["gpu"] in (2, 3) for request in runtime.created))

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required; execute unchanged on node1")
    def test_resource_busy_preserves_attempt_without_duplicate_training(self):
        runtime = FakeRuntime(self.plan, busy_first=True)
        result = self.run_fixture(runtime)
        self.assertEqual(result["counts"]["succeeded"], 24)
        self.assertEqual(len(runtime.created), 25)
        self.assertEqual(len(runtime.ran), 24)
        self.assertEqual(len(list(self.output.glob("cases/*/attempt_*"))), 25)

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required; execute unchanged on node1")
    def test_independent_failure_keeps_denominator_and_resume_skips_all_terminal(self):
        failed = (SCENARIOS[0], "M0", 17, BACKBONES[0])
        runtime = FakeRuntime(self.plan, fail_case=failed)
        result = self.run_fixture(runtime)
        self.assertEqual(result["counts"]["failed"], 1)
        self.assertEqual(result["counts"]["succeeded"], 23)
        fresh = FakeRuntime(self.plan)
        resumed = self.run_fixture(fresh, resume=True)
        self.assertEqual(resumed["counts"], result["counts"])
        self.assertFalse(fresh.created)

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required; execute unchanged on node1")
    def test_no_start_after_deadline_preserves24_not_started(self):
        runtime = FakeRuntime(self.plan)
        result = run_queue(self.plan_path, self.output, create_fn=runtime.create, run_fn=runtime.run,
                           members_fn=lambda _: [], check_hostname=False,
                           now_fn=lambda: datetime.fromisoformat(DEADLINE) + timedelta(seconds=1))
        self.assertEqual(result["counts"]["not_started"], 24)
        self.assertFalse(runtime.created)

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required")
    def test_later_ready_candidate_claims_cases_only_after_fresh_gate(self):
        self.plan['ready_assignments'] = self.plan['ready_assignments'][:1]
        self.plan['occupied_gpu_devices'] = [0,1,3,6]
        self.plan['candidate_gpu_devices'] = [2,3]
        self.plan['candidate_probe_interval_seconds'] = .001
        self.plan_path.write_text(json.dumps(self.plan))
        calls = {2:0,3:0}
        def observe(host,gpu,rss,root):
            calls[gpu]+=1
            return {'readiness':'ready' if gpu == 3 and calls[gpu] > 1 else 'busy',
                    'gpu':gpu,'host':host,'reasons':[] if gpu == 3 and calls[gpu] > 1 else ['fixture_external_compute']}
        runtime = FakeRuntime(self.plan)
        result = self.run_fixture(runtime,probe_fn=observe)
        self.assertEqual(result['counts']['succeeded'],24)
        self.assertEqual({x['resources']['gpu'] for x in runtime.created},{3})
        self.assertGreaterEqual(calls[3],2)
        ledger=Ledger(self.output/'queue_events.jsonl',sha(self.plan_path))
        first_running=next(i for i,e in enumerate(ledger.events) if e['status']=='running')
        preceding=ledger.events[:first_running]
        self.assertTrue(any(e.get('event')=='candidate_resource_observation' and e.get('gpu')==3 and e.get('readiness')=='busy' for e in preceding))
        self.assertTrue(any(e.get('event')=='candidate_resource_observation' and e.get('gpu')==3 and e.get('readiness')=='ready' for e in preceding))
        self.assertFalse(any(e.get('case_id') and e.get('gpu')==2 for e in ledger.events))

    def test_ready_duplicate_and_unfrozen_rejected(self):
        for mutate in (lambda p: p.update(status="draft"),
                       lambda p: p["ready_assignments"][0].update(readiness="busy"),
                       lambda p: p["ready_assignments"].append(deepcopy(p["ready_assignments"][0]))):
            invalid = deepcopy(self.plan)
            mutate(invalid)
            with self.assertRaises(ValueError):
                validate_plan(invalid)

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required; execute unchanged on node1")
    def test_exclusive_lock_refuses_second_owner(self):
        with exclusive_lock(self.scope / "same.lock"):
            with self.assertRaises(RuntimeError):
                with exclusive_lock(self.scope / "same.lock"):
                    pass

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required; execute unchanged on node1")
    def test_existing_queue_never_overwritten_and_changed_plan_not_resumable(self):
        self.run_fixture(FakeRuntime(self.plan))
        with self.assertRaises(FileExistsError):
            self.run_fixture(FakeRuntime(self.plan))
        self.plan_path.write_text(self.plan_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "same exact frozen plan"):
            self.run_fixture(FakeRuntime(self.plan), resume=True)

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required; execute unchanged on node1")
    def test_pin_drift_halts_common_execution(self):
        runtime = FakeRuntime(self.plan)
        original = runtime.run
        def mutate(*args, **kwargs):
            code = original(*args, **kwargs)
            Path(next(iter(self.plan["source_pins"]))).write_text("# changed\n")
            return code
        runtime.run = mutate
        result = self.run_fixture(runtime)
        self.assertTrue(result["halted"])
        self.assertLessEqual(len(runtime.ran), 2)
        self.assertEqual(sum(result["counts"].values()), 24)

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required; execute unchanged on node1")
    def test_ledger_tamper_fails_closed(self):
        self.run_fixture(FakeRuntime(self.plan))
        path = self.output / "queue_events.jsonl"
        text = path.read_text().replace('"status":"succeeded"', '"status":"failed"', 1)
        path.write_text(text)
        with self.assertRaisesRegex(ValueError, "Ledger integrity"):
            Ledger(path, sha(self.plan_path))

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required; execute unchanged on node1")
    def test_case_composes_fit_evaluate_and_keeps_same_group(self):
        calls = []
        def execute(command, **kwargs):
            self.assertEqual(set(kwargs), {"cwd", "check"})
            calls.append(command)
            self.assertEqual(command[command.index("--device") + 1], "cuda:0")
            write_new(command[command.index("--receipt") + 1], {
                "allocator_budget_bytes": self.plan["gpu_memory_budget_bytes"], "visible_devices": "2",
                "module": "src.mm_train", "peak_allocated_bytes": 123, "peak_reserved_bytes": 456})
            return SimpleNamespace(returncode=0)
        def evaluator(*_args):
            return {"status": "complete", "formal_eligible": True, "scenario": SCENARIOS[0], "arm": "M0", "seed": 17, "backbone": BACKBONES[0]}
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}):
            code = run_case(self.plan, SCENARIOS[0], "M0", 17, self.scope / "case",
                            execute=execute, evaluate=evaluator, backbone=BACKBONES[0])
        self.assertEqual(code, 0)
        self.assertEqual([cmd[cmd.index("--phase") + 1] for cmd in calls], ["fit", "evaluate"])
        self.assertTrue(all(cmd[2] == "src.mm_worker_limit" for cmd in calls))
        self.assertTrue(all(cmd[cmd.index("--module") + 1] == "src.mm_train" for cmd in calls))
        self.assertIn(str(self.scope / "case/fit/selected.pt"), calls[1])
        self.assertEqual(json.loads((self.scope / "case/case_request.json").read_text())["pgid"], os.getpgrp())

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required; execute unchanged on node1")
    def test_case_failure_does_not_evaluate_or_overwrite(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}):
            with self.assertRaises(RuntimeError):
                run_case(self.plan, SCENARIOS[0], "M0", 17, self.scope / "failedcase",
                         execute=lambda *a, **k: SimpleNamespace(returncode=1),
                         evaluate=lambda *a: self.fail("Must not evaluate failed fit"), backbone=BACKBONES[0])
            with self.assertRaises(FileExistsError):
                run_case(self.plan, SCENARIOS[0], "M0", 17, self.scope / "failedcase")
        self.assertTrue((self.scope / "failedcase/case_failed.json").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required; execute unchanged on node1")
    def test_resource_busy_is_bounded_and_exhaustion_is_not_training_failure(self):
        runtime = FakeRuntime(self.plan)
        def busy(request, *_args):
            runtime.created.append(request)
            return None
        runtime.create = busy
        result = self.run_fixture(runtime)
        self.assertEqual(result["counts"], {"not_started": 24, "running": 0, "succeeded": 0, "failed": 0})
        self.assertEqual(len(runtime.created), 48)
        self.assertFalse(runtime.ran)

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required; execute unchanged on node1")
    def test_resume_recovers_accepted_receipt_without_rerun(self):
        self.run_fixture(FakeRuntime(self.plan))
        ledger = Ledger(self.output / "queue_events.jsonl", sha(self.plan_path))
        first = fixed_cases()[0]["case_id"]
        ledger.append(first, "running", attempt=ledger.states[first]["attempt"], reason="fixture_interrupted_after_receipt")
        runtime = FakeRuntime(self.plan)
        result = self.run_fixture(runtime, resume=True)
        self.assertEqual(result["counts"]["succeeded"], 24)
        self.assertFalse(runtime.created)

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required; execute unchanged on node1")
    def test_resume_refuses_live_group(self):
        self.run_fixture(FakeRuntime(self.plan))
        ledger = Ledger(self.output / "queue_events.jsonl", sha(self.plan_path))
        first = fixed_cases()[0]["case_id"]
        relative = ledger.states[first]["attempt"]
        write_new(self.output / relative / "supervision/started.json", {"pgid": 999999})
        ledger.append(first, "running", attempt=relative)
        runtime = FakeRuntime(self.plan)
        with self.assertRaisesRegex(RuntimeError, "still-active"):
            run_queue(self.plan_path, self.output, resume=True, create_fn=runtime.create, run_fn=runtime.run,
                      members_fn=lambda _: [999999], check_hostname=False,
                      now_fn=lambda: datetime.fromisoformat(DEADLINE) - timedelta(hours=2))
        self.assertFalse(runtime.created)

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required; execute unchanged on node1")
    def test_signal_cleanup_only_accepts_exact_owned_process_signature(self):
        attempt = self.scope / "signal_attempt"
        (attempt / "contract").mkdir(parents=True)
        (attempt / "supervision").mkdir()
        command = [sys.executable, "-c", "import time; time.sleep(30)"]
        child = subprocess.Popen(command, start_new_session=True)
        try:
            path = attempt / "contract/contract.json"
            write_new(path, {"execution": {"command": command}})
            write_new(attempt / "supervision/started.json", {"pid": child.pid, "pgid": child.pid, "contract_sha256": sha(path)})
            groups = []
            result = stop_owned_attempt(attempt, terminate_fn=groups.append)
            self.assertEqual(result, "terminated_bound_owned_group")
            self.assertEqual(groups, [child.pid])
            changed = {"execution": {"command": [sys.executable, "-c", "different"]}}
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(RuntimeError, "unbound"):
                stop_owned_attempt(attempt, terminate_fn=lambda _: self.fail("Foreign group must not be killed"))
        finally:
            child.terminate()
            child.wait(timeout=5)

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required; execute unchanged on node1")
    def test_archive_claim_cannot_override_canonical_terminal_false(self):
        runtime = FakeRuntime(self.plan)
        original = runtime.archive
        rejected = []
        def reject_one(*args, **kwargs):
            receipt = original(*args, **kwargs)
            if not rejected:
                rejected.append(True)
                output = Path(args[4])
                validation = output / "terminal_validation.json"
                validation.write_text(json.dumps({"valid": False, "valid_for_acceptance": False, "files_skipped": []}))
                receipt["validation_sha256"] = sha(validation)
                (output / "archive_receipt.json").write_text(json.dumps(receipt))
            return receipt  # Still self-claims accepted: terminal false must win.
        runtime.archive = reject_one
        result = self.run_fixture(runtime)
        self.assertEqual(result["counts"]["failed"], 1)
        self.assertEqual(result["counts"]["succeeded"], 23)
        self.assertEqual(len(list(self.output.glob("cases/*/attempt_*/archive/terminal_validation.json"))), 24)

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required; execute unchanged on node1")
    def test_numeric_coverage_and_full_test_denominator_required_before_archive(self):
        runtime = FakeRuntime(self.plan)
        original = runtime.run
        changed = []
        def corrupt_one(*args, **kwargs):
            code = original(*args, **kwargs)
            if not changed:
                changed.append(True)
                result_path = Path(args[2]).parent / "case/result.json"
                result = json.loads(result_path.read_text())
                result["numeric"]["coverage"] = .5
                result_path.write_text(json.dumps(result))
            return code
        runtime.run = corrupt_one
        result = self.run_fixture(runtime)
        self.assertEqual(result["counts"]["failed"], 1)
        self.assertEqual(result["counts"]["succeeded"], 23)
        self.assertEqual(len(list(self.output.glob("cases/*/attempt_*/archive"))), 23)

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required")
    def test_other_backbone_result_cannot_fill_registered_case(self):
        runtime = FakeRuntime(self.plan)
        original = runtime.run
        changed = []
        def corrupt(*args, **kwargs):
            code = original(*args, **kwargs)
            if not changed:
                changed.append(True)
                path=Path(args[2]).parent/'case/result.json'
                result=json.loads(path.read_text())
                result['backbone']=next(b for b in BACKBONES if b!=result['backbone'])
                path.write_text(json.dumps(result))
            return code
        runtime.run=corrupt
        result=self.run_fixture(runtime)
        self.assertEqual(result['counts']['failed'],1)
        self.assertEqual(result['counts']['succeeded'],23)

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required; execute unchanged on node1")
    def test_case_watchdog_timer_cancel_and_previous_handler_restoration(self):
        binding = {"pid": 999999, "pgid": 999999, "parent_pgid_at_install": 123,
                   "timer_seconds": 10., "absolute_deadline": (datetime.now(timezone.utc) + timedelta(seconds=20)).isoformat(),
                   "monotonic_at_budget_calculation": time.monotonic()}
        prior = object()
        with patch("src.mm_case.watchdog_binding", return_value=binding), \
                patch("src.mm_case.signal.getsignal", return_value=prior), \
                patch("src.mm_case.signal.getitimer", return_value=(0., 0.)), \
                patch("src.mm_case.signal.signal") as handlers, \
                patch("src.mm_case.signal.setitimer") as timers:
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                with case_watchdog("fixture") as observed:
                    self.assertGreater(observed["timer_seconds"], 0.)
                    self.assertLess(observed["timer_seconds"], 10.)
                    raise RuntimeError("fixture failure")
            self.assertEqual(timers.call_args_list[-1].args, (signal.ITIMER_REAL, 0))
            self.assertEqual(handlers.call_args_list[-1].args, (signal.SIGALRM, prior))

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required; execute unchanged on node1")
    def test_watchdog_refuses_foreign_group_and_only_targets_bound_self(self):
        binding = {"pid": os.getpid(), "pgid": os.getpid(), "parent_pgid_at_install": os.getpid() + 10}
        sent = []
        with patch("src.mm_case.os.getpgrp", return_value=os.getpid()), \
                patch("src.mm_case.os.getsid", return_value=os.getpid()):
            expire_owned_group(binding, killpg=lambda *args: sent.append(args))
        self.assertEqual(sent, [(os.getpid(), signal.SIGKILL)])
        with patch("src.mm_case.os.getpgrp", return_value=os.getpid() + 1):
            with self.assertRaisesRegex(RuntimeError, "foreign"):
                expire_owned_group(binding, killpg=lambda *args: self.fail("Never signal external group"))

    @unittest.skipUnless(os.name == "posix", "POSIX runtime required; execute unchanged on node1")
    def test_independent_case_watchdog_kills_own_cpu_group_without_controller(self):
        attempt = self.scope / "watchdog_attempt"
        (attempt / "contract").mkdir(parents=True)
        (attempt / "supervision").mkdir()
        script = attempt / "watchdog_worker.py"
        child_pid_path = attempt / "child_pid"
        actual_scope = Path(__file__).resolve().parents[1]
        script.write_text("import sys,time,subprocess\nfrom pathlib import Path\n"
            + "sys.path.insert(0," + repr(str(actual_scope)) + ")\n"
            + "from src.mm_case import case_watchdog\n"
            + "with case_watchdog(sys.argv[1]):\n"
            + "    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
            + "    Path(" + repr(str(child_pid_path)) + ").write_text(str(child.pid))\n"
            + "    time.sleep(30)\n")
        contract_path = attempt / "contract/contract.json"
        command = [sys.executable, str(script), str(contract_path)]
        now = datetime.now(timezone.utc)
        write_new(contract_path, {"is_template": False, "contract_status": "frozen", "frozen_at": now.isoformat(),
            "execution": {"command": command, "working_directory": str(self.scope), "timeout_seconds": .6},
            "timing": {"effective_deadline_at": (now + timedelta(seconds=30)).isoformat()}})
        process = subprocess.Popen(command, cwd=self.scope, start_new_session=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            write_new(attempt / "supervision/started.json", {"pid": process.pid, "pgid": process.pid,
                "started_at": datetime.now(timezone.utc).isoformat(), "contract_sha256": sha(contract_path)})
            out, error = process.communicate(timeout=5)
            self.assertEqual(process.returncode, -signal.SIGKILL, error.decode())
            self.assertTrue(child_pid_path.exists())
            child_stat = Path("/proc") / child_pid_path.read_text() / "stat"
            if child_stat.exists():
                self.assertEqual(child_stat.read_text().rsplit(")", 1)[1].split()[0], "Z")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()

"""CPU-only tests: no model imports, GPU allocation or existing job changes."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from src import mt_queue as queue
from src import mt_supervise as supervisor
from src import mt_case as case_runtime
from src.mt_case import exact_ids, sha


class RuntimeTests(unittest.TestCase):
    def test_no_gpu_utilization_or_cpu_idle_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Path, "read_text", return_value="MemAvailable: 200000000 kB\n"), \
                 patch.object(os, "getloadavg", return_value=(999, 999, 999), create=True), \
                 patch.object(os, "statvfs", return_value=type("Disk", (), {"f_bavail": 100, "f_frsize": supervisor.GIB})(), create=True), \
                 patch.object(subprocess, "check_output", return_value="24576,22000,100") as call:
                receipt = supervisor.probe(root=directory, gpu=3, rss_limit_bytes=64 * supervisor.GIB,
                    gpu_memory_budget_bytes=18 * supervisor.GIB)
            self.assertTrue(receipt["ready"])
            self.assertEqual(call.call_count, 1)  # No exclusion of existing compute PIDs.

    def test_actual_memory_threshold_and_physical_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Path, "read_text", return_value="MemAvailable: 200000000 kB\n"), \
                 patch.object(os, "getloadavg", return_value=(1, 1, 1), create=True), \
                 patch.object(os, "statvfs", return_value=type("Disk", (), {"f_bavail": 100, "f_frsize": supervisor.GIB})(), create=True), \
                 patch.object(subprocess, "check_output", return_value="24576,10000,0"):
                receipt = supervisor.probe(root=directory, gpu=4, rss_limit_bytes=64 * supervisor.GIB,
                    gpu_memory_budget_bytes=18 * supervisor.GIB)
            self.assertEqual(receipt["reasons"], ["insufficient_gpu_memory"])
            with self.assertRaises(ValueError):
                supervisor.probe(root=directory, gpu=0, rss_limit_bytes=1)

    def test_ledger_preserves_all_attempt_events(self):
        with tempfile.TemporaryDirectory() as directory:
            queue.append_event(directory, {"case_id": "elm_ESM_17", "status": "pending"})
            queue.append_event(directory, {"case_id": "elm_ESM_17", "status": "failed", "attempt": "attempt_001"})
            self.assertEqual([x["status"] for x in queue.events(directory)], ["pending", "failed"])

    def test_exact_prediction_coverage_including_duplicate_rejection(self):
        exact_ids([{"sample_id": "a"}, {"sample_id": "b"}], ["a", "b"], "test")
        for rows in ([{"sample_id": "a"}], [{"sample_id": "a"}, {"sample_id": "a"}],
                     [{"sample_id": "a"}, {"sample_id": "c"}]):
            with self.assertRaises(ValueError):
                exact_ids(rows, ["a", "b"], "test")

    def test_plan_requires_exact_45_and_no_fourth_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "config.json"
            source.write_text("{}")
            cases = [{"case_id": f"{task}_{method}_{backbone}_{seed}", "task": task,
                "method": method, "backbone": backbone, "seed": seed}
                for task in ("meltome", "elm", "rt") for seed in (17, 42, 2026)
                for method, backbone in (("ESM", "none"), ("M0", "qwen25_1p5b"), ("M3", "qwen25_1p5b"),
                                         ("M0", "qwen3_4b_2507"), ("M3", "qwen3_4b_2507"))]
            plan = {"status": "frozen", "gpu_devices": [3, 4, 5], "cases": cases,
                "paths": {key: str(source.resolve()) for key in ("python", "scope_root", "data_root", "cache_root", "config")},
                "per_case_timeout_seconds": 3600, "source_pins": {str(source.resolve()): sha(source)},
                "asset_pins": {str(source.resolve()): sha(source)},
                "resource_policy": {"gpu_memory_budget_bytes": {"qwen25_1p5b": 8 * supervisor.GIB, "qwen3_4b_2507": 18 * supervisor.GIB}}}
            queue.validate_plan(plan)
            plan["gpu_devices"].append(6)
            with self.assertRaises(ValueError):
                queue.validate_plan(plan)
            plan["gpu_devices"].pop()
            plan["cases"].pop()
            with self.assertRaises(ValueError):
                queue.validate_plan(plan)


class QualificationTests(unittest.TestCase):
    def fixture(self, root):
        def save(name, value):
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(json.dumps(value), encoding="utf-8")
        save("config.json", {})
        plan = {"scope_id": "test-v5", "paths": {"config": str(root / "config.json"), "data_root": str(root)},
                "source_pins": {str(root / "config.json"): sha(root / "config.json")},
                "asset_pins": {str(root / "config.json"): sha(root / "config.json")}}
        save("plan.json", plan)
        case = {"case_id": "meltome_ESM_17", "task": "meltome", "method": "ESM", "backbone": "none", "seed": 17}
        spec = {"case": case, "plan_path": str(root / "plan.json"), "plan_sha256": sha(root / "plan.json"),
                "gpu": None, "gpu_memory_budget_bytes": 0}
        save("spec.json", spec)
        save("qualification_device_budget.json", {"gpu": None, "gpu_memory_budget_bytes": 0,
             "phase": "train-canary", "plan_sha256": spec["plan_sha256"]})
        save("qualification/checkpoint.pt", {"dummy": "checkpoint hashed without importing model weights"})
        save("qualification/training_history.json", [{"optimizer_steps": 4}])
        save("qualification/selection.json", {"selection": "none_TRAIN_only_canary"})
        train = [{"sample_id": str(i)} for i in range(8)]
        (root / "qualification/train_readout_predictions.jsonl").write_text("\n".join(
            json.dumps({"sample_id": str(i), "prediction": 1e10}) for i in range(8)), encoding="utf-8")
        (root / "qualification/progress.jsonl").write_text("\n".join(
            json.dumps({"event": "optimizer_step", "optimizer_step": i, "examples_seen": 4*i}) for i in range(1, 5)), encoding="utf-8")
        manifest = {**case, "scope_id": plan["scope_id"], "config_sha256": sha(root / "config.json"),
                    "status": "complete", "phase": "train-canary", "test_labels_used_for_fit": False,
                    "training": {"epochs_completed": 0, "optimizer_steps": 4, "examples_seen": 16},
                    "canary_qualification": {"train_rows_scored": 8, "development_only": True,
                        "validation_opened": False, "calibration_opened": False, "test_opened": False},
                    "runtime": {}, "elapsed_seconds": 1.2, "artifacts": {
                        "checkpoint": {"path": "checkpoint.pt", "sha256": sha(root / "qualification/checkpoint.pt")}}}
        save("qualification/run_manifest.json", manifest)
        return plan, spec, case, train, save

    def test_train_qualification_is_technical_not_accuracy_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, spec, case, train, save = self.fixture(root)
            with patch("src.mt_data.load_task", return_value=train):
                receipt = case_runtime.qualification_acceptance(plan, root, case)
            self.assertFalse(receipt["accuracy_used_for_qualification"])
            self.assertEqual(receipt["training_examples"], 16)
            self.assertTrue(receipt["excluded_from_formal_fit_counts"])
            self.assertIn("checkpoint.pt", receipt["artifact_sha256"])

    def test_missing_or_forged_qualification_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, spec, case, train, save = self.fixture(root)
            (root / "qualification/checkpoint.pt").write_text("modified checkpoint")
            with patch("src.mt_data.load_task", return_value=train), self.assertRaisesRegex(ValueError, "artifact SHA"):
                case_runtime.qualification_acceptance(plan, root, case)
            (root / "qualification/run_manifest.json").unlink()
            with self.assertRaises(FileNotFoundError):
                case_runtime.qualification_acceptance(plan, root, case)

    def test_evaluation_split_read_or_incomplete_training_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, spec, case, train, save = self.fixture(root)
            manifest = json.loads((root / "qualification/run_manifest.json").read_text())
            manifest["canary_qualification"]["test_opened"] = True
            save("qualification/run_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "evaluation split"):
                case_runtime.qualification_acceptance(plan, root, case)
            manifest["canary_qualification"]["test_opened"] = False
            manifest["training"]["optimizer_steps"] = 3
            save("qualification/run_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "four updates"):
                case_runtime.qualification_acceptance(plan, root, case)

    def test_canary_and_formal_use_fresh_processes_and_failure_stops_formal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, spec, case, train, save = self.fixture(root)
            phases = []
            def dummy_child(spec_path, phase, output, receipt):
                script = ("import json,os,random,sys;random.seed(17);"
                          "result={'pid':os.getpid(),'phase':sys.argv[2], 'initial':random.random(),"
                          "'inherited_canary':os.environ.get('V5_DUMMY_CANARY_STATE')};"
                          "os.environ['V5_DUMMY_CANARY_STATE']='changed';"
                          "open(sys.argv[1],'w').write(json.dumps(result))")
                path = root / f"{phase}_dummy.json"
                subprocess.run([sys.executable, "-c", script, str(path), phase], check=True)
                evidence = json.loads(path.read_text())
                phases.append(evidence)
                record = {"pid": evidence["pid"], "exit_code": 0}
                save(f"{phase}_process.json", record)
                return record
            case_runtime.preflight_then_formal(root / "spec.json", plan, spec,
                run_child=dummy_child, qualify=lambda *_: {"technical": True})
            self.assertEqual([r["phase"] for r in phases], ["train-canary", "formal"])
            self.assertNotEqual(phases[0]["pid"], phases[1]["pid"])
            self.assertEqual(phases[0]["initial"], phases[1]["initial"])
            self.assertIsNone(phases[1]["inherited_canary"])
            phases.clear()
            def reject(*_):
                raise ValueError("technical qualification failure")
            with self.assertRaisesRegex(ValueError, "technical qualification failure"):
                case_runtime.preflight_then_formal(root / "spec.json", plan, spec,
                    run_child=dummy_child, qualify=reject)
            self.assertEqual([r["phase"] for r in phases], ["train-canary"])


@unittest.skipUnless(os.name == "posix" and Path("/proc").exists(), "Linux owned-session integration")
class LinuxSupervisionTests(unittest.TestCase):
    def test_scope_lease_excludes_second_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.lock"
            with queue.lease(path):
                with self.assertRaises(RuntimeError):
                    with queue.lease(path):
                        self.fail("second owner acquired the same scope")
            with queue.lease(path):
                pass

    def execute(self, script, timeout=10, rss=supervisor.GIB):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {"gpu": None, "rss_limit_bytes": rss, "gpu_memory_budget_bytes": 0,
                "timeout_seconds": timeout, "storage_reserve_bytes": 0,
                "expected_host": socket.gethostname(), "command": [sys.executable, "-c", script],
                "cwd": str(Path(__file__).resolve().parents[1])}
            (root / "spec.json").write_text(json.dumps(spec))
            code = supervisor.supervise(root / "spec.json", root / "supervision")
            receipt = json.loads((root / "supervision/finished.json").read_text())
            self.assertFalse(receipt["remaining_group_members"])
            self.assertEqual(receipt["identity"]["pid"], receipt["identity"]["pgid"])
            return code, receipt

    def test_success_and_nonzero_exit_are_distinct(self):
        code, receipt = self.execute("print('dummy CPU success')")
        self.assertEqual((code, receipt["exit_code"], receipt["status"]), (0, 0, "completed"))
        code, receipt = self.execute("raise SystemExit(7)")
        self.assertEqual((code, receipt["exit_code"], receipt["status"]), (1, 7, "failed"))

    def test_independent_timeout_cleans_grandchild(self):
        code, receipt = self.execute("import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);time.sleep(30)", timeout=.5)
        self.assertEqual(code, 1)
        self.assertEqual(receipt["guard"]["reason"], "case_timeout")

    def test_rss_guard(self):
        code, receipt = self.execute("import time; blob=bytearray(32*1024**2);time.sleep(30)", rss=16 * 1024 ** 2)
        self.assertEqual(code, 1)
        self.assertEqual(receipt["guard"]["reason"], "rss_limit_exceeded")

    def test_independent_guard_survives_supervisor_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {"gpu": None, "rss_limit_bytes": supervisor.GIB, "gpu_memory_budget_bytes": 0,
                "timeout_seconds": 1.5, "storage_reserve_bytes": 0,
                "command": [sys.executable, "-c", "import time;time.sleep(30)"],
                "cwd": str(Path(__file__).resolve().parents[1])}
            (root / "spec.json").write_text(json.dumps(spec))
            process = subprocess.Popen([sys.executable, "-m", "src.mt_supervise", "--spec", str(root / "spec.json"),
                "--output", str(root / "supervision")], cwd=spec["cwd"], start_new_session=True)
            until = time.monotonic() + 8
            while not (root / "supervision/guard_started.json").exists() and time.monotonic() < until:
                time.sleep(.02)
            self.assertTrue((root / "supervision/guard_started.json").exists())
            identity = json.loads((root / "supervision/started.json").read_text())["identity"]
            process.kill()  # This exact Popen child only, not its worker/group.
            process.wait(timeout=5)
            until = time.monotonic() + 8
            while not (root / "supervision/guard_result.json").exists() and time.monotonic() < until:
                time.sleep(.05)
            receipt = json.loads((root / "supervision/guard_result.json").read_text())
            self.assertEqual(receipt["reason"], "case_timeout")
            self.assertFalse(supervisor.group_members(identity))


if __name__ == "__main__":
    unittest.main()

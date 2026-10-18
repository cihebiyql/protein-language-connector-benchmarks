"""CPU-only preservation and authorization tests; no model, SSH, Git or training."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCOPE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("reconcile_recovery_v1", SCOPE / "resources/reconcile_recovery_v1.py")
r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r)


class RecoveryViewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.scope = Path(self.temp.name).resolve()
        self.resources = self.scope / "resources"
        self.resources.mkdir()
        self.original = self.scope / "runs/formal_v1"
        self.original.mkdir(parents=True)
        self.retry = self.scope / "runs/recovery_retry_v1/cases" / r.TARGET / "attempt_002"
        self.retry.mkdir(parents=True)
        self.output = self.scope / "runs/reconciled_recovery_v1"
        cases = []
        for task in ("meltome", "elm", "rt"):
            for seed in (17, 42, 2026):
                for method, backbone in (("ESM", "none"), ("M0", "qwen25_1p5b"), ("M3", "qwen25_1p5b"),
                                         ("M0", "qwen3_4b_2507"), ("M3", "qwen3_4b_2507")):
                    cases.append({"task": task, "method": method, "backbone": backbone, "seed": seed,
                                  "case_id": f"{task}__{method}__{backbone}__seed{seed}"})
        self.plan_path = self.scope / "frozen_plan.json"
        self.plan = {"paths": {"scope_root": str(self.scope), "python": "/frozen/python"}, "cases": cases,
                     "resource_policy": {"gpu_memory_budget_bytes": {"qwen3_4b_2507": 18000},
                                         "rss_limit_bytes": 64000, "storage_reserve_bytes": 10000,
                                         "expected_host": "node1"}, "per_case_timeout_seconds": 28800}
        r.write(self.plan_path, self.plan)
        self.binding = {"plan_path": str(self.plan_path), "plan_sha256": r.sha(self.plan_path), "case_count": 45}
        r.write(self.original / "queue_binding.json", self.binding)
        events = []
        for case in cases:
            cid = case["case_id"]
            attempt = self.original / "cases" / cid / "attempt_001"
            attempt.mkdir(parents=True)
            (attempt / "raw_scientific.jsonl").write_bytes(b'{"prediction": 12.345}\r\n')
            (attempt / "spec.json").write_bytes(b'{"raw": "original immutable metadata"}\n')
            events.append({"case_id": cid, "status": "failed" if cid == r.TARGET else "accepted",
                           "attempt": str(attempt), "error": "CUDA out of memory" if cid == r.TARGET else None})
        self.original_events_bytes = "".join(json.dumps(e) + "\n" for e in events).encode()
        (self.original / "queue_events.jsonl").write_bytes(self.original_events_bytes)
        self.original_complete = {"case_count": 45, "plan_sha256": self.binding["plan_sha256"],
                                  "cases": {e["case_id"]: e for e in events}}
        r.write(self.original / "complete.json", self.original_complete)
        r.write(self.original / "analysis_exit.json", {"exit_code": 0})
        self.adapter = self.resources / "recovery_worker_v1.py"
        self.adapter.write_text("# explicit recovery entrypoint\n", encoding="utf-8")
        self.policy_path = self.resources / "recovery_policy_v1.json"
        policy = {"schema": "v5-recovery-resource-policy.v1", "case_id": r.TARGET,
                  "authorized_gpu_devices": [3, 4, 5, 6], "original_gpu_devices": [3, 4, 5],
                  "maximum_distinct_owned_gpus": 4, "scope_root": str(self.scope),
                  "original_plan_path": str(self.plan_path), "original_plan_sha256": self.binding["plan_sha256"]}
        r.write(self.policy_path, policy)
        spec = {"case": next(c for c in cases if c["case_id"] == r.TARGET), **self.binding,
                "gpu": 6, "gpu_memory_budget_bytes": 18000, "rss_limit_bytes": 64000,
                "storage_reserve_bytes": 10000, "expected_host": "node1", "timeout_seconds": 28800,
                "cwd": str(self.scope), "recovery_policy_path": str(self.policy_path),
                "recovery_policy_sha256": r.sha(self.policy_path), "recovery_adapter_sha256": r.sha(self.adapter),
                "command": ["/frozen/python", str(self.adapter), "--mode", "worker", "--spec", str(self.retry / "spec.json")]}
        r.write(self.retry / "spec.json", spec)
        (self.retry / "supervision").mkdir()
        r.write(self.retry / "supervision/finished.json", {"status": "completed", "exit_code": 0})
        (self.retry / "raw_scientific.jsonl").write_bytes(b'{"prediction": -999.0}\r\n')
        for phase, out, budget in (("train-canary", "qualification", "qualification_device_budget.json"),
                                    ("formal", "artifacts", "device_budget.json")):
            r.write(self.retry / f"{phase}_process.json", {"phase": phase,
                "recovery_policy_sha256": spec["recovery_policy_sha256"],
                "recovery_adapter_sha256": spec["recovery_adapter_sha256"],
                "command": ["/frozen/python", str(self.adapter), "--mode", "train-child", "--spec", str(self.retry / "spec.json"),
                            "--phase", phase, "--output", str(self.retry / out), "--receipt", str(self.retry / budget)]})

    def build(self, checker=None):
        with patch.object(r, "accepted_attempt", side_effect=checker or (lambda plan, attempt, case: {"case_id": case["case_id"]})) as mocked:
            manifest = r.build_view(self.scope, self.original, self.retry, self.output)
            return manifest, mocked

    def test_preserves_original_failure_and_every_scientific_byte(self):
        original_hash = r.tree(self.original)
        retry_hash = r.tree(self.retry)
        manifest, check = self.build()
        self.assertEqual(check.call_count, 45)
        self.assertEqual(r.tree(self.original), original_hash)
        self.assertEqual(r.tree(self.retry), retry_hash)
        self.assertEqual((self.output / "provenance/original_queue_events.jsonl").read_bytes(), self.original_events_bytes)
        case_copy = self.output / "cases" / r.TARGET
        self.assertEqual(r.tree(case_copy / "attempt_001"), r.tree(self.original / "cases" / r.TARGET / "attempt_001"))
        self.assertEqual(r.tree(case_copy / "attempt_002"), retry_hash)
        state = r.read(self.output / "complete.json")
        self.assertEqual(state["accepted"], 45)
        self.assertTrue(state["cases"][r.TARGET]["attempt"].endswith("attempt_002"))
        self.assertFalse(manifest["scores_used_to_choose_attempt"])
        self.assertEqual(r.read(self.output / "queue_binding.json")["plan_sha256"], self.binding["plan_sha256"])
        self.assertFalse((self.output / "analysis_exit.json").exists())
        receipt = r.read(case_copy / "attempt_002/formal_process.json")
        self.assertIn(str(self.retry / "spec.json"), receipt["command"])
        self.assertNotIn(str(case_copy / "attempt_002/spec.json"), receipt["command"])

    def test_technical_failure_never_becomes_score_selected_success(self):
        def check(plan, attempt, case):
            if case["case_id"] == r.TARGET:
                raise ValueError("technical acceptance failed, regardless of metrics")
            return {"case_id": case["case_id"]}
        manifest, _ = self.build(check)
        self.assertFalse(manifest["retry_accepted"])
        self.assertEqual(r.read(self.output / "complete.json")["accepted"], 44)
        self.assertEqual(r.read(self.output / "complete.json")["cases"][r.TARGET]["status"], "failed")

    def test_external_attempt_is_rejected_before_creating_view(self):
        ledger = self.original_events_bytes.decode().splitlines()
        event = json.loads(ledger[0])
        event["attempt"] = str(self.retry)
        ledger[0] = json.dumps(event)
        self.original_complete["cases"][event["case_id"]] = event
        r.write(self.original / "complete.json", self.original_complete)
        (self.original / "queue_events.jsonl").write_text("\n".join(ledger) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse(self.output.exists())

    def test_adapter_sha_tamper_is_rejected(self):
        self.adapter.write_text("# replaced\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_phase_argv_cannot_hide_original_training_entrypoint(self):
        path = self.retry / "formal_process.json"
        value = r.read(path)
        value["command"][1] = "-m"
        r.write(path, value)
        with self.assertRaisesRegex(ValueError, "Actual fresh phase"):
            self.build()

    def test_mismatched_phase_receipt_sha_is_rejected(self):
        path = self.retry / "formal_process.json"
        value = r.read(path)
        value["recovery_policy_sha256"] = "0" * 64
        r.write(path, value)
        with self.assertRaisesRegex(ValueError, "SHA binding"):
            self.build()

    def test_four_gpu_authorization_cannot_be_expanded(self):
        policy = r.read(self.policy_path)
        policy["authorized_gpu_devices"].append(7)
        r.write(self.policy_path, policy)
        spec_path = self.retry / "spec.json"
        spec = r.read(spec_path)
        spec["recovery_policy_sha256"] = r.sha(self.policy_path)
        r.write(spec_path, spec)
        with self.assertRaisesRegex(ValueError, "four-card"):
            self.build()

    def test_live_queue_or_missing_analysis_cannot_be_reconciled(self):
        (self.original / "analysis_exit.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.build()
        self.assertFalse(self.output.exists())

    def test_second_view_will_not_overwrite_first(self):
        self.build()
        with self.assertRaises(FileExistsError):
            self.build()

    def test_symlink_or_hardlink_in_attempt_is_rejected(self):
        import os
        raw = self.retry / "raw_scientific.jsonl"
        linked = self.retry / "hardlinked.jsonl"
        try:
            os.link(raw, linked)
        except OSError:
            self.skipTest("Filesystem does not support hard links")
        with self.assertRaisesRegex(ValueError, "regular files"):
            self.build()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()

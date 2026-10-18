"""Finite queue resource and completion-integrity controls; no GPUs needed."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "node_pipeline", Path(__file__).resolve().parents[1] / "resources" / "node_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class PipelineTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "Detached Linux identity/lock test")
    def test_real_detached_child_identity_and_exclusive_queue_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            scope = Path(temporary)
            with pipeline.exclusive_lock(scope / "queue.lock"):
                with self.assertRaisesRegex(RuntimeError, "Another pipeline"):
                    with pipeline.exclusive_lock(scope / "queue.lock"):
                        self.fail("Duplicate lock acquired")
            process, identity = pipeline.start_child(
                [sys.executable, "-c", "import time; time.sleep(0.2)"], scope / "child.log")
            self.assertEqual(identity["session"], process.pid)
            self.assertTrue(pipeline.same_live_process(identity))
            self.assertEqual(process.wait(timeout=5), 0)
            self.assertFalse(pipeline.same_live_process(identity))

    def test_gpu_selection_ignores_utilization_but_obeys_global_four_card_cap(self):
        snapshot = {"project_gpu_devices": [0, 1, 2], "devices": [
            {"index": 3, "free_mib": 20000, "utilization_percent": 100},
            {"index": 5, "free_mib": 22000, "utilization_percent": 0},
            {"index": 6, "free_mib": 23000, "utilization_percent": 0}]}
        self.assertEqual(pipeline.select_devices(snapshot, [3, 5, 6], set(), 18000), [3])
        snapshot["project_gpu_devices"] = [3]
        self.assertEqual(pipeline.select_devices(snapshot, [3, 5, 6], {3}, 18000), [5, 6])
        snapshot["devices"][1]["free_mib"] = 17000
        self.assertEqual(pipeline.select_devices(snapshot, [3, 5, 6], {3}, 18000), [6])

    def test_process_identity_handles_parenthesis_in_command_and_pid_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "123"
            directory.mkdir()
            fields = ["S", "1", "123", "123"] + ["0"] * 15 + ["98765"] + ["0"] * 5
            (directory / "stat").write_text("123 (python (worker)) " + " ".join(fields))
            identity = pipeline.process_identity(123, root)
            self.assertEqual(identity["start_ticks"], 98765)
            self.assertEqual(identity["session"], 123)
            with patch.object(pipeline, "process_identity", return_value={**identity, "start_ticks": 99999}):
                self.assertFalse(pipeline.same_live_process(identity))

    def test_freeze_verifies_bytes_and_rejects_escaping_or_duplicate_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            scope = Path(temporary)
            (scope / "src").mkdir()
            (scope / "config.json").write_text("{}")
            (scope / "src" / "qa_train.py").write_text("pass\n")
            entries = {name: pipeline.sha256(scope / name) for name in ("config.json", "src/qa_train.py")}
            pipeline.atomic_json(scope / "frozen_manifest.json", {"files": entries})
            self.assertEqual(len(pipeline.verify_freeze(scope)), 64)
            (scope / "src" / "qa_train.py").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                pipeline.verify_freeze(scope)
            with self.assertRaises(ValueError):
                pipeline.checked_path(scope, "../outside")

    def make_case(self, scope):
        case = {"id": "M3__seed17", "arm": "M3", "seed": 17}
        directory = scope / "runs" / case["id"] / "formal"
        directory.mkdir(parents=True)
        (scope / "data").mkdir()
        for split in ("train", "validation", "test"):
            (scope / "data" / f"{split}.jsonl").write_text(json.dumps({"id": "peptide1"}) + "\n")
        pipeline.atomic_json(scope / "config.json", {})
        pipeline.atomic_json(scope / "frozen_manifest.json", {"files": {}})
        frozen = {"manifest_sha256": pipeline.sha256(scope / "frozen_manifest.json"),
                  "config_sha256": pipeline.sha256(scope / "config.json"), "files_verified": 1}
        (directory / "best_adapter.pt").write_bytes(b"fake adapter checkpoint")
        checkpoint_sha = pipeline.sha256(directory / "best_adapter.pt")
        pipeline.atomic_json(directory / "selected_checkpoint.json", {
            "test_opened": False, "selected_epoch": 2, "epochs_completed": 6,
            "checkpoint_sha256": checkpoint_sha, "frozen": frozen,
            "optimizer_updates": 6, "training_examples_seen": 18})
        pipeline.atomic_json(directory / "training_history.json", [{"epoch_one_based": i} for i in range(1, 7)])
        pipeline.atomic_json(directory / "test_summary.json", {"normal": {}})
        pipeline.atomic_json(directory / "run_contract.json", {"arm": "M3", "seed": 17,
            "phase": "formal", "backbone_frozen": True, "frozen": frozen})
        for condition in ("normal", "mean", "shuffled", "off"):
            rows = [{"id": "peptide1", "question": question, "raw": "yes", "condition": condition}
                    for question in ("rt", "ccs", "both")]
            (directory / f"test_{condition}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n")
        normal = (directory / "test_normal.jsonl").read_text()
        for epoch in range(1, 7):
            (directory / f"validation_epoch_{epoch:02d}.jsonl").write_text(normal)
        artifacts = {path.name: {"sha256": pipeline.sha256(path), **({"rows": 3} if path.suffix == ".jsonl" else {})}
                     for path in directory.iterdir()}
        pipeline.atomic_json(directory / "completion.json", {
            "status": "complete", "arm": "M3", "seed": 17, "phase": "formal",
            "epochs_completed": 6, "optimizer_updates": 6, "training_examples_seen": 18,
            "frozen": frozen, "artifacts": artifacts})
        return case, directory

    def test_acceptance_checks_receipt_identity_sha_failure_and_complete_questions(self):
        with tempfile.TemporaryDirectory() as temporary:
            scope = Path(temporary)
            case, directory = self.make_case(scope)
            accepted = pipeline.verify_result(scope, case, {"training": {"epochs": 6}})
            self.assertEqual(accepted["optimizer_updates"], 6)
            (directory / "test_normal.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                pipeline.verify_result(scope, case, {})
            pipeline.atomic_json(directory / "failure.json", {"error": "preserved failure"})
            with self.assertRaisesRegex(ValueError, "failure"):
                pipeline.verify_result(scope, case, {})

    def test_receipt_file_alone_cannot_accept_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            scope = Path(temporary)
            case, directory = self.make_case(scope)
            receipt = pipeline.read_json(directory / "completion.json")
            receipt.pop("artifacts")
            pipeline.atomic_json(directory / "completion.json", receipt)
            with self.assertRaisesRegex(ValueError, "bind its artifacts"):
                pipeline.verify_result(scope, case, {})

    def test_live_launch_receipt_prevents_duplicate_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            scope = Path(temporary)
            pipeline.atomic_json(scope / "config.json", {})
            pipeline.atomic_json(scope / "private" / "pipeline_launch.json", {
                "process": {"pid": 123, "start_ticks": 456, "session": 123}})
            from contextlib import nullcontext
            with patch.object(pipeline, "exclusive_lock", return_value=nullcontext()), \
                 patch.object(pipeline, "validate_config"), \
                 patch.object(pipeline, "verify_freeze", return_value="frozen"), \
                 patch.object(pipeline, "same_live_process", return_value=True), \
                 patch.object(pipeline, "start_child") as start:
                receipt = pipeline.launch_pipeline(scope)
                self.assertTrue(receipt["already_running"])
                start.assert_not_called()

    def test_finite_queue_keeps_failed_case_and_completes_remaining_identities(self):
        from contextlib import nullcontext
        with tempfile.TemporaryDirectory() as temporary:
            scope = Path(temporary)
            pipeline.atomic_json(scope / "config.json", {"arms": list(pipeline.ARMS),
                "seeds": list(pipeline.SEEDS), "gpu_devices": [3, 5, 6],
                "min_free_memory_mib": 18000})
            launches = []
            class Completed:
                def poll(self):
                    return 0
            def start(arguments, log, gpu=None):
                launches.append((arguments, gpu))
                pid = 1000 + len(launches)
                return Completed(), {"pid": pid, "start_ticks": pid, "session": pid}
            def accept(_scope, case, _config):
                if case["id"] == "M3__seed17":
                    raise ValueError("retained technical failure")
                return {"verified": True}
            def postprocess(_scope, status):
                status["analysis"] = {"status": "complete"}
                status["publication"] = {"status": "complete"}
            snapshot = {"project_gpu_devices": [], "devices": [
                {"index": gpu, "free_mib": 24000} for gpu in (3, 5, 6)]}
            with patch.object(pipeline, "exclusive_lock", return_value=nullcontext()), \
                 patch.object(pipeline, "verify_freeze", return_value="frozen"), \
                 patch.object(pipeline, "validate_config"), \
                 patch.object(pipeline, "gpu_snapshot", return_value=snapshot), \
                 patch.object(pipeline, "start_child", side_effect=start), \
                 patch.object(pipeline, "same_live_process", return_value=False), \
                 patch.object(pipeline, "verify_result", side_effect=accept), \
                 patch.object(pipeline, "run_postprocess", side_effect=postprocess), \
                 patch.object(pipeline.time, "sleep"):
                status = pipeline.run_pipeline(scope)
            self.assertEqual(len(launches), 9)
            self.assertEqual(status["counts"]["accepted"], 8)
            self.assertEqual(status["counts"]["failed"], 1)
            self.assertEqual(status["status"], "terminal_with_failures")
            self.assertEqual([gpu for _, gpu in launches], [3, 5, 6] * 3)
            failed = next(case for case in status["cases"] if case["id"] == "M3__seed17")
            self.assertIn("retained technical failure", failed["reason"])


if __name__ == "__main__":
    unittest.main()

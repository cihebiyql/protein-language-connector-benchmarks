"""Publication policy tests never contact GitHub or execute git."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "resources/publish_pilot.py"
SPEC = importlib.util.spec_from_file_location("publish_pilot", SOURCE)
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


class PublicationTests(unittest.TestCase):
    def fixture(self, root):
        scope = root / publication.SCOPE_ID
        scope.mkdir()
        publication.write(scope / "config.json", {
            "qwen_path": "/data1/private/models/qwen", "esm_checkpoint": "/data/private/esm.pt",
            "publication": {"raw_elm_private": True}, "scheduled_monitor": False})
        (scope / "src").mkdir()
        (scope / "src/example.py").write_text("# Schema constants are code, not raw data.\nFIELD = 'parent_sequence'\n")
        (scope / "README.md").write_text("A frozen exploratory pilot.\n")
        return scope

    def test_allowlist_excludes_raw_elm_and_all_model_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            scope = self.fixture(Path(temporary))
            for relative in ("assets/raw.csv", "assets/checkpoint.pt", "data/panel.jsonl",
                             "cache/features.npz", "cards/test.jsonl", "dictionary.json",
                             "private/runtime.json", "reports/per_query.json"):
                path = scope / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("private")
            names = {p.relative_to(scope).as_posix() for p in publication.public_files(scope)}
        self.assertEqual(names, {"config.json", "src/example.py", "README.md"})

    def test_staging_sanitizes_values_and_dictionary_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scope = self.fixture(root)
            publication.write(scope / "reports/ASSET_PROVENANCE.json", {
                "/data/private/esm.pt": {"sha256": "abc"}, "qwen": "/data1/private/models/qwen"})
            entries, status = publication.stage(scope, root / "target", "prepared")
            config = publication.read(root / "target/config.json")
            provenance = publication.read(root / "target/reports/ASSET_PROVENANCE.json")
            self.assertEqual(config["qwen_path"], "${QWEN3_4B_MODEL_PATH}")
            self.assertEqual(config["esm_checkpoint"], "${ESM650M_CHECKPOINT}")
            self.assertIn("${ESM650M_CHECKPOINT}", provenance)
            self.assertEqual(status["status"], "prepared_outcomes_not_claimed")
            self.assertFalse(status["analysis_completed"])
            self.assertTrue(any(entry["source_sha256"] != entry["public_sha256"] for entry in entries))

    def test_private_fields_paths_sequences_and_secrets_rejected(self):
        values = [{"sample_id": "private"}, {"parent_motif_start": 3},
                  {"path": "/data1/private/cache/x"}, {"payload": "ACDEFGHIKLMNPQRSTVWYACD"},
                  {"token": "github_pat_" + "A" * 40}]
        for value in values:
            with self.subTest(value=list(value)):
                with self.assertRaises(ValueError):
                    publication.private_guard(value)

    def test_source_and_destination_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scope = self.fixture(root)
            external = root / "outside.py"
            external.write_text("private")
            try:
                (scope / "src/linked.py").symlink_to(external)
            except OSError:
                self.skipTest("Symlinks unavailable on this host")
            with self.assertRaises(ValueError):
                publication.public_files(scope)
            (scope / "src/linked.py").unlink()
            (root / "target").symlink_to(scope, target_is_directory=True)
            with self.assertRaises(ValueError):
                publication.stage(scope, root / "target", "prepared")

    def test_size_cap_rejected_before_file_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            scope = self.fixture(Path(temporary))
            path = scope / "src/oversize.py"
            with path.open("wb") as stream:
                stream.truncate(publication.MAX_BYTES + 1)
            with self.assertRaises(ValueError):
                publication.public_files(scope)

    def test_resume_own_staging_preserves_existing_history_and_excludes_unknown_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scope = self.fixture(root)
            target = root / "target"
            first, _ = publication.stage(scope, target, "prepared")
            second, _ = publication.stage(scope, target, "prepared")
            self.assertEqual(first, second)
            self.assertEqual((target / "src/example.py").read_bytes(), (scope / "src/example.py").read_bytes())
            (target / "private_data.txt").write_text("retain but refuse")
            with self.assertRaises(RuntimeError):
                publication.stage(scope, target, "prepared")
            self.assertTrue((target / "private_data.txt").exists())
        publication.verify_owned_changes(["experiments/owned/a.py"], {"experiments/owned/a.py"})
        with self.assertRaises(RuntimeError):
            publication.verify_owned_changes(["experiments/other/a.py"], {"experiments/owned/a.py"})

    def test_final_requires_recognized_terminal_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scope = self.fixture(root)
            with self.assertRaises(ValueError):
                publication.stage(scope, root / "target", "final")
            publication.write(scope / "reports/SUMMARY.json", {"status": "running"})
            with self.assertRaises(ValueError):
                publication.stage(scope, root / "target", "final")
            publication.write(scope / "reports/SUMMARY.json", {"status": "technical_inconclusive", "qa_completed": False})
            (scope / "reports/REPORT.md").write_text("The technical qualification was inconclusive.\n")
            _, status = publication.stage(scope, root / "target", "final")
            self.assertEqual(status["status"], "technical_inconclusive")
            self.assertFalse(status["qa_completed"])
            self.assertFalse(status["biological_function_or_isolated_activity_verified"])

    def test_sequence_control_aggregate_is_not_a_raw_sequence(self):
        aggregate = {"class": {"observed": 67}, "position": {"observed": 67},
                     "activity": {"observed": 67}, "evidence": {"observed": 67}}
        publication.private_guard({"qa": {"arms": {"sequence": aggregate}}})

    def test_private_sequence_still_rejected_inside_aggregate_envelope(self):
        aggregate = {"class": {"sequence": "ACDEFGHIKLMNPQRSTVWY"},
                     "position": {}, "activity": {}, "evidence": {}}
        with self.assertRaises(ValueError):
            publication.private_guard({"sequence": aggregate})

    def test_git_uses_strict_whitespace_and_bounded_noninteractive_process(self):
        with patch.object(publication.subprocess, "run") as run:
            run.return_value.stdout = "okay\n"
            self.assertEqual(publication.call(["git", "status"], "/unused", {}), "okay")
        self.assertIn("cr-at-eol", run.call_args.args[0][2])
        self.assertIs(run.call_args.kwargs["stdin"], publication.subprocess.DEVNULL)
        self.assertEqual(run.call_args.kwargs["timeout"], 600)
        self.assertTrue(run.call_args.kwargs["check"])

    def test_lore_and_environment_preserve_credentials_scope(self):
        message = publication.lore_message(final=True)
        for trailer in ("Constraint:", "Confidence:", "Scope-risk:", "Directive:", "Tested:", "Co-authored-by: OmX"):
            self.assertIn(trailer, message)
        auth = {"environment_overrides": {"BAD_KEY": "x"}}
        with self.assertRaises(ValueError):
            publication.publication_environment(auth)


if __name__ == "__main__":
    unittest.main()

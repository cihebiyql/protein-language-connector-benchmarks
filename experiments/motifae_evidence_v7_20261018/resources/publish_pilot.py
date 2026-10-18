"""Finite, allowlisted node1 publication; private ELM evidence stays private."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

SCOPE_ID = "motifae_evidence_v7_20261018"
REPOSITORY = "cihebiyql/protein-language-connector-benchmarks"
MAX_BYTES = 50 * 1024 ** 2
FIXED_PUBLIC = {
    "PROTOCOL.md", "README.md", "config.json", "reports/SUMMARY.json",
    "reports/REPORT.md", "reports/INTEGRATION_PLAN.md", "reports/ASSET_PROVENANCE.json",
    "data/PUBLIC_MANIFEST.json", "resources/CPU_TEST_RECEIPT.json", "resources/GPU_SMOKE_PUBLIC.json",
}
GENERATED = {"COMPLETION_STATUS.json", "PUBLICATION_MANIFEST.json"}
TERMINAL = {"technical_inconclusive", "complete_pilot_no_incremental_gain", "complete_pilot_promising"}
PRIVATE_KEYS = {
    "sample_id", "sample_ids", "parent_accession", "parent_sequence", "short_sequence",
    "segment_sequence", "sequence", "sequences", "parent_motif_start", "parent_motif_end",
    "short_motif_start", "short_motif_end", "known_positive_spans", "feature_vectors",
    "descriptions", "answers", "per_query", "per_query_answers", "cards", "private_key", "password",
}
PRIVATE_PATH = re.compile(r"(?:/[Dd]ata\d*/|/home/|/tmp/|[A-Za-z]:[\\/](?:Users|work)[\\/])")
SEQUENCE_TEXT = re.compile(r"\b[ACDEFGHIKLMNPQRSTVWY]{18,}\b")
SECRET_TEXT = re.compile(r"(?:gh[pousr]_[A-Za-z0-9_]{25,}|github_pat_[A-Za-z0-9_]{25,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def call(args, cwd, env):
    if args and args[0] == "git":
        args = ["git", "-c", "core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol", *args[1:]]
    return subprocess.run(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, check=True, timeout=600).stdout.strip()


def allowed_relative(relative):
    relative = Path(relative).as_posix()
    path = Path(relative)
    return relative in FIXED_PUBLIC or (len(path.parts) == 2 and path.parts[0] in
                                        {"src", "tests", "resources"} and path.suffix == ".py")


def ensure_within(path, root):
    path, root = Path(path), Path(root)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Publication path escapes its owning directory")
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError("Publication symlinks are forbidden")
        if current.parent == current:
            raise ValueError("Publication path has no owning root")
        current = current.parent
    if root.is_symlink():
        raise ValueError("Publication root symlink is forbidden")


def public_files(scope):
    scope = Path(scope)
    candidates = [scope / name for name in FIXED_PUBLIC]
    for name in ("src", "tests", "resources"):
        directory = scope / name
        if directory.is_symlink():
            raise ValueError("Publication source directory is a symlink")
        if directory.exists():
            candidates.extend(directory.glob("*.py"))
    result = []
    for path in candidates:
        if path.is_symlink():
            raise ValueError("Publication source is a symlink")
        if path.is_file():
            ensure_within(path, scope)
            if path.stat().st_size > MAX_BYTES:
                raise ValueError("Unapproved publication size")
            result.append(path)
    return sorted(set(result))


def sanitize(value, replacements):
    if isinstance(value, str):
        for source, target in replacements:
            value = value.replace(source, target)
        return value
    if isinstance(value, list):
        return [sanitize(item, replacements) for item in value]
    if isinstance(value, dict):
        return {sanitize(key, replacements): sanitize(item, replacements) for key, item in value.items()}
    return value


def private_guard(value):
    if isinstance(value, dict):
        for key, item in value.items():
            # "sequence" also names the aggregate raw-sequence control arm.
            # Only the exact four-question metric envelope is permitted here;
            # all nested fields are still recursively checked for private data.
            aggregate_sequence_arm = (str(key).lower() == "sequence" and isinstance(item, dict)
                and set(item) == {"class", "position", "activity", "evidence"}
                and all(isinstance(metrics, dict) for metrics in item.values()))
            if aggregate_sequence_arm:
                private_guard(item)
                continue
            if str(key).lower() in PRIVATE_KEYS:
                raise ValueError("Private per-query field is not allowed in publication metadata")
            private_guard(str(key))
            private_guard(item)
    elif isinstance(value, list):
        for item in value:
            private_guard(item)
    elif isinstance(value, str):
        if PRIVATE_PATH.search(value) or SEQUENCE_TEXT.search(value) or SECRET_TEXT.search(value):
            raise ValueError("Private path, sequence, or credential in publication metadata")


def replacements_for(scope, config):
    replacements = [(str(scope), "${EXPERIMENT_SCOPE}"),
                    (str(scope).replace("\\", "/"), "${EXPERIMENT_SCOPE}")]
    for key, placeholder in (("qwen_path", "${QWEN3_4B_MODEL_PATH}"),
                             ("esm_checkpoint", "${ESM650M_CHECKPOINT}"),
                             ("motif_checkpoint", "${MOTIFAE_CHECKPOINT}")):
        if key in config:
            value = config[key]
            if not isinstance(value, str) or not value:
                raise ValueError("Invalid configured model asset path")
            replacements.append((value, placeholder))
    return sorted(set(replacements), key=lambda pair: len(pair[0]), reverse=True)


def checked_public_bytes(source, replacements):
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    if SECRET_TEXT.search(text):
        raise ValueError("Credential material is forbidden even in public source files")
    transformation = "byte_exact"
    if source.suffix == ".json":
        original = json.loads(text)
        value = sanitize(original, replacements)
        private_guard(value)
        if value != original:
            raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                              allow_nan=False) + "\n").encode("utf-8")
            transformation = "external_runtime_paths_to_placeholders"
    elif source.suffix == ".md":
        clean = sanitize(text, replacements)
        private_guard(clean)
        if clean != text:
            raw = clean.encode("utf-8")
            transformation = "external_runtime_paths_to_placeholders"
    # Code contains legitimate schema/guard constants. Only executable source is
    # exempt from the biological-field-name check; real secrets remain forbidden.
    return raw, transformation


def stage(scope, target, stage_name="final"):
    scope, target = Path(scope), Path(target)
    if stage_name not in {"prepared", "final"}:
        raise ValueError("Unknown publication stage")
    if target.is_symlink():
        raise ValueError("Publication destination is a symlink")
    config = read(scope / "config.json")
    replacements = replacements_for(scope, config)
    files = public_files(scope)
    prepared = []
    for source in files:
        relative = source.relative_to(scope)
        data, transform = checked_public_bytes(source, replacements)
        prepared.append((source, relative, data, transform))
    summary_path = scope / "reports/SUMMARY.json"
    summary = read(summary_path) if summary_path.exists() else None
    if stage_name == "final" and (not summary or summary.get("status") not in TERMINAL):
        raise ValueError("Final publication requires a recognized terminal pilot summary")
    if stage_name == "final" and not (scope / "reports/REPORT.md").is_file():
        raise ValueError("Final publication requires the aggregate human-readable report")
    expected_paths = {relative.as_posix() for _, relative, _, _ in prepared} | GENERATED
    # Retain an interrupted staging directory, but refuse stale/private content.
    # Never reset, delete or sweep its files to make a publication pass.
    if target.exists():
        for existing in target.rglob("*"):
            ensure_within(existing, target)
            if existing.is_file() and existing.relative_to(target).as_posix() not in expected_paths:
                raise RuntimeError("Existing publication target contains an unapproved file")
    target.mkdir(parents=True, exist_ok=True)
    entries = []
    for source, relative, data, transform in prepared:
        destination = target / relative
        ensure_within(destination, target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        entries.append({"path": relative.as_posix(), "source_sha256": sha(source),
                        "public_sha256": sha(destination), "bytes": len(data), "transformation": transform})
    status = {
        "schema": "motifae-pilot-public-completion.v1", "scope_id": SCOPE_ID,
        "status": summary["status"] if stage_name == "final" else "prepared_outcomes_not_claimed",
        "publication_stage": stage_name, "analysis_completed": bool(stage_name == "final" and summary),
        "data_prepared": ((scope / "data/PUBLIC_MANIFEST.json").is_file() and
                          (scope / "data/panel.jsonl").is_file()),
        "feature_extraction_complete": ((scope / "cache/features_manifest.json").is_file() and
                                        read(scope / "cache/features_manifest.json").get("status") == "complete"),
        "feature_reference_smoke_public": (scope / "resources/GPU_SMOKE_PUBLIC.json").is_file(),
        "publication_host": "node1", "local_callback_required": False, "scheduled_monitor": False,
        "raw_elm_and_per_query_evidence_private": True, "prior_v6_untouched": True,
        "original_v6_stop_rule_unchanged": True, "exploratory_pilot_only": True,
        "biological_function_or_isolated_activity_verified": False,
    }
    if stage_name == "final" and summary:
        for key in ("decision", "qa_completed", "completed_queries", "expected_queries", "feature_qa_complete"):
            if key in summary:
                cleaned = sanitize(summary[key], replacements)
                private_guard(cleaned)
                status[key] = cleaned
    write(target / "COMPLETION_STATUS.json", status)
    write(target / "PUBLICATION_MANIFEST.json", {
        "schema": "motifae-pilot-public-manifest.v1", "publication_stage": stage_name,
        "files": entries, "scope_id": SCOPE_ID,
        "excluded": ["assets/**", "cache/**", "private/**", "raw ELM rows and coordinates",
                     "per-query cards and model answers", "learned feature dictionaries and model weights"],
        "public_scope": "source code, protocol, aggregate measurements and source fingerprints",
    })
    return entries, status


def verify_owned_changes(changed, allowed):
    unexpected = set(changed) - set(allowed)
    if unexpected:
        raise RuntimeError("Dedicated worktree contains changes outside the exact owned publication allowlist")


def lore_message(final=False, merge=False):
    if merge:
        why = "Preserve concurrent research publications alongside the MotifAE pilot"
        body = "Merge independent remote updates without discarding result history."
    else:
        why = ("Make the MotifAE pilot outcome independently reviewable" if final else
               "Fix the MotifAE integration protocol before inspecting held-out results")
        body = "Publish allowed code and aggregate evidence while keeping licensed raw ELM material private."
    return (why + "\n\n" + body + "\n\n"
            "Constraint: Preserve the frozen v6 experiment and stopping rule; no private biological rows or personal credentials\n"
            "Confidence: high\nScope-risk: narrow\n"
            "Directive: This exploratory pilot does not establish isolated peptide activity or rescue the original direction\n"
            "Tested: Publication allowlist, private metadata guard, staged tree and anonymous public byte checks\n"
            "Not-tested: Biological function or generalization beyond the frozen exploratory panel\n"
            "Co-authored-by: OmX <omx@oh-my-codex.dev>")


@contextmanager
def locked(path):
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def publication_environment(auth):
    overrides = auth.get("environment_overrides", {})
    if set(overrides) - {"PATH", "GIT_EXEC_PATH", "GIT_TEMPLATE_DIR"}:
        raise ValueError("Unexpected publication runtime environment overrides")
    return {**os.environ, **overrides, "GIT_SSH_COMMAND": auth["git_ssh_command"],
            "GIT_TERMINAL_PROMPT": "0", "GIT_AUTHOR_NAME": auth["git_author_name"],
            "GIT_AUTHOR_EMAIL": auth["git_author_email"], "GIT_COMMITTER_NAME": auth["git_author_name"],
            "GIT_COMMITTER_EMAIL": auth["git_author_email"], "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}


def publish(scope, stage_name="final"):
    scope = Path(scope).resolve()
    if scope.name != SCOPE_ID:
        raise ValueError("Wrong publication scope")
    runtime = read(scope / "private/runtime.json")
    auth = read(runtime["publication_auth_config"])
    env = publication_environment(auth)
    repo = Path(auth["repo"]).resolve()
    worktree = scope / "private/publication_worktree"
    receipt = scope / ("private/publication_" + stage_name + "_receipt.json")
    with locked(scope / "private/publication.lock"):
        if receipt.exists():
            return read(receipt)
        call(["git", "fetch", "origin", "main"], repo, env)
        prefix = "experiments/" + SCOPE_ID + "/"
        allowed = {prefix + p.relative_to(scope).as_posix() for p in public_files(scope)}
        allowed |= {prefix + name for name in GENERATED}
        if not worktree.exists():
            call(["git", "worktree", "add", "--detach", str(worktree), "origin/main"], repo, env)
        else:
            dirty = call(["git", "status", "--porcelain"], worktree, env)
            if dirty:
                changed = call(["git", "diff", "--name-only", "HEAD"], worktree, env).splitlines()
                untracked = call(["git", "ls-files", "--others", "--exclude-standard"], worktree, env).splitlines()
                verify_owned_changes(changed + untracked, allowed)
            else:
                try:
                    call(["git", "merge", "--ff-only", "origin/main"], worktree, env)
                except subprocess.CalledProcessError:
                    call(["git", "merge", "-m", lore_message(merge=True), "origin/main"], worktree, env)
        target = worktree / "experiments" / SCOPE_ID
        entries, status = stage(scope, target, stage_name)
        relative = target.relative_to(worktree).as_posix()
        call(["git", "add", "-f", "--", relative], worktree, env)
        call(["git", "diff", "--cached", "--check"], worktree, env)
        changed = call(["git", "diff", "--cached", "--name-only"], worktree, env)
        verify_owned_changes(changed.splitlines(), allowed)
        if changed:
            call(["git", "commit", "-m", lore_message(final=stage_name == "final")], worktree, env)
        commit = call(["git", "rev-parse", "HEAD"], worktree, env)
        for attempt in range(3):
            try:
                call(["git", "push", "origin", "HEAD:refs/heads/main"], worktree, env)
                break
            except subprocess.CalledProcessError:
                if attempt == 2:
                    raise
                call(["git", "fetch", "origin", "main"], worktree, env)
                call(["git", "merge", "-m", lore_message(merge=True), "origin/main"], worktree, env)
                commit = call(["git", "rev-parse", "HEAD"], worktree, env)
                time.sleep(2 * (attempt + 1))
        call(["git", "fetch", "origin", "main"], worktree, env)
        call(["git", "merge-base", "--is-ancestor", commit, "origin/main"], worktree, env)
        remote_head = call(["git", "rev-parse", "origin/main"], worktree, env)
        names = list(sorted(GENERATED))
        if stage_name == "final":
            names += ["reports/SUMMARY.json", "reports/REPORT.md"]
        verified = []
        for name in names:
            path = target / name
            url = f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/{relative}/{name}"
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=45) as response:
                        data = response.read(MAX_BYTES + 1)
                    if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != sha(path):
                        raise ValueError("Anonymous public bytes mismatch")
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(3 * (attempt + 1))
            verified.append({"path": relative + "/" + name, "sha256": sha(path)})
        result = {"status": "published_verified", "stage": stage_name, "commit": commit,
                  "remote_head": remote_head, "verified": verified, "files": len(entries),
                  "publication_host": "node1", "local_callback_required": False,
                  "scheduled_monitor": False, "at_utc": datetime.now(timezone.utc).isoformat(),
                  "completion_status": status}
        write(receipt, result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--stage", choices=("prepared", "final"), default="final")
    arguments = parser.parse_args()
    print(json.dumps(publish(arguments.scope, arguments.stage)), flush=True)

"""Create a private, TRAIN-only exploratory MotifAE panel without altering B1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

PREFIX = "motifae-pilot-v1|"
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
LOCKED_DIR = "data/processed/bench_v02/b1_homology_locked"
RAW_DIR = "data/raw/bench_v02/elm"
SOURCE_PATHS = (
    f"{LOCKED_DIR}/b1_samples.jsonl",
    f"{LOCKED_DIR}/b1_classes_main.jsonl",
    f"{LOCKED_DIR}/benchmark_lock.json",
    f"{LOCKED_DIR}/b1_manifest.json",
    f"{LOCKED_DIR}/group_audit.json",
    f"{LOCKED_DIR}/alignment_protocol.json",
    f"{RAW_DIR}/instances.fasta",
    f"{RAW_DIR}/instances.tsv",
    f"{RAW_DIR}/classes.tsv",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_fasta(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    accession = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            header = line[1:].split()[0]
            accession = header.split("|")[1] if "|" in header else header
            if accession in result:
                raise ValueError(f"Duplicate FASTA accession: {accession}")
            result[accession] = ""
        elif line.strip():
            if accession is None:
                raise ValueError("Sequence before FASTA header")
            result[accession] += line.strip()
    return result


def flank_size(sample_id: str, side: str) -> int:
    if side not in {"left", "right"}:
        raise ValueError("Flank side must be left or right")
    return 4 + int(_sha((sample_id + "|" + side).encode()), 16) % 9


def component_identity(member_ids: list[str]) -> tuple[str, str]:
    stable_members = sorted(set(member_ids))
    digest = _sha((PREFIX + "|".join(stable_members)).encode())
    bucket = int(digest, 16) % 10
    split = "train" if bucket < 6 else "qualification" if bucket < 8 else "test"
    return "component_" + digest, split


def build_panel(old_rows: list[dict], parents: dict[str, str], catalog: dict[str, dict]):
    """Only old TRAIN may contribute rows; group before applying eligibility filters."""
    training = [r for r in old_rows if r["split"] == "train"]
    protected = [r for r in old_rows if r["split"] != "train"]
    ids = [r["sample_id"] for r in old_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate original sample IDs")
    uf = {}

    def find(key):
        uf.setdefault(key, key)
        if uf[key] != key:
            uf[key] = find(uf[key])
        return uf[key]

    def union(a, b):
        uf[find(a)] = find(b)

    def group_keys(row):
        keys = ["parent:" + row["parent_accession"],
                "homology:" + row["groups"]["parent_homology"],
                "context:" + row["groups"]["peptide_family"]]
        keys.extend("pmid:" + str(x) for x in row["groups"]["source_publication"])
        return keys

    protected_keys = {key for row in protected for key in group_keys(row)}
    for row in training:
        key = "sample:" + row["sample_id"]
        for group in group_keys(row):
            union(key, group)
    components = defaultdict(list)
    parent_annotations = defaultdict(list)
    for row in training:
        components[find("sample:" + row["sample_id"])].append(row)
        parent_annotations[row["parent_accession"]].append(row)
    records, exclusions = [], []
    old_classes = sorted({r["class_id"] for r in training})
    for members in components.values():
        member_ids = sorted(r["sample_id"] for r in members)
        component_id, split = component_identity(member_ids)
        blocked = any(key in protected_keys for r in members for key in group_keys(r))
        for row in members:
            reason = None
            sequence = parents.get(row["parent_accession"])
            if blocked:
                reason = "component_intersects_old_validation_or_test"
            elif sequence is None:
                reason = "parent_sequence_missing"
            elif len(sequence) > 1022:
                reason = "parent_length_exceeds_1022"
            elif not sequence or not set(sequence) <= STANDARD_AA:
                reason = "parent_not_standard_20_amino_acids"
            if reason:
                exclusions.append({"sample_id": row["sample_id"], "reason": reason})
                continue
            start, end = row["start"], row["end"]
            if not (1 <= start <= end <= len(sequence)):
                raise ValueError(f"Invalid coordinates: {row['sample_id']}")
            if sequence[start - 1:end] != row["segment_sequence"]:
                raise ValueError(f"Motif sequence mismatch: {row['sample_id']}")
            left = flank_size(row["sample_id"], "left")
            right = flank_size(row["sample_id"], "right")
            window_start = max(1, start - left)
            window_end = min(len(sequence), end + right)
            short = sequence[window_start - 1:window_end]
            short_start, short_end = start - window_start + 1, end - window_start + 1
            if short[short_start - 1:short_end] != row["segment_sequence"]:
                raise ValueError("Short coordinate transfer failed")
            positive_spans = []
            for annotation in parent_annotations[row["parent_accession"]]:
                if annotation["end"] < window_start or annotation["start"] > window_end:
                    continue
                annotation_start, annotation_end = annotation["start"], annotation["end"]
                clipped_left = annotation_start < window_start
                clipped_right = annotation_end > window_end
                positive_spans.append({
                    "sample_id": annotation["sample_id"], "class_id": annotation["class_id"],
                    "parent_start": annotation_start, "parent_end": annotation_end,
                    "short_start": max(annotation_start, window_start) - window_start + 1,
                    "short_end": min(annotation_end, window_end) - window_start + 1,
                    "clipped_left": clipped_left, "clipped_right": clipped_right,
                    "fully_contained": not clipped_left and not clipped_right,
                })
            positive_spans.sort(key=lambda r: (r["parent_start"], r["parent_end"], r["sample_id"]))
            cls = catalog[row["class_id"]]
            records.append({
                "sample_id": row["sample_id"], "original_split": "train", "split": split,
                "parent_accession": row["parent_accession"], "parent_sequence": sequence,
                "parent_length": len(sequence), "short_sequence": short,
                "short_parent_start": window_start, "short_parent_end": window_end,
                "parent_motif_start": start, "parent_motif_end": end,
                "short_motif_start": short_start, "short_motif_end": short_end,
                "coordinate_system": "1-based-inclusive",
                "motif_sequence": row["segment_sequence"], "class_id": row["class_id"],
                "regex": cls["regex"], "description": cls["description"],
                "source_pmids": sorted(str(x) for x in row["groups"]["source_publication"]),
                "source_refs": row["source_refs"], "elm_instance_id": row["elm_instance_id"],
                "parent_homology_group": row["groups"]["parent_homology"],
                "peptide_family_group": row["groups"]["peptide_family"],
                "component_id": component_id, "component_member_ids": member_ids,
                "requested_left_flank": left, "requested_right_flank": right,
                "actual_left_flank": start - window_start,
                "actual_right_flank": window_end - end,
                "known_positive_spans": positive_spans,
                "known_positive_class_ids": sorted({s["class_id"] for s in positive_spans if s["fully_contained"]}),
                "annotation_use": "evaluation_only_except_new_train_dictionary_fitting",
                "molecule_scope": "annotated_region_in_parent_context",
                "isolated_peptide_activity_validated": False,
            })
    records.sort(key=lambda r: r["sample_id"])
    exclusions.sort(key=lambda r: r["sample_id"])
    class_counts = {cl: {sp: sum(r["class_id"] == cl and r["split"] == sp for r in records)
                        for sp in ("train", "qualification", "test")} for cl in old_classes}
    summary = {
        "original_train_rows": len(training), "original_protected_rows": len(protected),
        "original_train_classes": len(old_classes), "all_train_components_before_filters": len(components),
        "eligible_rows": len(records), "eligible_parents": len({r["parent_accession"] for r in records}),
        "excluded_counts": dict(sorted(Counter(r["reason"] for r in exclusions).items())),
        "excluded_rows": exclusions, "class_counts": class_counts,
        "classes_without_new_train_examples": [cl for cl, c in class_counts.items() if not c["train"]],
        "classes_missing_any_partition": [cl for cl, c in class_counts.items() if not all(c.values())],
        "splits": {sp: {
            "rows": sum(r["split"] == sp for r in records),
            "classes": len({r["class_id"] for r in records if r["split"] == sp}),
            "components": len({r["component_id"] for r in records if r["split"] == sp}),
            "parents": len({r["parent_accession"] for r in records if r["split"] == sp}),
        } for sp in ("train", "qualification", "test")},
        "parent_length": {"min": min((r["parent_length"] for r in records), default=None),
                          "max": max((r["parent_length"] for r in records), default=None)},
        "all_coordinates_validated": True,
        "old_validation_and_test_untouched": True,
    }
    return records, summary


def _write_frozen(path: Path, text: str):
    data = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Refusing to replace existing frozen output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def prepare_panel(project_root: Path | str, scope: Path | str) -> dict:
    project_root, scope = Path(project_root).resolve(), Path(scope).resolve()
    source_bytes = {rel: (project_root / rel).read_bytes() for rel in SOURCE_PATHS}
    pins = {rel: {"sha256": _sha(data), "bytes": len(data)} for rel, data in source_bytes.items()}
    old_rows = _jsonl(project_root / LOCKED_DIR / "b1_samples.jsonl")
    descriptions = {r["class_id"]: r for r in _jsonl(project_root / LOCKED_DIR / "b1_classes_main.jsonl")}
    raw_classes = list(csv.DictReader(io.StringIO("\n".join(
        line for line in source_bytes[f"{RAW_DIR}/classes.tsv"].decode().splitlines()
        if not line.startswith("#"))), delimiter="\t"))
    raw_classes = {r["ELMIdentifier"]: r for r in raw_classes}
    class_ids = sorted({r["class_id"] for r in old_rows if r["split"] == "train"})
    catalog = {cl: {"class_id": cl, "description": descriptions[cl]["class_description_no_rule"],
                    "regex": raw_classes[cl]["Regex"],
                    "source_refs": descriptions[cl]["source_refs"]} for cl in class_ids}
    parents = read_fasta(project_root / RAW_DIR / "instances.fasta")
    rows, summary = build_panel(old_rows, parents, catalog)
    panel_text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    catalog_text = _json({"scope": "old_B1_TRAIN_classes_only", "classes": list(catalog.values())})
    manifest = {
        "schema_version": 1, "panel_id": "motifae_evidence_v7_train_only_pilot_v1",
        "status": "frozen_exploratory_panel", "private_data": True,
        "eligibility_rules": ["original_B1_split_equals_train", "complete_parent_length_1_to_1022",
                              "complete_parent_standard_20_amino_acids_only",
                              "exclude_components_intersecting_old_validation_or_test"],
        "grouping": "union_all_original_train_parent_accession_parent_homology_peptide_family_and_shared_PMID_before_filtering",
        "split_rule": "int(sha256('motifae-pilot-v1|' + '|'.join(sorted(component_sample_ids))).hexdigest(),16)%10: <6 train, <8 qualification, else test",
        "flank_rule": "4+int(sha256((sample_id+'|'+side).encode()).hexdigest(),16)%9; left/right independently; clipped at parent boundaries",
        "source_pins": pins, "summary": summary,
        "source_code_pins": {"src/pilot_data.py": _sha(Path(__file__).read_bytes())},
        "outputs": {"data/panel.jsonl": {"sha256": _sha(panel_text.encode()), "bytes": len(panel_text.encode())},
                    "data/catalog.json": {"sha256": _sha(catalog_text.encode()), "bytes": len(catalog_text.encode())}},
        "limitations": [
            "This separate exploratory panel uses only old TRAIN; old validation and TEST remain frozen and unused.",
            "Class names and regex in the catalog are a candidate vocabulary, not per-example predicted evidence.",
            "Anchor and known_positive_spans annotations must be ignored by feature extraction and inference card building; only new TRAIN dictionary fitting and evaluation may consume them.",
            "Annotation transfer from parent to a short context does not measure isolated peptide activity.",
            "Unannotated positions are not experimentally validated biological negatives.",
            "Feature naming is learned only on new train; qualification must not reuse final test labels.",
            "Missing-train classes remain in evaluation; no score-driven resplitting or row filtering.",
            "ELM raw rows, sequences, coordinates and per-query evidence remain private pending redistribution review.",
        ],
    }
    for rel, data in source_bytes.items():
        if (project_root / rel).read_bytes() != data:
            raise RuntimeError(f"Original source changed while preparing: {rel}")
    _write_frozen(scope / "data/panel.jsonl", panel_text)
    _write_frozen(scope / "data/catalog.json", catalog_text)
    _write_frozen(scope / "data/manifest.json", _json(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_panel(args.project_root, args.scope)
    print(_json(manifest["summary"]))


if __name__ == "__main__":
    main()

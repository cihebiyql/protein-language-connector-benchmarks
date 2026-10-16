"""Offline A2 adapter. Never execute DeepLC or infer missing chromatography metadata."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
KNOWN_MODS = {"Acetyl", "Carbamidomethyl", "Oxidation"}
CONDITION = "PXD064311:experiment_2_march:basic_pH9:author_mean_apex"


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def parse_modifications(raw: str, sequence: str) -> tuple[list[dict], list[str]]:
    """Preserve explicit PTMs; reject unparsed states instead of stripping them."""
    if not raw:
        return [], []
    parts = raw.split("|")
    if len(parts) % 2:
        return [{"raw": raw}], ["unparsed_modification_encoding"]
    mods, reasons = [], []
    for pos, name in zip(parts[::2], parts[1::2]):
        if not re.fullmatch(r"\d+", pos) or not name:
            return [{"raw": raw}], ["unparsed_modification_encoding"]
        position = int(pos)
        mods.append({"position": position, "name": name, "source_encoding": "DeepLC_position_name"})
        if position > len(sequence) + 1:
            reasons.append("modification_position_out_of_range")
        if name not in KNOWN_MODS:
            reasons.append("unsupported_modification_preserved")
        if name == "Carbamidomethyl" and (not 1 <= position <= len(sequence) or sequence[position - 1] != "C"):
            reasons.append("carbamidomethyl_residue_mismatch")
        if name == "Oxidation" and (not 1 <= position <= len(sequence) or sequence[position - 1] != "M"):
            reasons.append("oxidation_residue_mismatch")
        if name == "Acetyl" and position != 0:
            reasons.append("acetyl_state_requires_review")
    return sorted(mods, key=lambda m: (m["position"], m["name"])), sorted(set(reasons))


def normalize_march(row: dict, number: int, source_available: bool = False) -> dict:
    seq = row["Sequence"]  # no uppercasing, deleting brackets, or silent stereo conversion
    mods, reasons = parse_modifications(row["Modifications"], seq)
    if not seq or any(c not in AA for c in seq):
        reasons.append("unsupported_sequence_state_preserved")
    value = float(row["RT"])
    std = float(row["RT_std"])
    count = int(row["RT_count"])
    if not math.isfinite(value) or value <= 0 or not math.isfinite(std) or std < 0 or count < 1:
        reasons.append("invalid_measurement")
    if value > 0 and std / value > 0.1:
        reasons.append("coefficient_of_variation_above_10_percent")
    peptidoform = seq + "|" + "|".join(f"{m.get('position', '?')}:{m.get('name', m.get('raw', ''))}" for m in mods)
    valid = not reasons
    if not source_available:
        reasons.append("PXD005573_source_domain_not_local")
    return {
        "sample_id": "a2_" + stable_id(CONDITION + "|" + peptidoform),
        "panel_id": "A2", "task": "retention_time",
        "peptide_sequence": seq, "peptidoform": peptidoform,
        "peptidoform_format": "bench_v02_sequence_and_position_name_v1",
        "modifications": mods, "condition_id": CONDITION,
        "rt_value": value, "rt_unit": "min", "source_id": "PXD064311",
        "project_id": "PXD064311", "source_record_id": f"peptides_ph9_march.csv:data_row:{number}",
        "groups": {"sequence": stable_id(seq), "peptidoform": stable_id(peptidoform), "experiment": CONDITION},
        "eligibility": {"measurement_valid": valid, "source_domain_available": source_available},
        "readiness_reasons": sorted(set(reasons)),
        "rawmetadata": {"modifications": row["Modifications"], "rt_std": std, "rt_count": count,
                        "unit_evidence": "10.1038/s41467-026-68981-5 basic-LC figure and Methods",
                        "condition_metadata": {"pH": 9, "phase": "reversed_phase", "sample": "HeLa_S3",
                            "stationary_phase": "XBridge Premier Peptide BEH C18 2.5um 75um-id 18cm 130A",
                            "mobile_phase_A": "2.5mM imidazole, 3% IPA in water",
                            "mobile_phase_B": "2.5mM imidazole, 3% IPA in 95% acetonitrile",
                            "aggregation": "author-reported mean apex RT across detections"},
                        "stereochemistry": "not individually annotated; source proteolytic peptide convention",
                        "parent_protein_mapping": "not present in distributed processed CSV"},
    }


def fixed_split(rows: list[dict]) -> dict[str, str]:
    """Keep every chemical variant of a base sequence in the same assignment."""
    groups = sorted({r["groups"]["sequence"] for r in rows}, key=lambda g: stable_id("rt-split-v1|" + g))
    n = len(groups)
    return {g: "train" if i < int(.7*n) else "validation" if i < int(.85*n) else "test" for i, g in enumerate(groups)}


def adaptation_budgets(rows: list[dict], source_sequences: set[str] | None = None, source_semantics_verified: bool = False) -> dict:
    """A target label is charged even when used only for calibration/selection."""
    assignments = fixed_split(rows)
    holdout = sorted(r["sample_id"] for r in rows if assignments[r["groups"]["sequence"]] == "test")
    pool = sorted((r for r in rows if assignments[r["groups"]["sequence"]] != "test"),
                  key=lambda r: stable_id("rt-budget-v1|" + r["sample_id"]))
    panels = {}
    for difficulty in ("condition_transfer_allow_source_overlap", "condition_and_new_sequence_transfer"):
        eligible = pool if difficulty.endswith("allow_source_overlap") else [r for r in pool if source_sequences is not None and r["peptide_sequence"] not in source_sequences]
        eval_ids = holdout if difficulty.endswith("allow_source_overlap") else sorted(r["sample_id"] for r in rows if r["sample_id"] in set(holdout) and source_sequences is not None and r["peptide_sequence"] not in source_sequences)
        panels[difficulty] = {"test_ids": eval_ids, "source_overlap_audit_available": source_sequences is not None, "budgets": {}}
        for budget in (32, 128, 512, 2048):
            selected = eligible[:budget]
            supported = len(selected) == budget and bool(eval_ids) and source_sequences is not None and source_semantics_verified
            # Keep PTM variants of one sequence together in fit or selection.
            fit_groups, fit, selection = set(), [], []
            for r in selected:
                g = r["groups"]["sequence"]
                if g in fit_groups or len(fit) < int(.8*budget):
                    fit_groups.add(g); fit.append(r["sample_id"])
                else:
                    selection.append(r["sample_id"])
            panels[difficulty]["budgets"][str(budget)] = {
                "status": "prepared" if supported else "unsupported",
                "reason": None if supported else "source_domain_or_overlap_audit_missing" if source_sequences is None else "source_rt_coordinate_and_chemical_state_unresolved" if not source_semantics_verified else "insufficient_unique_target_labels_or_holdout",
                "total_target_label_budget": budget, "actual_target_labels": len(selected),
                "label_ids": [r["sample_id"] for r in selected], "fit_ids": fit,
                "validation_ids": selection, "calibration_allowed_ids": [r["sample_id"] for r in selected],
                "accounting": "union of all fit/calibration/validation/selection labels may not exceed label_ids; no other target labels",
            }
    return panels


def prepare(raw: Path, output: Path) -> dict:
    source_path = raw.parent / "zenodo2021_PXD005573_mcp.csv"
    source_rows = []
    if source_path.exists():
        with source_path.open(encoding="utf-8-sig", newline="") as f:
            source_rows = list(csv.DictReader(f))
    source_sequences = {r["seq"] for r in source_rows} if source_rows else None
    with raw.open(encoding="utf-8-sig", newline="") as f:
        rows = [normalize_march(r, i, bool(source_rows)) for i, r in enumerate(csv.DictReader(f), 1)]
    counts = Counter(r["sample_id"] for r in rows)
    if any(n != 1 for n in counts.values()):
        raise ValueError("Duplicate condition-peptidoform keys require explicit aggregation, not silent deduplication")
    assignments = fixed_split(rows)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "a2_target_march.jsonl").open("w", encoding="utf8") as f:
        for r in rows:
            r["proposed_split"] = assignments[r["groups"]["sequence"]]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    budgets = adaptation_budgets(rows, source_sequences)
    (output / "a2_target_budgets.json").write_text(json.dumps(budgets, indent=2), encoding="utf8")
    # Source rows are candidates only: retaining legacy positions prevents a false
    # chemical match to the target's 1-based PTM format. No row is silently dropped.
    with (output / "a2_source_candidates.jsonl").open("w", encoding="utf8") as f:
        for i, r in enumerate(source_rows, 1):
            parts = r["modifications"].split("|")
            mods = [{"raw_position": p, "name": n, "position_basis": "legacy_unresolved"} for p, n in zip(parts[::2], parts[1::2])]
            obj = {"sample_id": "a2_source_" + stable_id(str(i) + "|" + str(r)), "panel_id": "A2", "task": "retention_time",
                   "peptide_sequence": r["seq"], "peptidoform": r["seq"] + "|legacy_raw:" + r["modifications"],
                   "modifications": mods, "condition_id": "PXD005573:mcp:author_processed:coordinate_unresolved",
                   "rt_value": float(r["tr"]), "rt_unit": None, "source_id": "PXD005573", "project_id": "PXD005573",
                   "source_record_id": f"train_eval/datasets/PXD005573_mcp.csv:data_row:{i}",
                   "groups": {"sequence": stable_id(r["seq"]), "legacy_peptidoform": stable_id(r["seq"] + "|" + r["modifications"])},
                   "eligibility": {"measurement_valid": False, "source_domain_available": True},
                   "readiness_reasons": ["source_RT_coordinate_unit_unverified", "legacy_PTM_index_and_fixed_modifications_unverified", "source_condition_identity_incomplete"],
                   "rawmetadata": {"original": r, "exclusion": "none; candidate only; not emitted to training-ready records"}}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    source_keys = Counter((r["seq"], r["modifications"]) for r in source_rows)
    target_seqs = {r["peptide_sequence"] for r in rows}
    audit = {"status": "target_available_source_candidates_semantics_blocked", "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
             "rows": len(rows), "unique_sequences": len({r["peptide_sequence"] for r in rows}),
             "modified_rows": sum(bool(r["modifications"]) for r in rows),
             "valid_measurement_rows": sum(r["eligibility"]["measurement_valid"] for r in rows),
             "proposed_split_rows": dict(Counter(r["proposed_split"] for r in rows)),
             "rt_range_min": [min(r["rt_value"] for r in rows), max(r["rt_value"] for r in rows)],
             "paper_reported_target_count": 6363, "count_difference": "7070 local CSV rows; 6363 equals 90% numerically, explanation not confirmed",
             "source_domain_rows_local": len(source_rows),
             "source_candidate_unique_legacy_peptidoforms": len(source_keys),
             "source_candidate_duplicate_key_rows": sum(n-1 for n in source_keys.values()),
             "source_candidate_negative_coordinate_rows": sum(float(r["tr"]) < 0 for r in source_rows),
             "source_candidate_raw_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest() if source_rows else None,
             "paper_reported_source_count": 113213,
             "source_count_difference": "113215 local candidate rows vs 113213 reported; no two-row exclusion established",
             "source_rt_coordinate_interpretation": "unknown; negative values retained, neither assigned minutes nor discarded",
             "source_overlap": {"base_sequence_intersection": len(target_seqs & source_sequences) if source_sequences is not None else None,
                                "target_base_sequences_absent_from_source": len(target_seqs - source_sequences) if source_sequences is not None else None,
                                "peptidoform_intersection": "unsupported until old/new PTM position and fixed-Cys state reconciled"},
             "novel_peptide_generalization_ready": False,
             "split_limitation": "exact sequence grouping only; parent mapping and near-sequence homology require later global leakage audit",
             "source_within_condition_panel": "blocked_source_RT_unit_chemical_state_condition_identity",
             "cross_condition_panel": "blocked_source_RT_unit_chemical_state_condition_identity",
             "target_within_condition_panel": "diagnostic_proposed_only_not_replacement_for_specified_source_domain",
             "deepLC_current_exposure": "6543-setup pretrained model; training membership not fully audited; actual-use panel only",
             "downloaded_raw_mass_spectrometry": False}
    archive_target = raw.parent / "zenodo2026_ph9_march_A.csv"
    if archive_target.exists():
        with archive_target.open(newline="") as f:
            archive_rows = list(csv.DictReader(f))
        lookup = {(r["peptide_sequence"], r["rawmetadata"]["modifications"]): r["rt_value"] for r in rows}
        archive_keys = {(r["seq"], r["modifications"]) for r in archive_rows}
        same_keys = archive_keys == set(lookup)
        audit["zenodo2026_target_crosscheck"] = {
            "rows": len(archive_rows), "keys_match_pride": same_keys,
            "max_RT_rounding_difference_min": max(abs(lookup[(r["seq"], r["modifications"])] - float(r["tr"])) for r in archive_rows) if same_keys else None,
            "predicted_tr_column": "ignored; model prediction not measured label",
        }
    (output / "a2_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf8")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/bench_v02/rt/peptides_ph9_march.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/bench_v02")
    args = parser.parse_args()
    print(json.dumps(prepare(args.raw, args.output), ensure_ascii=False, indent=2))

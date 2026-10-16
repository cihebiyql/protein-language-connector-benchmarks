"""Prepare auditable PPIKB KD candidates; never allocate a supervised split.

The main workbook supplies bibliographic and chemistry fields. Its raw Affinity
string must agree with its standardized endpoint and concentration. The research
export establishes source membership, but its rewritten Affinity is not truth.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[2]
AA = set("ACDEFGHIKLMNPQRSTVWY")
UNITS = {"M": 1., "mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12, "fM": 1e-15}
NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_book(path):
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    values = sheet.iter_rows(values_only=True)
    headers = next(values)
    rows = [dict(zip(headers, row)) for row in values]
    workbook.close()
    return list(headers), rows


def parse_affinity(text):
    """Parse only explicit single KD values; retain censoring and error bars.

    Scientific-notation multiplication is supported. Ambiguous intervals and
    approximate values remain unactionable, with the source text retained.
    """
    raw = str(text or "").strip()
    clean = raw.replace("μ", "u").replace("µ", "u").replace("−", "-")
    clean = clean.replace("≤", "<=").replace("≥", ">=")
    clean = re.sub(r"(\d+(?:\.\d+)?)\s*[×x]\s*10\s*\^\s*([+-]?\d+)", r"\1e\2", clean)
    match = re.fullmatch(
        rf"\s*K[_ ]?D\s*(=|<=|>=|<|>)\s*({NUMBER})(?:\s*(?:±|\+/-)\s*({NUMBER}))?\s*(M|mM|uM|nM|pM|fM)\s*",
        clean, re.I,
    )
    if not match:
        return {"measurement_type": None, "relation": None, "kd_molar": None,
                "pkd": None, "label_exact": False, "reported_error_molar": None,
                "parse_status": "not_explicit_single_kd", "raw": raw}
    relation, value, error, unit = match.groups()
    # Unit prefix case is meaningful; the insensitive endpoint match must not
    # silently turn mM into M or nM into another unit.
    if unit not in UNITS:
        return {"measurement_type": "KD", "relation": relation, "kd_molar": None,
                "pkd": None, "label_exact": False, "reported_error_molar": None,
                "parse_status": "unsupported_unit_case", "raw": raw}
    kd = float(value) * UNITS[unit]
    if not math.isfinite(kd) or kd <= 0:
        return {"measurement_type": "KD", "relation": relation, "kd_molar": None,
                "pkd": None, "label_exact": False, "reported_error_molar": None,
                "parse_status": "nonpositive_kd", "raw": raw}
    return {"measurement_type": "KD", "relation": relation, "kd_molar": kd,
            "pkd": -math.log10(kd), "label_exact": relation == "=",
            "reported_error_molar": float(error) * UNITS[unit] if error else None,
            "parse_status": "parsed", "raw": raw}


def meaningful(value):
    return str(value or "").strip().lower() not in {"", "none", "no", "n/a", "na", "null"}


def chemical_state(row):
    nterm, cterm = row.get("Nterminal"), row.get("Cterminal")
    supported_n = {"free", "acetyl"}
    supported_c = {"free", "amide"}
    errors = []
    if str(row.get("Linear/Cyclic", "")).lower() != "linear":
        errors.append("not_linear")
    if meaningful(row.get("Cyclization_Method")) or meaningful(row.get("Cyclization_Way")):
        errors.append("cyclization_annotation_present")
    if str(row.get("Residue_Modification", "")).lower() != "no" or meaningful(row.get("Chemical_Modification")):
        errors.append("residue_modification_or_chemistry_unresolved")
    if str(nterm or "").lower() not in supported_n or str(cterm or "").lower() not in supported_c:
        errors.append("terminal_chemistry_unsupported_or_missing")
    return {"topology": row.get("Linear/Cyclic"), "n_terminal": nterm,
            "c_terminal": cterm, "residue_modification": row.get("Residue_Modification"),
            "chemical_modification": row.get("Chemical_Modification"),
            "cyclization_method": row.get("Cyclization_Method"),
            "explicitly_representable": not errors}, errors


def load_external_locks(root):
    """Lock prior revealed evaluation rows and the external source publications."""
    files = ["t3_g1_records.jsonl", "t3_d2_pairs.jsonl", "benz_samples.jsonl"]
    sequences, record_ids, groups, evidence = set(), [], set(), []
    for name in files:
        path = root / "data/processed" / name
        if not path.exists():
            continue
        evidence.append({"path": str(path.relative_to(root)), "sha256": digest(path)})
        for line in path.read_text(encoding="utf8").splitlines():
            row = json.loads(line)
            if row.get("split") not in {"external", "test", "val", "validation"}:
                continue
            record_ids.append(row.get("pair_id", row.get("sample_id")))
            for field in ["sequence_a", "sequence_b", "sequence", "peptide_sequence"]:
                if row.get(field):
                    sequences.add(row[field])
            if row.get("gene"):
                groups.add(row["gene"])
    return {"locked_source_dois": ["10.1038/s44320-024-00055-4", "10.1038/s41597-025-05754-7"],
            "pepbi_dataset_doi": "10.5061/dryad.wstqjq2wk",
            "pepbi_original_source_doi_inventory_complete": False,
            "locked_peptide_sequences": sorted(sequences), "locked_old_record_ids": record_ids,
            "locked_parent_gene_series": sorted(groups), "evidence": evidence,
            "limitation": "PEPBI source measurement workbook unavailable locally; cross-library original-experiment exclusion remains incomplete. Near-family exclusion requires global similarity grouping."}


def normalize(row, research_row, locks):
    affinity = parse_affinity(row.get("Affinity"))
    reasons = []
    peptide, target = row.get("Peptide_Sequence"), row.get("Protein_Sequence")
    if row.get("Document_Type") != "Research Articles" or research_row is None:
        reasons.append("not_verified_research_source")
    if str(row.get("Affinity_Parameters", "")).lower() != "kd":
        reasons.append("standardized_endpoint_not_kd")
    if affinity["parse_status"] != "parsed":
        reasons.append("raw_affinity_not_unambiguous_kd")
    standard_um = row.get("uM")
    concentration_agrees = None
    if affinity["kd_molar"] is not None and isinstance(standard_um, (int, float)):
        concentration_agrees = math.isclose(affinity["kd_molar"], standard_um * 1e-6, rel_tol=1e-6, abs_tol=1e-18)
        if not concentration_agrees:
            reasons.append("raw_vs_standardized_concentration_conflict")
    elif affinity["kd_molar"] is not None:
        reasons.append("standardized_concentration_missing")
    if not peptide or not 5 <= len(peptide) <= 50 or not set(peptide) <= AA:
        reasons.append("peptide_not_5_50_uppercase_standard_aa")
    if not target or not set(target) <= AA:
        reasons.append("target_sequence_missing_or_nonstandard")
    chemistry, chemistry_reasons = chemical_state(row)
    reasons.extend(chemistry_reasons)
    cross_conflicts = []
    if research_row:
        for field in ["Peptide_Sequence", "Protein_Sequence", "Linear/Cyclic"]:
            if row.get(field) != research_row.get(field):
                cross_conflicts.append(field)
        if cross_conflicts:
            reasons.append("main_research_molecular_identity_conflict")
    doi = str(row.get("DOI") or "").strip().lower()
    if not doi:
        reasons.append("source_doi_missing")
    external = []
    if doi in locks["locked_source_dois"]:
        external.append("locked_external_publication")
    if peptide in locks["locked_peptide_sequences"]:
        external.append("prior_revealed_evaluation_peptide_exact_match")
    if external:
        reasons.append("external_or_prior_evaluation_quarantine")
    identity = {"peptide": peptide, "target": target, "chemistry": chemistry,
                "doi": doi, "affinity": row.get("Affinity"), "assay": row.get("Affinity_Assay")}
    duplicate_group = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    broad = not reasons
    strict_reasons = ["actual_assay_construct_not_verified", "conditions_missing_temperature_ph_buffer"]
    # Blank conditions never constitute a shared condition, including within a DOI.
    return {
        "sample_id": "ppikb:" + str(row["ID"]), "panel_id": "A1", "task": "affinity",
        "peptide_sequence": peptide, "target_sequence": target,
        "target_sequence_role": "database_reference_sequence_actual_assay_construct_unverified",
        **{key: affinity[key] for key in ["measurement_type", "relation", "kd_molar", "pkd", "label_exact", "reported_error_molar"]},
        "pkd_relation": {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "=": "="}.get(affinity["relation"]),
        "chemical_state": chemistry,
        "experimental_context": {"assay": row.get("Affinity_Assay"), "temperature_kelvin": None,
                                  "ph": None, "buffer": None, "actual_construct_sequence": None,
                                  "actual_construct_verified": False, "comparable_condition_group": None},
        "source_doi": doi or None, "source_record_id": row["ID"],
        "groups": {"source_doi": doi or None, "peptide_exact": peptide,
                   "target_uniprot": row.get("UniProt_ID"), "target_sequence_sha256": hashlib.sha256((target or "").encode()).hexdigest() if target else None,
                   "repeat_measurement_candidate": duplicate_group, "peptide_family": None,
                   "target_family": None, "original_experiment": None},
        "eligibility": {"broad_reference_target": broad, "strict_comparable_pair": False,
                        "A1_broad_reference_target": broad,
                        "A1_exact_regression_candidate": broad and affinity["label_exact"],
                        "A1_censored_candidate": broad and not affinity["label_exact"],
                        "A1_strict_comparable_pairs": False},
        "split_status": "pending_global_group_split" if broad else "quarantined",
        "readiness_reasons": reasons + (["global_family_and_original_experiment_holdout_audit_pending"] if broad else []),
        "strict_comparable_pair_reasons": strict_reasons,
        "external_holdout_reasons": external,
        "provenance": {"per_field_source": {"molecular_identity": "Main.xlsx", "chemical_state": "Main.xlsx",
                                               "endpoint_value": "Main.xlsx:Affinity", "source_membership": "research.xlsx:ID",
                                               "source_doi": "Main.xlsx:DOI", "target_sequence": "Main.xlsx:Protein_Sequence"},
                       "raw_vs_standardized_concentration_agrees": concentration_agrees,
                       "main_research_identity_conflicts": cross_conflicts,
                       "research_rewritten_affinity": research_row.get("Affinity") if research_row else None},
        "raw_metadata": row,
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf8")


def prepare(root=ROOT):
    raw, out = root / "data/raw/bench_v02/ppikb", root / "data/processed/bench_v02"
    out.mkdir(parents=True, exist_ok=True)
    main_headers, main = load_book(raw / "Main.xlsx")
    research_headers, research = load_book(raw / "research.xlsx")
    _, branch = load_book(raw / "Branch.xlsx")
    _, patent = load_book(raw / "patent.xlsx")
    rindex = {row["ID"]: row for row in research}
    locks = load_external_locks(root)
    rows = [normalize(row, rindex.get(row["ID"]), locks) for row in main]
    main_ids = {row["source_record_id"] for row in rows}
    orphan = [{"source_record_id": row["ID"], "source": "research.xlsx", "status": "quarantined",
               "reasons": ["missing_from_main_workbook_bibliography_and_chemistry_unverified"], "raw_metadata": row}
              for row in research if row["ID"] not in main_ids]
    write_jsonl(out / "a1_normalized.jsonl", rows)
    write_jsonl(out / "a1_row_audit.jsonl", [{key: row[key] for key in ["sample_id", "source_record_id", "source_doi", "eligibility", "readiness_reasons", "strict_comparable_pair_reasons", "external_holdout_reasons", "provenance"]} for row in rows] + orphan)
    write_jsonl(out / "a1_broad_candidates.jsonl", [row for row in rows if row["eligibility"]["A1_broad_reference_target"]])
    write_jsonl(out / "a1_strict_pairs.jsonl", [])
    (out / "a1_external_locks.json").write_text(json.dumps(locks, ensure_ascii=False, indent=2), encoding="utf8")
    repeats = collections.Counter(row["groups"]["repeat_measurement_candidate"] for row in rows if row["eligibility"]["A1_broad_reference_target"])
    summary = {"schema_version": "pepbridge_a1_v0.2", "source_workbook_rows": {"Main.xlsx": len(main), "research.xlsx": len(research), "patent.xlsx": len(patent), "Branch.xlsx": len(branch)},
               "paper_reported_kd": 10607, "main_standardized_kd": sum(str(row.get("Affinity_Parameters")).lower() == "kd" for row in main),
               "research_orphans": len(orphan), "broad_reference_target_candidates": sum(row["eligibility"]["A1_broad_reference_target"] for row in rows),
               "exact_regression_candidates": sum(row["eligibility"]["A1_exact_regression_candidate"] for row in rows),
               "censored_candidates": sum(row["eligibility"]["A1_censored_candidate"] for row in rows),
               "strict_condition_controlled_pairs": 0, "approved_supervised_split_rows": 0,
               "duplicate_measurement_candidate_groups": sum(v > 1 for v in repeats.values()),
               "duplicate_measurement_candidate_rows": sum(v for v in repeats.values() if v > 1),
               "exclusion_reason_counts_nonexclusive": dict(collections.Counter(reason for row in rows for reason in row["readiness_reasons"])),
               "main_research_peptide_conflicts": sum("Peptide_Sequence" in row["provenance"]["main_research_identity_conflicts"] for row in rows),
               "source_sha256": {name: digest(raw / name) for name in ["Main.xlsx", "research.xlsx", "patent.xlsx", "Branch.xlsx"]},
               "preparation_script_sha256": digest(Path(__file__)),
               "global_readiness_blockers": ["PEPBI original experiment DOI inventory missing", "peptide and target family similarity split not yet allocated", "actual assay constructs and comparable conditions require original-source audit"],
               "data_dictionary": {"main_headers": main_headers, "research_headers": research_headers,
                                   "target_sequence": "Provided reference protein sequence, not verified as actual measured construct",
                                   "pkd": "-log10(KD in mol/L); censored labels keep relation and invert pkd inequality",
                                   "repeats": "Rows are retained; identical metadata hashes indicate possible duplicate experiments, not independent units",
                                   "experimental_context": "No pH, buffer, temperature, or actual construct fields occur in downloaded tables; explicit nulls are retained"}}
    (out / "a1_preparation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({key: value for key, value in summary.items() if key != "data_dictionary"}, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    prepare(parser.parse_args().root)

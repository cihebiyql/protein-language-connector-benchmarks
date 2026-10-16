"""Traceable Kliche screening pairs and separately locked quantitative G1 panel.

No group-level effect is used as a peptide-pair label. EV3 supplies endpoint
p-value metadata and abundance; EV1 supplies the physical library sequences.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re

import openpyxl

from prepare_benz_baseline import AA, UnionFind, expand_ev8_sequence, near_duplicate

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def load_sheet(raw, file, sheet):
    it = openpyxl.load_workbook(raw / file, read_only=True, data_only=True)[sheet].values
    headers = next(it)
    return [dict(zip(headers, row), source_row=i) for i, row in enumerate(it, 2) if row[0]]


def parse_kd(text):
    """Retain censoring; n.b. has no numeric lower bound in Table 1."""
    text = re.sub(r"\s+", "", text).replace("−", "-")
    if text in ("n.b.", "n.d."):
        return {"kind": text, "value": None}
    relation = "upper" if text.startswith("<") else "lower" if text.startswith(">") else "point"
    vals = text.lstrip("<>").split("±")
    return {"kind": relation, "value": float(vals[0]), "reported_error": float(vals[1]) if len(vals) > 1 else None}


def affinity_direction(a, b):
    """Predeclared >=2-fold central-value/bound rule, never impute n.b. to 20x."""
    if a["value"] is None or b["value"] is None:
        return None, "no_numeric_bound"
    av, bv = a["value"], b["value"]
    if a["kind"] in ("point", "upper") and b["kind"] in ("point", "lower") and bv >= 2 * av:
        return "A", "direction_resolvable"
    if b["kind"] in ("point", "upper") and a["kind"] in ("point", "lower") and av >= 2 * bv:
        return "B", "direction_resolvable"
    return None, "small_change" if a["kind"] == b["kind"] == "point" else "ambiguous_bounds"


def published_sequences(text):
    text = re.sub(r"\d+|-|\s", "", text)
    # Table 1 BRCA2 denotes three adjacent substitutions as independent brackets.
    a, b = text, text
    for match in re.finditer(r"\(([A-Z]+)/([A-Z]+)\)", text):
        a = a.replace(match.group(), match[1], 1)
        b = b.replace(match.group(), match[2], 1)
    if not all(re.fullmatch("[" + AA + "]+", x) for x in (a, b)):
        raise ValueError(text)
    return a, b


def published_mutation(text, canonical_gene):
    """Parse the full mutation suffix after the complete printed gene token."""
    printed_gene = {"CHEK2": "CHK2"}.get(canonical_gene, canonical_gene)
    if not text.startswith(printed_gene):
        raise ValueError(f"Published gene does not match {canonical_gene}: {text}")
    suffix = text[len(printed_gene):].strip()
    pattern = "[" + AA + r"]\d+[" + AA + "]"
    if not re.fullmatch(pattern + "(?:/" + pattern + ")*", suffix):
        raise ValueError(f"Unparsed full published mutation suffix: {text}")
    return suffix


def make_g1(raw, library, targets, benz):
    ev8 = load_sheet(raw, "EV8.xlsx", "Peptides")
    peptides = []
    for row in ev8:
        if row["Peptide"].startswith("FITC-"):
            continue
        gene = re.split(r"\s|\(", row["Peptide"])[0]
        if gene == "CHK2":
            gene = "CHEK2"
        a, b = expand_ev8_sequence(row["Sequence wt/mutant"])
        peptides.append(dict(row, gene=gene, sequence_a=a, sequence_b=b))
    by_gene = defaultdict(list)
    for row in peptides:
        by_gene[row["gene"]].append(row)
    rows = []
    for table in csv.DictReader((raw / "table1_expanded.csv").open(encoding="utf-8-sig")):
        gene = re.match(r"[A-Z0-9]+", table["gene_mutation_as_published"])[0]
        if gene == "BUB1S492F":
            gene = "BUB1"
        if gene == "CHK2":
            gene = "CHEK2"
        a, b = published_sequences(table["peptide_sequence_as_published"])
        matches = [r for r in by_gene[gene] if (r["sequence_a"], r["sequence_b"]) == (a, b)]
        notes = []
        identity_ok = len(matches) == 1
        if not identity_ok:
            notes.append("Table1 sequence and EV8 synthetic peptide do not agree; retain published Table1 sequence, exclude direction scoring")
        if gene == "BCL11A":
            notes.append("C48F supported by EV1/EV8/Table1; narrative C47F differs; synthetic C50A/library C-to-A retained")
        if gene == "ZNF526":
            notes.append("Table1 L160T conflicts with K/T sequence; EV8 and EV1 explicitly support K160T")
        if gene == "KMT2D":
            notes.append("Table1/EV8 residue range 2993-3308 conflicts with P2301S; exact peptide matches EV1 0-based start 2292")
        appendix_sequence = None
        if gene == "ABRAXAS1":
            appendix_sequence = "LDLDDRWQF(K/Q)RSRLLD"
        elif gene == "BCL11A":
            appendix_sequence = "DHDLLT(C/F)GQCQMNY"
        elif gene == "FANCA":
            appendix_sequence = "MSD(S/L)WVPNSASGQK"
        appendix_conflict = appendix_sequence is not None and published_sequences(appendix_sequence) != (a, b)
        if appendix_conflict:
            notes.append(f"Appendix Figure S5 page 5 sequence {appendix_sequence} conflicts with Table1/EV8; primary panel prioritizes Table1+EV8, sensitivity panel excludes this row")
        domain = table["protein_domain"]
        domain_gene = domain.split()[0].replace("MAP1-", "MAP1")
        if domain_gene == "MAP1":
            domain_gene = "MAP1LC3A" if "LC3A" in domain else "MAP1LC3B"
        matches_t = [t for t in targets if t["gene_name"].upper() == domain_gene.upper()]
        if len(matches_t) > 1:
            matches_t = [t for t in matches_t if t["domain_name"].upper() in domain.upper()]
        target_id = matches_t[0]["bait"] if len(matches_t) == 1 else "g1:" + re.sub(r" (major|minor)$", "", domain)
        construct_id = target_id
        pocket_match = re.search(r" (major|minor)$", domain)
        pocket = pocket_match[1] if pocket_match else "unspecified"
        if pocket_match:
            target_id += "|pocket=" + pocket
        kd_a, kd_b = parse_kd(table["kd_wt_uM_as_published"]), parse_kd(table["kd_mut_uM_as_published"])
        label, status = affinity_direction(kd_a, kd_b)
        if not identity_ok:
            status, label = "identity_conflict", None
        accessions = sorted({r["protein_accession"] for r in library if r["protein_gene"] == gene})
        exposure = []
        for sample in benz:
            if sample["split"] != "train":
                continue
            parents = sample["parents"]
            parent_match = any(p["accession"] in accessions or p["gene"] == gene for p in parents)
            seq_match = any(near_duplicate(sample["sequence"], seq) for seq in (a, b)) if not parent_match else False
            if parent_match or seq_match:
                exposure.append({"sample_id": sample["sample_id"], "reason": "same_parent" if parent_match else "near_sequence"})
        mutation = published_mutation(table["gene_mutation_as_published"], gene)
        row = dict(table, pair_id=f"g1_table1_{int(table['source_table_row']):02d}",
                   sequence_a=a, sequence_b=b, gene=gene, parent_accessions=accessions,
                   target_id=target_id, target_info=domain, label=label,
                   target_construct_id=construct_id, measurement_pocket=pocket,
                   group_id="g1_" + gene, series_id=gene + ":" + mutation,
                   split="external", kd_a=kd_a, kd_b=kd_b, direction_status=status,
                   identity_verified=identity_ok, identity_notes=notes,
                   sequence_source_consistent=identity_ok and not appendix_conflict,
                   appendix_s5_sequence_as_published=appendix_sequence,
                   appendix_s5_sequence_conflict=appendix_conflict,
                   source_evidence={"table1_row": int(table["source_table_row"]), "ev8_rows": [r["source_row"] for r in matches]},
                   benz_train_exposure=exposure, clean_external=not exposure,
                   chemistry="N-terminal acetylation; C-terminal amide; any synthetic Y retained as published")
        rows.append(row)
    return rows, peptides


def supported_pair(group, wt, mut, hits):
    """Recover measured count direction; endpoint p-values are selection metadata.

    EV3 does not expose the comparison-object index for endpoint p-values, so
    these values never certify significance of this specific physical pair.
    """
    a = hits.get((group["bait"], wt["protein_accession"], wt["library_peptide"]))
    b = hits.get((group["bait"], mut["protein_accession"], mut["library_peptide"]))
    if (a and a["wt_or_mut"] != "wt") or (b and (b["wt_or_mut"] != "mut" or b["mutation"] != group["mutation"])):
        return None, "EV3_endpoint_mutation_identity_disagrees"
    evidence = {"ev3_wt_row": a["source_row"] if a else None, "ev3_mut_row": b["source_row"] if b else None,
                "endpoint_reported_p_values": {"wild_type": a["p_value"] if a else None, "mutant": b["p_value"] if b else None},
                "single_pair_significance_verified": False,
                "significance_mapping": "endpoint_comparison_object_index_not_publicly_resolved",
                "label_basis": "measured_screen_count_direction_weak_supervision"}
    if a and b:
        if a["p_value"] != b["p_value"] or a["p_value"] >= .01:
            return None, "endpoint_p_metadata_selection_condition_failed"
        av, bv = a["mean_counts_%"], b["mean_counts_%"]
        if av == bv:
            return None, "equal_peptide_abundance"
        evidence.update(mode="both_EV3_endpoints", wt_mean_counts_pct=av, mut_mean_counts_pct=bv)
        return ("A" if av > bv else "B", evidence), None
    for observed, absent_name, label in ((a, "mutants", "A"), (b, "wildtypes", "B")):
        if not observed or observed["p_value"] >= .01 or observed["mean_counts_%"] <= 0:
            continue
        if group[absent_name + "_observed"] == 0 and all(float(v) == 0 for v in group[absent_name + "_proportion"].split(",")):
            evidence.update(mode="EV3_observed_endpoint_and_EV6_zero_observed_other_type",
                            observed_mean_counts_pct=observed["mean_counts_%"], absent_type=absent_name,
                            absent_type_observed=0, absent_type_proportions=group[absent_name + "_proportion"])
            return (label, evidence), None
    return None, "missing_significant_endpoint_or_missing_counterpart_abundance"


def make_d2(raw, library, g1, targets):
    by_id = {r["relationship_id"]: r for r in library}
    by_mut = defaultdict(list)
    for row in library:
        if row["wt_or_mut"] == "mut":
            by_mut[row["mutation_id"]].append(row)
    hit_rows = load_sheet(raw, "EV3.xlsx", "A-binding_peptides")
    grouped_hits = defaultdict(list)
    for r in hit_rows:
        grouped_hits[(r["bait"], r["prey_accession"], r["prey_hit_peptide"])].append(r)
    # Repeated endpoints without a unique measurement are deliberately unusable.
    hits = {key: rows[0] for key, rows in grouped_hits.items() if len(rows) == 1}
    target_map = {r["bait"]: r for r in targets}
    reserved_genes = {r["gene"] for r in g1}
    reserved_acc = {a for r in g1 for a in r["parent_accessions"]}
    candidates = [r for r in load_sheet(raw, "EV6.xlsx", "A Domain-mutation_pairs")
                  if r["mutation_effect_on_PPI"] in ("enhanced", "diminished") and r["number_significant_peptide_pairs"] > 0]
    result, audit = [], []
    for group in candidates:
        record = {"ev6_row": group["source_row"], "bait": group["bait"], "mutation": group["mutation"],
                  "group_effect_audit_only": group["mutation_effect_on_PPI"], "relationship_checks": []}
        supported = []
        for mut in sorted(by_mut[group["mutation"]], key=lambda r: (r["relates_to"], r["relationship_id"])):
            wt = by_id.get(mut["relates_to"])
            check = {"mutant_relationship_id": mut["relationship_id"], "wt_relationship_id": mut["relates_to"]}
            reason = None
            if not wt or mut["relationship_id"] not in wt["relates_to"].split(";"):
                reason = "nonreciprocal_or_missing_EV1_relationship"
            elif wt["protein_accession"] != mut["protein_accession"] or wt["peptide_start"] != mut["peptide_start"]:
                reason = "EV1_parent_or_coordinates_disagree"
            elif any(len(r["library_peptide"]) != 16 or not set(r["library_peptide"]) <= set(AA) for r in (wt, mut)):
                reason = "not_canonical_16_residue_library_peptide"
            elif wt["library_peptide"] == mut["library_peptide"]:
                reason = "physical_library_sequences_are_identical"
            else:
                pair, reason = supported_pair(group, wt, mut, hits)
                if pair:
                    label, evidence = pair
                    supported.append((wt, mut, label, evidence))
                    check["evidence"] = evidence
            check["reason"] = reason or "supported_specific_pair"
            record["relationship_checks"].append(check)
        if not supported:
            record["status"] = "excluded_no_specific_pair_evidence"
        else:
            wt, mut, label, evidence = supported[0]
            record["selected_relationship_ids"] = [wt["relationship_id"], mut["relationship_id"]]
            if group["prey_gene_name"] in reserved_genes or group["prey_accession"] in reserved_acc:
                record["status"] = "excluded_G1_entire_parent"
            elif any(near_duplicate(wt["library_peptide"], r["sequence_a"]) or near_duplicate(mut["library_peptide"], r["sequence_b"]) for r in g1):
                record["status"] = "excluded_G1_near_sequence"
            else:
                record["status"] = "included"
                target = target_map.get(group["bait"])
                if target is None:
                    # EV2 SMURF2_WW__ versus EV3/6 SMURF2_WW_3_; preserve EV6 ID.
                    suffix = "_".join(group["bait"].split("_")[-3:])
                    matches = [t for t in targets if t["bait"].endswith(suffix)]
                    if len(matches) != 1:
                        raise ValueError(f"Ambiguous target identity: {group['bait']}")
                    target = matches[0]
                pair_id = "d2_" + hashlib.sha256((group["bait"] + ":" + mut["relationship_id"]).encode()).hexdigest()[:12]
                result.append({"pair_id": pair_id, "sequence_a": wt["library_peptide"], "sequence_b": mut["library_peptide"],
                               "target_id": group["bait"], "target_info": f"{target['gene_name']} {target['domain_name']} residues {target['amino_acids_range']}",
                               "label": label, "gene": group["prey_gene_name"], "parent_accession": group["prey_accession"],
                               "mutation": group["mutation"], "start_zero_based": wt["peptide_start"],
                               "wt_relationship_id": wt["relationship_id"], "mutant_relationship_id": mut["relationship_id"],
                               "source_evidence": dict(evidence, ev1_wt_row=wt["source_row"], ev1_mut_row=mut["source_row"], ev6_row=group["source_row"]),
                               "label_semantics": "relative phage-selection abundance, not an absolute affinity or a nonbinding claim"})
        audit.append(record)
    return result, audit


def split_pairs(rows):
    uf = UnionFind(len(rows))
    for i, a in enumerate(rows):
        for j in range(i):
            b = rows[j]
            if a["parent_accession"] == b["parent_accession"] or any(near_duplicate(a[x], b[y]) for x in ("sequence_a", "sequence_b") for y in ("sequence_a", "sequence_b")):
                uf.union(i, j)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[uf.find(i)].append(row)
    strata = defaultdict(list)
    for key in sorted(groups, key=lambda k: min(r["pair_id"] for r in groups[k])):
        strata["".join(sorted({r["label"] for r in groups[key]}))].append(key)
    rng = random.Random(42)
    train_counts = {s: int(.7 * len(keys)) for s, keys in strata.items()}
    val_counts = {s: int(.15 * len(keys)) for s, keys in strata.items()}
    # Largest remainders preserve the global 70/15/15 group counts.
    for counts, fraction in ((train_counts, .7), (val_counts, .15)):
        order = sorted(strata, key=lambda s: (-(fraction * len(strata[s]) - counts[s]), s))
        for s in order[:int(fraction * len(groups)) - sum(counts.values())]:
            counts[s] += 1
    for stratum in sorted(strata):
        keys = strata[stratum]
        rng.shuffle(keys)
        n_train, n_val = train_counts[stratum], val_counts[stratum]
        for i, key in enumerate(keys):
            split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
            group_id = "d2g_" + hashlib.sha256("|".join(sorted(r["pair_id"] for r in groups[key])).encode()).hexdigest()[:12]
            for row in groups[key]:
                row.update(group_id=group_id, split=split)
    return rows


def apply_input_coverage_qc(rows, library):
    """Drop selected pairs without both molecules observed in the input library.

    Keep the already locked selected pair and group/split; never replace a failed
    pair with a later relation from the same group.
    """
    by_id = {r["relationship_id"]: r for r in library}
    kept, removed = [], []
    for row in rows:
        wt_n = by_id[row["wt_relationship_id"]]["peptide_coverage_counts"]
        mut_n = by_id[row["mutant_relationship_id"]]["peptide_coverage_counts"]
        row["source_evidence"].update(ev1_wt_input_library_coverage=wt_n, ev1_mut_input_library_coverage=mut_n)
        row["input_library_coverage_passed"] = wt_n is not None and mut_n is not None and wt_n > 0 and mut_n > 0
        if row["input_library_coverage_passed"]:
            kept.append(row)
        else:
            removed.append(dict(row, qc_exclusion_reason="at_least_one_endpoint_not_observed_in_input_library"))
    return kept, removed


def main():
    raw = ROOT / "data/raw/kliche2024"
    output = ROOT / "data/processed"
    for name in ("t3_d2_pairs.jsonl", "t3_d2_candidate_audit.jsonl", "t3_split_manifest.json"):
        source = output / name
        snapshot = output / (source.stem + "_precoverage65" + source.suffix)
        if source.exists() and not snapshot.exists():
            snapshot.write_bytes(source.read_bytes())
    library = load_sheet(raw, "EV1.xlsx", "B_GenVar_Peptides")
    targets = load_sheet(raw, "EV2.xlsx", "A_Domains_in_study")
    benz_path = output / "benz_samples.jsonl"
    benz = [json.loads(line) for line in benz_path.read_text(encoding="utf-8").splitlines()]
    g1, peptides = make_g1(raw, library, targets, benz)
    d2, excluded = make_d2(raw, library, g1, targets)
    locked_path = output / "t3_split_manifest_precoverage65.json"
    if locked_path.exists():
        locked = {r["pair_id"]: r for r in json.loads(locked_path.read_text(encoding="utf-8"))["pairs"]}
        assert set(locked) == {r["pair_id"] for r in d2}, "Pre-QC selected pair population changed"
        for row in d2:
            row.update(group_id=locked[row["pair_id"]]["group_id"], split=locked[row["pair_id"]]["split"])
    else:
        split_pairs(d2)
        write_json(locked_path, {"seed": 42, "pairs": [{k: r[k] for k in ("pair_id", "group_id", "split")} for r in d2]})
    for row in d2:
        exposure = []
        for sample in benz:
            if sample["split"] != "train":
                continue
            same_parent = any(p["accession"] == row["parent_accession"] for p in sample["parents"])
            if same_parent or any(near_duplicate(sample["sequence"], row[x]) for x in ("sequence_a", "sequence_b")):
                exposure.append({"sample_id": sample["sample_id"], "reason": "same_parent" if same_parent else "near_sequence"})
        row.update(benz_train_exposure=exposure, clean_after_t1_pretraining=not exposure)
    d2, coverage_excluded = apply_input_coverage_qc(d2, library)
    removed_by_ev6 = {r["source_evidence"]["ev6_row"]: r for r in coverage_excluded}
    for record in excluded:
        if record["ev6_row"] in removed_by_ev6:
            removed = removed_by_ev6[record["ev6_row"]]
            record.update(status="excluded_zero_input_library_coverage", precoverage_status="included", qc_excluded_pair=removed)
    write_jsonl(output / "t3_d2_input_coverage_exclusions.jsonl", coverage_excluded)
    write_jsonl(output / "t3_d2_pairs.jsonl", d2)
    write_jsonl(output / "t3_d2_clean_test_pairs.jsonl", [r for r in d2 if r["split"] == "test" and r["clean_after_t1_pretraining"]])
    write_jsonl(output / "t3_g1_records.jsonl", g1)
    write_jsonl(output / "t3_g1_direction_pairs.jsonl", [r for r in g1 if r["label"]])
    write_jsonl(output / "t3_g1_clean_direction_pairs.jsonl", [r for r in g1 if r["label"] and r["clean_external"]])
    write_jsonl(output / "t3_g1_source_consistent_direction_pairs.jsonl", [r for r in g1 if r["label"] and r["sequence_source_consistent"]])
    write_jsonl(output / "t3_d2_candidate_audit.jsonl", excluded)
    summary = {"seed": 42, "split_rule": "preserve original 65-pair seed42 parent/near-sequence grouped 28/6/6 allocation; exclude zero-input-library-coverage pairs after allocation without replacement or re-splitting",
               "schema_version": "t3_coverage_qc_series_suffix_v3",
               "preparation_script_sha256": sha(Path(__file__)),
               "generated_data_sha256": {name: sha(output / name) for name in (
                   "t3_d2_pairs.jsonl", "t3_g1_records.jsonl", "t3_g1_direction_pairs.jsonl",
                   "t3_g1_clean_direction_pairs.jsonl", "t3_g1_source_consistent_direction_pairs.jsonl")},
               "d2_label_strength": "real screen-count direction weak supervision; endpoint p comparison identity unresolved; specific-pair statistical significance not verified",
               "input_library_qc": "both EV1 peptide_coverage_counts > 0",
               "precoverage_manifest_sha256": sha(locked_path), "input_coverage_excluded_count": len(coverage_excluded),
               "benz_samples_sha256": sha(benz_path), "benz_split_manifest_sha256": sha(output / "benz_split_manifest.json"),
               "source_sha256": {p.name: sha(p) for p in [*(raw / f"EV{i}.xlsx" for i in (1, 2, 3, 6, 8)), raw / "table1_expanded.csv"]},
               "candidate_groups": len(excluded), "candidate_status": dict(Counter(r["status"] for r in excluded)),
               "d2_pairs": len(d2), "d2_splits": dict(Counter(r["split"] for r in d2)),
               "d2_group_splits": {s: len({r["group_id"] for r in d2 if r["split"] == s}) for s in ("train", "val", "test")},
               "d2_labels_by_split": {s: dict(Counter(r["label"] for r in d2 if r["split"] == s)) for s in ("train", "val", "test")},
               "d2_t1_train_exposed_by_split": {s: sum(bool(r["benz_train_exposure"]) for r in d2 if r["split"] == s) for s in ("train", "val", "test")},
               "g1_records": len(g1), "g1_status": dict(Counter(r["direction_status"] for r in g1)),
               "g1_scored": sum(bool(r["label"]) for r in g1), "g1_clean_scored": sum(bool(r["label"]) and r["clean_external"] for r in g1),
               "g1_source_consistent_scored": sum(bool(r["label"]) and r["sequence_source_consistent"] for r in g1),
               "g1_parent_groups": len({r["group_id"] for r in g1}),
               "g1_scored_mutation_series": len({r["series_id"] for r in g1 if r["label"]}),
               "g1_t1_train_exposed_records": sum(bool(r["benz_train_exposure"]) for r in g1),
               "g1_reserved_parent_genes": sorted({r["gene"] for r in g1}),
               "pairs": [{k: r[k] for k in ("pair_id", "group_id", "split")} for r in d2]}
    split_parents = {s: {r["parent_accession"] for r in d2 if r["split"] == s} for s in ("train", "val", "test")}
    assert all(not split_parents[a] & split_parents[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    assert not {r["gene"] for r in d2} & {r["gene"] for r in g1}
    summary["leak_checks"] = {"d2_cross_split_same_parent": 0, "d2_G1_same_parent_gene": 0,
                              "d2_cross_split_near_sequences": 0, "original_benz_split_unchanged": True}
    write_json(output / "t3_split_manifest.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("pairs", "source_sha256")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

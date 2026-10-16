"""Reconcile the legacy source against primitive bytes in Guan 2019's iRT data.

This is an opcode *reader*, not an unpickler: globals/reducers are never called.
Only fixed-shape float64 one-hot matrices and iRT vectors are interpreted.
The old CSV and its previously exported candidates are not modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickletools
import struct
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VOCABULARY = "KRPTNAQVSGILCMJHFYWEDB"
CONDITION = "PXD005573:Guan2019:HeLa_HEK293:1m_4h:pooled_median_iRT"


class PrimitiveInflateReader:
    """Bounded-memory raw-deflate reader with complete primitive-byte checksums."""

    def __init__(self, compressed: bytes):
        self.decoder = zlib.decompressobj(-15)
        self.remaining = compressed
        self.buffer = b""
        self.position = 0
        self.sha256 = hashlib.sha256()
        self.crc32 = 0

    def read(self, n: int) -> bytes:
        while len(self.buffer) < n:
            block = self.decoder.decompress(self.remaining, max(65536, n - len(self.buffer)))
            self.remaining = self.decoder.unconsumed_tail
            if not block:
                break
            self.buffer += block
        result, self.buffer = self.buffer[:n], self.buffer[n:]
        self.position += len(result)
        self.sha256.update(result)
        self.crc32 = zlib.crc32(result, self.crc32)
        return result

    def readline(self) -> bytes:
        output = bytearray()
        while True:
            char = self.read(1)
            output.extend(char)
            if char in (b"\n", b""):
                return bytes(output)

    def tell(self) -> int:
        return self.position


def legacy_encoding(row: dict) -> str:
    """Interpret only the already audited old CSV's zero-based oxidation format."""
    sequence = list(row["seq"])
    acetyl = False
    parts = row["modifications"].split("|")
    for position, name in zip(parts[::2], parts[1::2]):
        if name == "Oxidation":
            pos = int(position)
            if not 0 <= pos < len(sequence) or sequence[pos] != "M":
                raise ValueError("Unexpected legacy oxidation state")
            sequence[pos] = "J"
        elif name == "Acetyl" and position == "0":
            acetyl = True
        else:
            raise ValueError("Unverified legacy modification")
    return ("B" if acetyl else "") + "".join(sequence)


def decode_peptidoform(encoded: str) -> tuple[str, list[dict]]:
    """Restore B/J and fixed Cys state; do not resolve lossy D/E identities."""
    acetyl = encoded.startswith("B")
    residues = encoded[1:] if acetyl else encoded
    if "B" in residues or any(x not in VOCABULARY[:-1] for x in residues):
        raise ValueError("Unsupported original one-hot chemical token")
    sequence = residues.replace("J", "M")
    modifications = []
    if acetyl:
        modifications.append({"position": 0, "name": "Acetyl", "evidence": "Guan2019 original one-hot token B"})
    for position, residue in enumerate(residues, 1):
        if residue == "J":
            modifications.append({"position": position, "name": "Oxidation", "evidence": "Guan2019 original one-hot token J"})
        if residue == "C":
            modifications.append({"position": position, "name": "Carbamidomethyl", "evidence": "Bruderer2017 fixed-Cys search and Guan2019 encoder removing explicit Carbamidomethyl"})
    return sequence, modifications


def reconcile(compressed_path: Path, legacy_path: Path, output_dir: Path, evidence_path: Path | None = None) -> dict:
    with legacy_path.open(newline="") as handle:
        legacy = list(csv.DictReader(handle))
    compressed = compressed_path.read_bytes()
    stream = PrimitiveInflateReader(compressed)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Only small, auditable encodings are retained; the 0.9GB array is streamed.
    encodings: list[str] = []
    original_y = None
    key = None
    events = []
    seen_training_X = seen_training_y = False
    while True:
        offset = stream.tell()
        code = stream.read(1)
        if not code:
            raise ValueError("Truncated pickle before STOP")
        opcode = pickletools.code2op[code.decode("latin1")]
        argument = None
        if opcode.name in ("BINBYTES", "BINBYTES8", "BYTEARRAY8"):
            size = struct.unpack("<I" if opcode.name == "BINBYTES" else "<Q", stream.read(4 if opcode.name == "BINBYTES" else 8))[0]
            if size > 100000 or key in ("training_X", "training_y"):
                events.append({"key": key, "byte_count": size, "offset": stream.tell()})
                if key == "training_X":
                    if seen_training_X or size != len(legacy) * 41 * 22 * 8:
                        raise ValueError("Unexpected training_X shape/duplicate payload")
                    seen_training_X = True
                    for i in range(0, len(legacy), 200):
                        count = min(200, len(legacy) - i)
                        matrix = np.frombuffer(stream.read(count * 41 * 22 * 8), dtype="<f8").reshape(count, 41, 22)
                        if not np.all((matrix == 0) | (matrix == 1)) or not np.all(matrix.sum(axis=2) <= 1):
                            raise ValueError("Original array is not a verified one-hot matrix")
                        for sample in matrix:
                            encodings.append("".join(VOCABULARY[int(v.argmax())] for v in sample if v.sum() == 1))
                elif key == "training_y":
                    if seen_training_y or size != len(legacy) * 8:
                        raise ValueError("Unexpected training_y shape/duplicate payload")
                    seen_training_y = True
                    original_y = np.frombuffer(stream.read(size), dtype="<f8").copy()
                else:
                    left = size
                    while left:
                        chunk = stream.read(min(left, 1048576))
                        if not chunk:
                            raise ValueError("Truncated array payload")
                        left -= len(chunk)
                continue
            argument = stream.read(size)
        elif opcode.arg is not None:
            argument = opcode.arg.reader(stream)
        if opcode.name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE") and argument in ("training_X", "training_y", "testing_X", "testing_y"):
            key = argument
        if opcode.name == "STOP":
            break
    if original_y is None or len(encodings) != len(legacy):
        raise ValueError("Missing verified primitive arrays")
    legacy_y = np.array([float(row["tr"]) for row in legacy])
    if not np.array_equal(original_y, legacy_y):
        raise ValueError("Original measured iRT labels do not exactly match the candidate CSV")
    if evidence_path is not None:
        evidence = json.loads(evidence_path.read_text())
        if evidence["uncompressed_bytes"] != stream.tell() or evidence["crc32"].lower() != f"{stream.crc32:08x}":
            raise ValueError("Original ZIP member size/CRC verification failed")
    changed = []
    uncertain = 0
    oxidation_sites = fixed_cam_sites = fixed_cam_rows = acetyl_rows = 0
    unique_forms = set()
    output_path = output_dir / "a2_source_irt_reconciled.jsonl"
    mapping_path = output_dir / "a2_source_irt_row_mapping.jsonl"
    with output_path.open("w", encoding="utf8") as output, mapping_path.open("w", encoding="utf8") as mapping:
        for i, (row, encoded, irt) in enumerate(zip(legacy, encodings, original_y), 1):
            old_encoded = legacy_encoding(row)
            repair = encoded != old_encoded
            if repair and encoded != "B" + old_encoded:
                raise ValueError("Mismatch requires a transformation beyond verified missing N-acetyl")
            seq, mods = decode_peptidoform(encoded)
            if seq != row["seq"]:
                raise ValueError("Original one-hot base sequence differs from legacy candidate")
            has_lossy_de = any(c in "DE" for c in seq)
            oxidation_sites += sum(m["name"] == "Oxidation" for m in mods)
            fixed_cam_sites += sum(m["name"] == "Carbamidomethyl" for m in mods)
            fixed_cam_rows += any(m["name"] == "Carbamidomethyl" for m in mods)
            acetyl_rows += any(m["name"] == "Acetyl" for m in mods)
            uncertain += has_lossy_de
            if repair:
                changed.append(i)
            peptidoform = seq + "|" + "|".join(f"{m['position']}:{m['name']}" for m in mods)
            unique_forms.add(peptidoform)
            row_key = f"Guan2019:training:{i-1}"
            record = {
                "sample_id": "a2_irt_" + hashlib.sha256(row_key.encode()).hexdigest()[:24],
                "panel_id": "A2", "task": "retention_time", "peptide_sequence": seq,
                "peptidoform": peptidoform, "modifications": mods,
                "condition_id": CONDITION, "rt_value": float(irt), "rt_unit": "iRT",
                "source_id": "Guan2019_Zenodo2652602", "project_id": "PXD005573",
                "source_record_id": row_key,
                "groups": {"sequence": hashlib.sha256(seq.encode()).hexdigest()[:24], "peptidoform": hashlib.sha256(peptidoform.encode()).hexdigest()[:24]},
                "eligibility": {"measurement_valid": True, "source_domain_available": True, "chemical_identity_resolved": not has_lossy_de, "exact_2026_source_version_verified": False, "training_ready": False},
                "readiness_reasons": ["exact_2026_source_version_unverified"] + (["native_D_E_versus_deamidated_N_Q_identity_lost_upstream"] if has_lossy_de else []),
                "rawmetadata": {"original_one_hot_encoding": encoded, "legacy_csv_row": i, "original_training_array_index": i-1,
                                "missing_N_acetyl_restored": repair, "rt_scale_evidence": "exact original Guan2019 training_y equality; iRT article and plot_irt.py",
                                "fixed_Cys_status": "Carbamidomethyl per original experimental protocol; original encoder removed its text",
                                "lossy_deamidation_policy": "D/E tokens can encode native acidic residues or upstream deamidated N/Q; never assert unmodified D/E identity",
                                "source_snapshot": "2019 original training arrays cross-matching 2021 CSV; exact 2026 training snapshot unverified"},
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            mapping.write(json.dumps({"sample_id": record["sample_id"], "legacy_csv_data_row": i, "original_array_row": i-1,
                                      "iRT_raw_equal": True, "sequence_and_variable_modifications_equal_before_repair": not repair,
                                      "restored_modification": {"position": 0, "name": "Acetyl"} if repair else None,
                                      "chemical_identity_resolved": not has_lossy_de}) + "\n")
    summary = {
        "status": "iRT_scale_and_original_one_hot_PTMs_reconciled_exact_2026_snapshot_still_blocked",
        "rows": len(legacy), "original_iRT_labels_exactly_equal": len(legacy), "label_value_changes": 0,
        "rt_unit": "iRT", "minute_conversion_performed": False, "min_shift_performed": False,
        "source_measurement_scale_verified": True,
        "legacy_oxidation_position_basis": "zero_based", "reconciled_residue_position_basis": "one_based", "N_terminal_position": 0,
        "oxidation_site_annotations": oxidation_sites,
        "fixed_carbamidomethyl_site_annotations": fixed_cam_sites,
        "rows_with_fixed_carbamidomethyl": fixed_cam_rows,
        "N_acetyl_rows_after_restoration": acetyl_rows,
        "missing_N_acetyl_restored_rows": len(changed), "restored_acetyl_legacy_csv_rows": changed,
        "native_D_E_or_upstream_deamidation_uncertain_rows": uncertain,
        "chemical_identity_resolved_rows": len(legacy)-uncertain, "unique_reconciled_peptidoform_keys": len(unique_forms),
        "exact_2026_source_version_verified": False, "cross_condition_training_ready": False,
        "original_pickle_uncompressed_bytes": stream.tell(), "original_pickle_sha256": stream.sha256.hexdigest(), "original_pickle_crc32": f"{stream.crc32:08x}",
        "compressed_member_sha256": hashlib.sha256(compressed).hexdigest(),
        "legacy_csv_sha256": hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
        "primitive_payloads": events,
        "outputs": {"reconciled": str(output_path), "row_mapping": str(mapping_path)},
        "old_raw_and_candidate_artifacts_modified": False,
    }
    (output_dir / "a2_source_irt_reconciliation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compressed", type=Path, default=ROOT / "data/raw/bench_v02/rt/Guan2019_irt_data.deflate")
    parser.add_argument("--legacy", type=Path, default=ROOT / "data/raw/bench_v02/rt/zenodo2021_PXD005573_mcp.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/bench_v02")
    parser.add_argument("--evidence", type=Path, default=ROOT / "data/raw/bench_v02/rt/Guan2019_irt_zip_member_evidence.json")
    args = parser.parse_args()
    result = reconcile(args.compressed, args.legacy, args.output, args.evidence)
    print(json.dumps({k:v for k,v in result.items() if k != "restored_acetyl_legacy_csv_rows"}, ensure_ascii=False, indent=2))

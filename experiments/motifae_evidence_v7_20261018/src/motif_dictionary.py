"""Frozen TRAIN-only MotifAE association dictionary and private held-out audit.

An association is an annotation-localization result, never verified peptide
activity. Unannotated residues are a reference population, not biological
negatives. TEST annotations are accepted only by the evaluation functions;
inference cards use sequence-derived caches and the frozen dictionary alone.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge


DEFAULTS = {
    "qualification_auc": 0.70,
    "min_train_parents": 2,
    "min_qualification_parents": 2,
    "top_k_associations": 3,
    "negative_reference_quantile": 0.95,
    "raw_probe_ridge_alpha": 1.0,
}
ASSOCIATION_SCOPE = "motifassociation_notverifiedactivity"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_text(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                      allow_nan=False) + "\n"


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def frozen_write(path: Path, data: str) -> None:
    value = data.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != value:
            raise RuntimeError(f"Refusing to overwrite a frozen artifact: {path}")
        return
    with path.open("xb") as stream:
        stream.write(value)


def load_config(scope: Path) -> dict:
    config_path = scope / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    method = {**DEFAULTS, **config.get("dictionary", {})}
    if set(method) != set(DEFAULTS):
        raise ValueError("Unrecognized dictionary configuration key")
    if not (0 <= method["qualification_auc"] <= 1
            and 0 <= method["negative_reference_quantile"] <= 1):
        raise ValueError("Invalid AUROC or quantile")
    for key in ("min_train_parents", "min_qualification_parents", "top_k_associations"):
        if not isinstance(method[key], int) or method[key] < 1:
            raise ValueError(f"Invalid {key}")
    if method["raw_probe_ridge_alpha"] <= 0:
        raise ValueError("Ridge alpha must be positive")
    return {"dictionary": method, "cache_root": config.get("cache_root", "cache")}


class FeatureStore:
    """Bounded CPU cache; no full L x 40960 dense materialization."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.read_receipts: dict[str, dict] = {}

    def path(self, kind: str, sequence: str) -> Path:
        if kind not in {"full", "short"}:
            raise ValueError("Feature kind must be full or short")
        return self.root / kind / (sha256_bytes(sequence.encode()) + ".npz")

    @lru_cache(maxsize=3)
    def load(self, kind: str, sequence: str) -> tuple[np.ndarray, sparse.csr_matrix]:
        path = self.path(kind, sequence)
        with np.load(path, allow_pickle=False) as obj:
            h33 = obj["h33"]
            values = obj["csr_data"]
            indices = obj["csr_indices"]
            indptr = obj["csr_indptr"]
            shape = obj["csr_shape"]
        if (h33.dtype != np.float32 or h33.shape != (len(sequence), 1280)
                or values.dtype != np.float32 or indices.dtype != np.int32
                or indptr.dtype != np.int32 or shape.dtype != np.int64
                or tuple(shape) != (len(sequence), 40960)):
            raise ValueError(f"Invalid cache dtype/shape: {path}")
        if not np.all(np.isfinite(h33)) or not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite cache: {path}")
        if np.any(values < 0):
            raise ValueError("MotifAE activations must be nonnegative")
        matrix = sparse.csr_matrix((values, indices, indptr), shape=tuple(shape))
        matrix.check_format(full_check=True)
        if not matrix.has_canonical_format:
            raise ValueError("CSR cache must have sorted, unique feature columns per residue")
        matrix.eliminate_zeros()
        relative = path.relative_to(self.root).as_posix()
        self.read_receipts[relative] = {"sha256": file_sha(path), "bytes": path.stat().st_size}
        return h33, matrix


def sparse_column_auc(matrix: sparse.spmatrix, positive: np.ndarray) -> np.ndarray:
    """Mann-Whitney AUROC for every column, with all zero/value ties exact.

    Sort only stored nonzeros. Implicit zeros are one tied group per feature,
    so runtime/memory scale with nnz rather than residues times 40960. Negative
    values are supported here for independent mathematical unit tests.
    """
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    positive = np.asarray(positive, dtype=bool)
    if positive.shape != (matrix.shape[0],):
        raise ValueError("Positive mask length differs from residue count")
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return np.full(matrix.shape[1], np.nan)
    coo = matrix.tocoo()
    if not coo.nnz:
        return np.full(matrix.shape[1], 0.5)
    if not np.all(np.isfinite(coo.data)):
        raise ValueError("Nonfinite activation")
    order = np.lexsort((coo.data, coo.col))
    cols, values = coo.col[order], coo.data[order]
    is_pos = positive[coo.row[order]].astype(np.float64)
    starts = np.r_[0, np.flatnonzero((cols[1:] != cols[:-1])
                                   | (values[1:] != values[:-1])) + 1]
    group_cols, group_values = cols[starts], values[starts]
    group_sizes = np.diff(np.r_[starts, len(cols)])
    pos = np.add.reduceat(is_pos, starts)
    neg = group_sizes - pos
    col_starts = np.r_[0, np.flatnonzero(group_cols[1:] != group_cols[:-1]) + 1]
    col_sizes = np.diff(np.r_[col_starts, len(group_cols)])
    global_before = np.cumsum(neg) - neg
    before = global_before - np.repeat(global_before[col_starts], col_sizes)
    n_features = matrix.shape[1]
    active_pos = np.bincount(group_cols, weights=pos, minlength=n_features)
    active_neg = np.bincount(group_cols, weights=neg, minlength=n_features)
    zero_pos, zero_neg = n_pos - active_pos, n_neg - active_neg
    wins = np.bincount(group_cols, weights=pos * (before + 0.5 * neg), minlength=n_features)
    wins += 0.5 * zero_pos * zero_neg
    wins += np.bincount(group_cols[group_values > 0],
                        weights=pos[group_values > 0] * zero_neg[group_cols[group_values > 0]],
                        minlength=n_features)
    wins += np.bincount(group_cols[group_values < 0],
                        weights=neg[group_values < 0] * zero_pos[group_cols[group_values < 0]],
                        minlength=n_features)
    return wins / (n_pos * n_neg)


def dense_auc(values: np.ndarray, positive: np.ndarray) -> float | None:
    value = sparse_column_auc(sparse.csr_matrix(np.asarray(values)[:, None]), positive)[0]
    return float(value) if np.isfinite(value) else None


def spans_mask(length: int, spans: list[tuple[int, int]]) -> np.ndarray:
    """All public-facing coordinates are 1-based inclusive."""
    mask = np.zeros(length, dtype=bool)
    for start, end in spans:
        if not (1 <= start <= end <= length):
            raise ValueError(f"Out-of-bounds 1-based span {(start, end)} for {length}")
        mask[start - 1:end] = True
    return mask


def row_spans(row: dict, class_id: str, kind: str) -> list[tuple[int, int]]:
    prefix = "parent" if kind == "full" else "short"
    spans = set()
    if row["class_id"] == class_id:
        spans.add((int(row[f"{prefix}_motif_start"]), int(row[f"{prefix}_motif_end"])))
    for annotation in row.get("known_positive_spans", []):
        if annotation["class_id"] == class_id:
            spans.add((int(annotation[f"{prefix}_start"]), int(annotation[f"{prefix}_end"])))
    return sorted(spans)


def class_parents(rows: list[dict], class_id: str) -> list[dict]:
    groups = {}
    for row in rows:
        spans = row_spans(row, class_id, "full")
        if not spans:
            continue
        key = row["parent_accession"]
        group = groups.setdefault(key, {"parent_accession": key,
                                       "sequence": row["parent_sequence"], "spans": set()})
        if group["sequence"] != row["parent_sequence"]:
            raise ValueError("Inconsistent parent sequence")
        group["spans"].update(spans)
    return [dict(g, spans=sorted(g["spans"])) for _, g in sorted(groups.items())]


def mean_or_none(values) -> float | None:
    values = [float(x) for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(values)) if values else None


def _probe_scores(h33: np.ndarray, model: dict) -> np.ndarray:
    return ((h33.astype(np.float64) - np.asarray(model["mean"]))
            / np.asarray(model["scale"])) @ np.asarray(model["coef"]) + model["intercept"]


def fit_raw_probe(groups: list[dict], store: FeatureStore, alpha: float, quantile: float) -> dict:
    xs, ys, weights = [], [], []
    for group in groups:
        h33, _ = store.load("full", group["sequence"])
        positive = spans_mask(len(h33), group["spans"])
        if not positive.any() or positive.all():
            continue
        xs.append(h33)
        ys.append(positive.astype(np.float64))
        # Equal parent contributions; within each parent equal total weight for
        # annotated and unannotated residues. Normalize total to N below.
        weights.append(np.where(positive, 0.5 / positive.sum(), 0.5 / (~positive).sum()))
    if not xs:
        raise ValueError("No valid TRAIN parent localization contrast")
    x, y, weight = np.vstack(xs).astype(np.float64), np.concatenate(ys), np.concatenate(weights)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1
    x = (x - mean) / scale
    weight *= len(weight) / weight.sum()
    fit = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
    fit.fit(x, y, sample_weight=weight)
    model = {"mean": mean.tolist(), "scale": scale.tolist(),
             "coef": fit.coef_.tolist(), "intercept": float(fit.intercept_),
             "alpha": alpha, "standardization": "unweighted TRAIN residues only",
             "weights": "equal parents; within parent equal annotated/unannotated mass; total weight N"}
    pred = fit.predict(x)
    model["threshold"] = float(np.quantile(pred[y == 0], quantile))
    offset, peaks, aucs = 0, [], []
    for yy in ys:
        scores = pred[offset:offset + len(yy)]
        mask = yy.astype(bool)
        peaks.append(float(scores[mask].max()))
        aucs.append(dense_auc(scores, mask))
        offset += len(yy)
    model["train_positive_peak_median"] = float(np.median(peaks))
    model["train_auc"] = mean_or_none(aucs)
    return model


def fit_dictionary(rows: list[dict], catalog: list[dict], store: FeatureStore,
                   method: dict | None = None) -> dict:
    method = {**DEFAULTS, **(method or {})}
    train = [r for r in rows if r["split"] == "train"]
    qualification = [r for r in rows if r["split"] == "qualification"]
    entries = []
    for candidate in sorted(catalog, key=lambda r: r["class_id"]):
        cl = candidate["class_id"]
        groups, qual_groups = class_parents(train, cl), class_parents(qualification, cl)
        entry = {"class_id": cl, "description": candidate["description"],
                 "regex": candidate["regex"], "feature_idx": None,
                 "train_parent_count": len(groups), "qualification_parent_count": len(qual_groups),
                 "train_auc": None, "qualification_auc": None, "threshold": None,
                 "train_positive_peak_median": None, "qualified": False,
                 "qualification_reason": "insufficient_train_parents", "raw_probe": None,
                 "association_scope": ASSOCIATION_SCOPE, "activity_claim": "not established"}
        if len(groups) < method["min_train_parents"]:
            entries.append(entry)
            continue
        sum_auc, count = np.zeros(40960, dtype=np.float64), 0
        for group in groups:
            _, activations = store.load("full", group["sequence"])
            mask = spans_mask(len(group["sequence"]), group["spans"])
            auc = sparse_column_auc(activations, mask)
            if np.isfinite(auc).all():
                sum_auc += auc
                count += 1
        entry["train_defined_parent_count"] = count
        if count < method["min_train_parents"]:
            entry["qualification_reason"] = "insufficient_defined_train_parent_aucs"
            entries.append(entry)
            continue
        mean_auc = sum_auc / count
        # np.argmax returns the lowest feature ID among exact ties.
        feature = int(np.argmax(mean_auc))
        background, peaks = [], []
        for group in groups:
            _, activations = store.load("full", group["sequence"])
            mask = spans_mask(len(group["sequence"]), group["spans"])
            values = activations[:, feature].toarray().ravel()
            if not mask.any() or mask.all():
                continue
            background.extend(values[~mask].tolist())
            peaks.append(float(values[mask].max()))
        entry.update(feature_idx=feature, train_auc=float(mean_auc[feature]),
                     threshold=float(np.quantile(background, method["negative_reference_quantile"])),
                     train_positive_peak_median=float(np.median(peaks)))
        probe = fit_raw_probe(groups, store, method["raw_probe_ridge_alpha"],
                              method["negative_reference_quantile"])
        qual_aucs, probe_aucs = [], []
        for group in qual_groups:
            h33, activations = store.load("full", group["sequence"])
            mask = spans_mask(len(h33), group["spans"])
            qual_aucs.append(dense_auc(activations[:, feature].toarray().ravel(), mask))
            probe_aucs.append(dense_auc(_probe_scores(h33, probe), mask))
        entry["qualification_auc"] = mean_or_none(qual_aucs)
        entry["qualification_defined_parent_count"] = sum(x is not None for x in qual_aucs)
        probe["qualification_auc"] = mean_or_none(probe_aucs)
        probe["qualification_defined_parent_count"] = sum(x is not None for x in probe_aucs)
        for target, auc_key in ((entry, "qualification_auc"), (probe, "qualification_auc")):
            enough = target["qualification_defined_parent_count"] >= method["min_qualification_parents"]
            auc = target[auc_key]
            target["qualified"] = bool(enough and auc is not None and auc >= method["qualification_auc"])
            target["qualification_reason"] = ("passed_fixed_qualification" if target["qualified"]
                                               else "insufficient_qualification_parents" if not enough
                                               else "qualification_auc_below_fixed_threshold")
        entry["raw_probe"] = probe
        entries.append(entry)
    return {
        "schema_version": 1, "association_scope": ASSOCIATION_SCOPE,
        "activity_claim": "not established", "method": method, "classes": entries,
        "feature_selection": "single best TRAIN mean within-parent annotated-vs-unannotated AUROC; lowest feature ID exact tie",
        "reference_population": "unannotated residues; not experimentally validated negatives",
        "feature_threshold_rule": "TRAIN unannotated quantile; if zero require strictly positive activation",
        "naming_rule": "TRAIN-selected feature frozen; qualification count and AUROC gate only; no reselection",
        "test_used_for_fitting_or_naming": False,
        "source_pins": {
            "train_rows_sha256": sha256_bytes(json_text(train).encode()),
            "qualification_rows_sha256": sha256_bytes(json_text(qualification).encode()),
            "catalog_sha256": sha256_bytes(json_text(catalog).encode()),
            "fit_cache_files": dict(sorted(store.read_receipts.items())),
        },
    }


def feature_summary(matrix: sparse.csr_matrix, count: int = 32) -> list[dict]:
    """Top active feature peaks; no annotation or biological names involved."""
    coo = matrix.tocoo()
    if not coo.nnz:
        return []
    order = np.lexsort((coo.row, -coo.data, coo.col))
    columns = coo.col[order]
    first = np.r_[0, np.flatnonzero(columns[1:] != columns[:-1]) + 1]
    selected = order[first]
    selected = selected[np.lexsort((coo.col[selected], -coo.data[selected]))][:count]
    return [{"feature_id": int(coo.col[i]), "peak": float(coo.data[i]),
             "position": int(coo.row[i]) + 1} for i in selected]


def peak_region(values: np.ndarray, threshold: float) -> tuple[int, int, int, float] | None:
    if not len(values):
        return None
    peak_idx = int(np.argmax(values))
    peak = float(values[peak_idx])
    if not peak > threshold:
        return None
    start = end = peak_idx
    while start > 0 and values[start - 1] > threshold:
        start -= 1
    while end + 1 < len(values) and values[end + 1] > threshold:
        end += 1
    return start + 1, end + 1, peak_idx + 1, peak


def score_candidates(h33: np.ndarray, matrix: sparse.csr_matrix, dictionary: dict,
                     arm: str) -> list[dict]:
    candidates = []
    for entry in dictionary["classes"]:
        model = entry if arm == "motifae" else entry.get("raw_probe")
        if not model or not model.get("qualified"):
            continue
        denominator = model["train_positive_peak_median"]
        if denominator is None or denominator <= 0:
            continue
        values = (matrix[:, entry["feature_idx"]].toarray().ravel() if arm == "motifae"
                  else _probe_scores(h33, model))
        peak = float(values.max())
        candidates.append({"entry": entry, "values": values,
                           "threshold": model["threshold"], "score": peak / denominator})
    return sorted(candidates, key=lambda r: (-r["score"], r["entry"]["class_id"]))


def build_card(sample_id: str, sequence: str, h33: np.ndarray,
               matrix: sparse.csr_matrix, dictionary: dict, arm: str = "motifae") -> dict:
    """Inference-only interface intentionally cannot receive truth annotations."""
    if arm not in {"motifae", "raw_probe"}:
        raise ValueError("Unknown feature card arm")
    if len(h33) != len(sequence) or matrix.shape[0] != len(sequence):
        raise ValueError("Sequence/features length mismatch")
    associations = []
    for candidate in score_candidates(h33, matrix, dictionary, arm):
        region = peak_region(candidate["values"], candidate["threshold"])
        if region is None:
            continue
        start, end, peak_position, peak = region
        entry = candidate["entry"]
        associations.append({
            "class_id": entry["class_id"], "description": entry["description"],
            "feature_id": entry["feature_idx"] if arm == "motifae" else "raw_h33_ridge:" + entry["class_id"],
            "start": start, "end": end, "peak_position": peak_position,
            "peak": peak, "threshold": candidate["threshold"], "normalized_score": candidate["score"],
            "association_evidence": "TRAIN/qualification frozen dictionary",
            "activity_claim": "not established",
        })
        if len(associations) >= dictionary["method"]["top_k_associations"]:
            break
    named_ids = {entry["feature_idx"] for entry in dictionary["classes"] if entry["qualified"]}
    # Request more than 32 initially so the first 32 unknown features survive
    # removal of named associations, even when no class passes qualification.
    # Raw-H33 control must not expose MotifAE activations, even unnamed ones.
    top = feature_summary(matrix, 32 + len(named_ids)) if arm == "motifae" else []
    unknown = [item for item in top if item["feature_id"] not in named_ids][:32]
    return {"sample_id": sample_id, "sequence_length": len(sequence), "arm": arm,
            "coordinate_system": "1-based-inclusive", "associations": associations,
            "unknown_active_top_features": unknown,
            "activity_claim": "not established"}


def regex_spans(sequence: str, pattern: str) -> list[tuple[int, int]]:
    # Standard finditer semantics, including original ^/$ anchors, are fixed.
    return [(match.start() + 1, match.end()) for match in re.finditer(pattern, sequence)
            if match.end() > match.start()]


def build_regex_card(sample_id: str, sequence: str, dictionary: dict) -> dict:
    associations = []
    for entry in dictionary["classes"]:
        for start, end in regex_spans(sequence, entry["regex"]):
            associations.append({"class_id": entry["class_id"],
                                 "description": entry["description"], "start": start, "end": end,
                                 "feature_id": "catalog_regex:" + entry["class_id"],
                                 "peak_position": None, "peak": None, "threshold": None,
                                 "normalized_score": None,
                                 "association_evidence": "unqualified catalog regex pattern match",
                                 "activity_claim": "not established"})
    return {"sample_id": sample_id, "sequence_length": len(sequence), "arm": "regex",
            "coordinate_system": "1-based-inclusive", "associations": associations,
            "unknown_active_top_features": [], "activity_claim": "not established"}


def _macro_parent(rows: list[dict], metric: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None:
            groups[row["parent_accession"]].append(value)
    parent_means = [np.mean(x) for x in groups.values()]
    return {"macro_parent": mean_or_none(parent_means), "defined_parents": len(groups),
            "defined_rows": sum(len(x) for x in groups.values()), "total_rows": len(rows)}


def evaluate_test(rows: list[dict], store: FeatureStore, dictionary: dict) -> tuple[dict, dict]:
    """Only call after dictionary.json and its SHA sidecar are durably written."""
    tests = [row for row in rows if row["split"] == "test"]
    by_class = {entry["class_id"]: entry for entry in dictionary["classes"]}
    full_spans = {(cl, group["parent_accession"]): group["spans"]
                  for cl in by_class for group in class_parents(tests, cl)}
    cards = {arm: [] for arm in ("motifae", "raw_probe", "regex")}
    per_row = []
    for row in tests:
        hshort, short = store.load("short", row["short_sequence"])
        hfull, full = store.load("full", row["parent_sequence"])
        entry = by_class[row["class_id"]]
        result = {"sample_id": row["sample_id"], "parent_accession": row["parent_accession"],
                  "anchor_class_id": row["class_id"], "isolated_peptide_activity_validated": False}
        # Annotation-free cards are constructed before consulting TEST labels.
        for arm in ("motifae", "raw_probe"):
            cards[arm].append(build_card(row["sample_id"], row["short_sequence"], hshort, short,
                                         dictionary, arm))
        cards["regex"].append(build_regex_card(row["sample_id"], row["short_sequence"], dictionary))
        positives = (set(row.get("known_positive_class_ids", []))
                     | {item["class_id"] for item in row.get("known_positive_spans", []) if item.get("fully_contained")}
                     | {row["class_id"]})
        for arm in ("motifae", "raw_probe"):
            model = entry if arm == "motifae" else entry.get("raw_probe")
            qualified_ids = {e["class_id"] for e in dictionary["classes"]
                             if (e if arm == "motifae" else e.get("raw_probe") or {}).get("qualified")}
            candidates = score_candidates(hshort, short, dictionary, arm)
            ranked = [c["entry"]["class_id"] for c in candidates]
            card_ranked = [a["class_id"] for a in cards[arm][-1]["associations"]]
            result[arm + "_covered_anchor"] = row["class_id"] in qualified_ids
            result[arm + "_covered_any_known_positive"] = bool(positives & qualified_ids)
            for k in (1, 3):
                result[f"{arm}_ranking_top{k}_positive_hit"] = bool(positives & set(ranked[:k]))
                result[f"{arm}_ranking_top{k}_known_positive_recall"] = len(positives & set(ranked[:k])) / len(positives)
                result[f"{arm}_card_top{k}_positive_hit"] = bool(positives & set(card_ranked[:k]))
            for kind, h33, matrix in (("full", hfull, full), ("short", hshort, short)):
                known = (full_spans[(row["class_id"], row["parent_accession"])] if kind == "full"
                         else row_spans(row, row["class_id"], kind))
                mask = spans_mask(len(h33), known)
                values = None
                if model and entry["feature_idx"] is not None:
                    values = (matrix[:, entry["feature_idx"]].toarray().ravel() if arm == "motifae"
                              else _probe_scores(h33, model))
                result[f"{arm}_{kind}_anchor_auc"] = dense_auc(values, mask) if values is not None else None
                result[f"{arm}_{kind}_anchor_argmax_hit"] = (bool(mask[int(np.argmax(values))])
                                                                           if values is not None and np.any(values != values[0]) else None)
                result[f"{arm}_{kind}_qualified_anchor_auc"] = (result[f"{arm}_{kind}_anchor_auc"]
                                                                                if model and model["qualified"] else None)
        full_top = {item["feature_id"] for item in feature_summary(full, 32)}
        short_top = {item["feature_id"] for item in feature_summary(short, 32)}
        union = full_top | short_top
        result["top32_full_short_jaccard"] = len(full_top & short_top) / len(union) if union else None
        window_start, window_end = row["short_parent_start"], row["short_parent_end"]
        window_top = {item["feature_id"] for item in feature_summary(full[window_start - 1:window_end], 32)}
        window_union = window_top | short_top
        result["top32_context_window_short_jaccard"] = (len(window_top & short_top) / len(window_union)
                                                        if window_union else None)
        if entry["feature_idx"] is not None:
            feature = entry["feature_idx"]
            full_values = full[:, feature].toarray().ravel()
            start, end = row["short_parent_start"], row["short_parent_end"]
            context_peak = float(full_values[start - 1:end].max())
            short_peak = float(short[:, feature].max())
            result["anchor_feature_context_window_peak"] = context_peak
            result["anchor_feature_short_peak"] = short_peak
            result["anchor_feature_short_context_ratio"] = short_peak / context_peak if context_peak > 0 else None
        for kind, sequence in (("full", row["parent_sequence"]), ("short", row["short_sequence"])):
            spans = regex_spans(sequence, entry["regex"])
            known = (full_spans[(row["class_id"], row["parent_accession"])] if kind == "full"
                     else row_spans(row, row["class_id"], kind))
            mask = spans_mask(len(sequence), known)
            match_mask = spans_mask(len(sequence), spans)
            result[f"regex_{kind}_match_count"] = len(spans)
            result[f"regex_{kind}_known_positive_residue_recall"] = float(match_mask[mask].mean()) if mask.any() else None
            result[f"regex_{kind}_known_positive_span_recovered"] = sum(
                any(a <= stop and b >= begin for a, b in spans) for begin, stop in known)
            result[f"regex_{kind}_known_positive_span_count"] = len(known)
            result[f"regex_{kind}_known_positive_overlap"] = bool(np.any(match_mask & mask))
        per_row.append(result)
    metric_names = sorted({key for row in per_row for key in row
                           if key not in {"sample_id", "parent_accession", "anchor_class_id",
                                          "isolated_peptide_activity_validated"}})
    summaries = {metric: _macro_parent(per_row, metric) for metric in metric_names}
    for arm in ("motifae", "raw_probe"):
        for k in (1, 3):
            metric = f"{arm}_ranking_top{k}_positive_hit"
            summaries[metric]["all_row_mean_including_unsupported"] = mean_or_none([r[metric] for r in per_row])
            covered = [r[metric] for r in per_row if r[arm + "_covered_any_known_positive"]]
            summaries[metric]["covered_row_mean"] = mean_or_none(covered)
            summaries[metric]["covered_rows"] = len(covered)
    supported = {arm: [e["class_id"] for e in dictionary["classes"]
                       if (e if arm == "motifae" else e.get("raw_probe") or {}).get("qualified")]
                 for arm in ("motifae", "raw_probe")}
    report = {"schema_version": 1, "test_rows": len(tests), "test_parents": len({r["parent_accession"] for r in tests}),
              "qualified_classes": supported, "metrics": summaries, "private_per_row": per_row,
              "association_scope": ASSOCIATION_SCOPE, "activity_claim": "not established",
              "retrieval_semantics": "known-positive recovery among qualified candidates; unlabeled classes are not experimentally negative",
              "ranking_semantics": "max short activation / TRAIN annotated peak median; all qualified candidates ranked before thresholding",
              "card_semantics": "only ranked candidates exceeding frozen TRAIN unannotated threshold, up to top3",
              "macro_parent_semantics": "average rows within parent then equal-weight mean across parents with defined value",
              "raw_control_scope": "same frozen H33 backbone; independent qualification gate under identical fixed rule",
              "regex_semantics": "original catalog finditer including anchors; sequence-pattern match, not verified function",
              "null_semantics": "undefined AUROC, absent TRAIN feature, constant localization argmax, or zero context denominator remain null",
              "limitations": ["shorts are excised parent regions without validated isolated-peptide activity",
                              "one fixed split; this feature-localization pilot does not establish cross-seed LLM gains",
                              "class coverage is preserved; unsupported classes contribute zero to overall retrieval hits",
                              "full-parent selected-feature AUROC includes all known same-class spans in each sample row"],
              "cache_file_receipts": dict(sorted(store.read_receipts.items()))}
    return report, cards


def run(scope: Path, phase: str = "all") -> dict:
    scope = scope.resolve()
    config = load_config(scope)
    cache_root = Path(config["cache_root"])
    if not cache_root.is_absolute():
        cache_root = scope / cache_root
    store = FeatureStore(cache_root)
    rows = jsonl(scope / "data/panel.jsonl")
    catalog = json.loads((scope / "data/catalog.json").read_text(encoding="utf-8"))["classes"]
    dictionary_path = scope / "dictionary.json"
    if phase in {"all", "fit"}:
        dictionary = fit_dictionary(rows, catalog, store, config["dictionary"])
        frozen_write(dictionary_path, json_text(dictionary))
        frozen_write(scope / "dictionary.sha256", file_sha(dictionary_path) + "\n")
    else:
        dictionary = json.loads(dictionary_path.read_text(encoding="utf-8"))
    expected_sha = (scope / "dictionary.sha256").read_text(encoding="utf-8").strip()
    if file_sha(dictionary_path) != expected_sha:
        raise RuntimeError("Frozen dictionary fingerprint mismatch")
    if dictionary["method"] != config["dictionary"]:
        raise RuntimeError("Configuration differs from frozen dictionary")
    for split in ("train", "qualification"):
        expected = dictionary["source_pins"][f"{split}_rows_sha256"]
        observed = sha256_bytes(json_text([r for r in rows if r["split"] == split]).encode())
        if observed != expected:
            raise RuntimeError(f"Frozen dictionary {split} input fingerprint mismatch")
    if sha256_bytes(json_text(catalog).encode()) != dictionary["source_pins"]["catalog_sha256"]:
        raise RuntimeError("Frozen candidate catalog fingerprint mismatch")
    if phase in {"all", "evaluate"}:
        report, cards = evaluate_test(rows, store, dictionary)
        report["dictionary_sha256"] = expected_sha
        for arm, values in cards.items():
            frozen_write(scope / "data" / f"cards_{arm}.jsonl", "".join(
                json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for value in values))
        frozen_write(scope / "analysis/feature_metrics.json", json_text(report))
        return {"dictionary_sha256": expected_sha, "test_rows": report["test_rows"],
                "qualified_classes": report["qualified_classes"]}
    return {"dictionary_sha256": expected_sha,
            "qualified_classes": [e["class_id"] for e in dictionary["classes"] if e["qualified"]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--phase", choices=("fit", "evaluate", "all"), default="all")
    args = parser.parse_args()
    print(json_text(run(args.scope, args.phase)))


if __name__ == "__main__":
    main()

"""Pinned FLIP human thermal-stability baseline; CPU only, no added packages.

Run from the benchmark root:
  python src/meltome.py audit --config configs/meltome_human_v1.json
  python src/meltome.py run --config configs/meltome_human_v1.json --run-id baseline_v1
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
FEATURES = ["sequence_length"] + [f"fraction_{aa}" for aa in AA] + ["fraction_other"]


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["sequence", "target", "set", "validation"]:
            raise ValueError("Unexpected official CSV schema")
        rows = []
        for index, raw in enumerate(reader):
            seq = raw["sequence"]
            if not seq or not seq.isascii() or not seq.isalpha() or seq != seq.upper():
                raise ValueError(f"Invalid sequence at official row {index}")
            if raw["set"] not in {"train", "test"} or raw["validation"] not in {"True", "False"}:
                raise ValueError(f"Invalid split at official row {index}")
            if raw["set"] == "test" and raw["validation"] == "True":
                raise ValueError(f"Test row marked validation: {index}")
            target = float(raw["target"])
            if not math.isfinite(target):
                raise ValueError(f"Non-finite target at official row {index}")
            rows.append({"official_row_index": index,
                         "row_id": f"human:{index}:{hashlib.sha256(seq.encode()).hexdigest()[:16]}",
                         "sequence": seq, "target": target,
                         "split": "validation" if raw["validation"] == "True" else raw["set"]})
    if not rows:
        raise ValueError("Empty official dataset")
    return rows


def audit_rows(rows):
    splits = collections.Counter(row["split"] for row in rows)
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("All three nonempty official splits are required")
    sequence_splits = collections.defaultdict(set)
    within = collections.Counter()
    noncanonical = collections.Counter()
    for row in rows:
        sequence_splits[row["sequence"]].add(row["split"])
        within[(row["split"], row["sequence"])] += 1
        noncanonical.update(char for char in row["sequence"] if char not in AA)
    cross = [seq for seq, names in sequence_splits.items() if len(names) > 1]
    if cross:
        raise ValueError(f"Blocking exact-sequence cross-split leakage: {len(cross)} sequences")
    return {"status": "passed", "counts": dict(splits), "total_rows": len(rows),
            "unique_sequences": len(sequence_splits), "exact_sequence_cross_split_duplicates": 0,
            "within_split_extra_duplicate_rows": {s: sum(n - 1 for (sp, _), n in within.items() if sp == s) for s in splits},
            "independent_exact_sequence_groups": {s: sum(1 for sp, _ in within if sp == s) for s in splits},
            "dependency_note": "Same-sequence repeated rows may represent correlated experimental observations. Row metrics preserve the official denominator; group-macro metrics weight exact-sequence groups equally.",
            "noncanonical_residue_counts": dict(noncanonical),
            "duplicates_policy": "Preserve all official rows, including same-split repeated sequences; no filtering.",
            "identifier_policy": "CSV has no original protein IDs; stable IDs use zero-based official row index and sequence SHA256 prefix."}


def verify_inputs(config, root):
    for relative, expected in config["files_sha256"].items():
        actual = sha(root / relative)
        if actual != expected:
            raise ValueError(f"Frozen input changed: {relative}")


def dataset_audit(config, root):
    verify_inputs(config, root)
    rows = read_rows(root / config["dataset"])
    report = audit_rows(rows)
    if report["counts"] != config["expected_counts"]:
        raise ValueError("Official split counts changed")
    representatives = set()
    components = set()
    with (root / config["cluster_metadata"]).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            representatives.add(row["cluster_representative"])
            components.add(row["cluster_component"])
    report["cluster_metadata"] = {
        "representatives": len(representatives), "unique_components": len(components),
        "source": config["cluster_metadata"], "declared_mmseqs_min_seq_id": 0.2,
        "human_csv_row_cluster_membership_verified": False,
        "reason": "The official CSV and derived FASTA omit original protein IDs used by the clustering table.",
        "interpretation": "Official cluster-derived split preserved. A 20% clustering threshold is not a guarantee of pairwise identity <=20%."}
    report["test_targets_usage"] = "Integrity checks only (finite values/schema); no test metric or selection before the frozen model."
    return rows, report


def features(rows):
    result = np.empty((len(rows), len(FEATURES)), dtype=np.float64)
    for i, row in enumerate(rows):
        sequence = row["sequence"]
        counts = collections.Counter(sequence)
        result[i] = [len(sequence)] + [counts[a] / len(sequence) for a in AA] + [sum(n for a, n in counts.items() if a not in AA) / len(sequence)]
    return result


def fit_standardizer(train):
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0, ddof=0)
    scale[scale == 0] = 1.0
    return mean, scale


def rank_average(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(truth, prediction):
    a = rank_average(truth)
    b = rank_average(prediction)
    a -= a.mean()
    b -= b.mean()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / norm) if norm else None


def score(truth, prediction, expected_n):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != (expected_n,) or prediction.shape != truth.shape or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Scoring requires the complete finite official denominator")
    error = prediction - truth
    return {"n": expected_n, "prediction_coverage": 1.0, "spearman": spearman(truth, prediction),
            "mae_degrees_C": float(np.mean(np.abs(error))), "rmse_degrees_C": float(np.sqrt(np.mean(error ** 2)))}


def group_score(rows, predictions):
    """Equal sequence-group weighting of row errors; Spearman on group means."""
    if len(rows) != len(predictions) or not np.isfinite(predictions).all():
        raise ValueError("Group scoring requires complete finite predictions")
    groups = collections.defaultdict(list)
    for row, prediction in zip(rows, predictions):
        groups[row["sequence"]].append((row["target"], float(prediction)))
    mae, mse, targets, preds = [], [], [], []
    for members in groups.values():
        truth, prediction = np.array(members).T
        mae.append(float(np.mean(np.abs(prediction - truth))))
        mse.append(float(np.mean((prediction - truth) ** 2)))
        targets.append(float(truth.mean()))
        preds.append(float(prediction.mean()))
    return {"n_rows": len(rows), "n_exact_sequence_groups": len(groups),
            "group_macro_mae_degrees_C": float(np.mean(mae)),
            "group_macro_rmse_degrees_C": float(np.sqrt(np.mean(mse))),
            "group_mean_spearman": spearman(targets, preds),
            "definition": "MAE/MSE averaged within sequence then equally across groups; Spearman correlates group-mean target and prediction."}


def fit_and_select(train_rows, validation_rows, alphas):
    if any(row["split"] != "train" for row in train_rows) or any(row["split"] != "validation" for row in validation_rows):
        raise ValueError("Model fitting accepts TRAIN only and selection accepts VALIDATION only")
    if not train_rows or not validation_rows:
        raise ValueError("Nonempty fitting and selection splits required")
    if set(r["sequence"] for r in train_rows) & set(r["sequence"] for r in validation_rows):
        raise ValueError("Training/validation exact-sequence leakage")
    if not alphas or any(not math.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("Positive finite ridge alphas required")
    x_train = features(train_rows)
    x_val = features(validation_rows)
    y_train = np.array([row["target"] for row in train_rows])
    y_val = np.array([row["target"] for row in validation_rows])
    mean, scale = fit_standardizer(x_train)
    train_z = (x_train - mean) / scale
    val_z = (x_val - mean) / scale
    intercept = float(y_train.mean())
    centered = y_train - intercept
    gram = train_z.T @ train_z
    rhs = train_z.T @ centered
    selection = []
    best_value = -math.inf
    best = None
    for alpha in alphas:
        coef = np.linalg.solve(gram + alpha * np.eye(train_z.shape[1]), rhs)
        metrics = score(y_val, val_z @ coef + intercept, len(validation_rows))
        selection.append({"alpha": alpha, "validation": metrics})
        value = metrics["spearman"]
        comparable = value if value is not None else -math.inf
        if best is None or comparable > best_value:
            best = (alpha, coef.copy())
            best_value = comparable
    return {"feature_names": FEATURES, "standardization_mean": mean.tolist(), "standardization_scale": scale.tolist(),
            "intercept": intercept, "alpha": best[0], "coefficients": best[1].tolist(),
            "train_mean": intercept, "fitted_rows": len(train_rows), "selection": selection,
            "selection_rule": "Maximum complete-validation Spearman; exact ties choose first registered alpha.",
            "refit_on_train_plus_validation": False}


def predict(model, rows):
    x = features(rows)
    return ((x - np.array(model["standardization_mean"])) / np.array(model["standardization_scale"])) @ np.array(model["coefficients"]) + model["intercept"]


def run(config_path, run_id):
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("run-id must be a single safe directory name")
    output = root / "results" / "meltome_human_v1" / run_id
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "started", "started_utc": utc(), "config_sha256": sha(config_path),
               "source_sha256": sha(__file__), "python_version": sys.version, "numpy_version": np.__version__,
               "seed_policy": "Deterministic closed-form baseline; no fabricated multi-seed replication.",
               "task": "Independent single-sequence thermal-stability infrastructure baseline"}
    shutil.copyfile(config_path, output / "frozen_config.json")
    shutil.copyfile(__file__, output / "source_meltome.py")
    write_json(output / "configuration_frozen.json", {"frozen_utc": utc(), "sha256": receipt["config_sha256"]})
    try:
        rows, audit = dataset_audit(config, root)
        write_json(output / "data_audit.json", audit)
        train = [row for row in rows if row["split"] == "train"]
        validation = [row for row in rows if row["split"] == "validation"]
        model = fit_and_select(train, validation, config["ridge_alphas"])
        write_json(output / "model.json", model)
        write_json(output / "selection.json", model["selection"])
        write_json(output / "model_frozen_before_test.json", {"frozen_utc": utc(), "model_sha256": sha(output / "model.json"), "test_metrics_read": False})
        test = [row for row in rows if row["split"] == "test"]
        metrics = {}
        with (output / "predictions.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["row_id", "official_row_index", "sequence_sha256", "split", "target", "train_mean_prediction", "composition_ridge_prediction"])
            writer.writeheader()
            for split, subset in [("train", train), ("validation", validation), ("test", test)]:
                ridge = predict(model, subset)
                mean = np.full(len(subset), model["train_mean"])
                target = [row["target"] for row in subset]
                metrics[split] = {"train_mean": score(target, mean, config["expected_counts"][split]),
                                  "composition_ridge": score(target, ridge, config["expected_counts"][split]),
                                  "supplementary_group_metrics": {"train_mean": group_score(subset, mean), "composition_ridge": group_score(subset, ridge)}}
                for row, m, r in zip(subset, mean, ridge):
                    writer.writerow({"row_id": row["row_id"], "official_row_index": row["official_row_index"],
                                     "sequence_sha256": hashlib.sha256(row["sequence"].encode()).hexdigest(),
                                     "split": split, "target": row["target"], "train_mean_prediction": m, "composition_ridge_prediction": r})
        write_json(output / "metrics.json", metrics)
        receipt.update(status="complete", completed_utc=utc(), total_predictions=len(rows), test_predictions=len(test),
                       files={p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()},
                       dataset_sha256=config["files_sha256"], selected_alpha=model["alpha"])
        write_json(output / "receipt.json", receipt)
        print(json.dumps({"output": str(output), "status": "complete", "selected_alpha": model["alpha"], "test": metrics["test"]}, indent=2))
        return output
    except Exception as exc:
        receipt.update(status="blocked", completed_utc=utc(), error_type=type(exc).__name__, error=str(exc))
        write_json(output / "receipt.json", receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "run"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.command == "run":
        if not args.run_id:
            parser.error("run requires --run-id (existing result directories are never overwritten)")
        run(args.config, args.run_id)
    else:
        config_path = Path(args.config).resolve()
        _, audit = dataset_audit(json.loads(config_path.read_text(encoding="utf-8")), config_path.parent.parent)
        print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

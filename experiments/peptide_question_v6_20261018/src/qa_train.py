"""Frozen-protocol shared-decoder peptide question experiment.

Only the adapter is optimized. Selection uses validation; TEST is first opened
after the selected checkpoint and its immutable digest receipt are persisted.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np

QUESTIONS = ("rt", "ccs", "both")
TEMPLATES = {
    "rt": (
        "Is this peptide's retention time greater than {rt:.10g} minutes?",
        "Does the retention time of this peptide exceed {rt:.10g} minutes?",
        "For this peptide, is the measured retention time above {rt:.10g} minutes?"),
    "ccs": (
        "Is this peptide's collision cross section greater than {ccs:.10g} square angstroms?",
        "Does the collision cross section of this peptide exceed {ccs:.10g} square angstroms?",
        "For this peptide, is the measured collision cross section above {ccs:.10g} square angstroms?"),
    "both": (
        "Are both the retention time greater than {rt:.10g} minutes and the collision cross section greater than {ccs:.10g} square angstroms?",
        "Does this peptide exceed both thresholds: retention time {rt:.10g} minutes and collision cross section {ccs:.10g} square angstroms?",
        "For this peptide, are the measured retention time above {rt:.10g} minutes AND the measured collision cross section above {ccs:.10g} square angstroms?")}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def resolve(scope, value):
    path = Path(value)
    return path if path.is_absolute() else Path(scope) / path


def verify_frozen(scope, config_path, config):
    """Verify all frozen files, including future TEST files, without parsing TEST."""
    manifest_path = resolve(scope, config.get("frozen_manifest", "frozen_manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_sha256", sha256(config_path)) != sha256(config_path):
        raise ValueError("Frozen configuration digest differs")
    files = manifest.get("files", {})
    if isinstance(files, list):
        files = {item["path"]: item["sha256"] for item in files}
    if not files:
        raise ValueError("Frozen manifest must pin data, assets, configuration and source")
    checked = {}
    for name, expected in files.items():
        actual = sha256(resolve(scope, name))
        if actual != expected:
            raise ValueError(f"Frozen file changed: {name}")
        checked[str(resolve(scope, name).resolve())] = actual
    required = [config_path, Path(__file__), Path(__file__).with_name("qa_model.py"),
                Path(__file__).with_name("qa_metrics.py")]
    data_root = resolve(scope, config.get("data_root", "data"))
    required += [data_root / (split + ".jsonl") for split in ("train", "validation", "test")]
    required += [resolve(scope, config.get("features_path", "cache/features.npz")),
                 resolve(scope, config.get("attributes_path", "data/attributes.npz")),
                 resolve(scope, config.get("control_mapping_path", "data/control_mapping.json"))]
    for path in required:
        if str(Path(path).resolve()) not in checked:
            raise ValueError(f"Required scientific input is not frozen: {path.name}")
    return {"manifest_sha256": sha256(manifest_path), "files_verified": len(checked),
            "config_sha256": sha256(config_path)}


def template_index(peptide_id, question, epoch):
    key = f"{peptide_id}\0{question}\0{epoch}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 2


def experiment_key(row):
    return json.dumps(row.get("experiment", "unspecified"), sort_keys=True, separators=(",", ":"))


def make_prompt(row, question, thresholds, arm, attributes=None, epoch=None,
                method_context="Fixed experimental method from the frozen dataset.",
                molecular_row=None):
    if question not in QUESTIONS:
        raise ValueError("Unknown question")
    index = 2 if epoch is None else template_index(row["id"], question, epoch)
    experiment = row.get("experiment", "unspecified")
    # Only explicitly sanctioned common context enters the prompt. No source ID,
    # split, group, raw sequence, measured property or other row field is copied.
    context = f"Experiment context: {json.dumps(experiment, sort_keys=True)}. Charge: 2. {method_context}"
    if arm == "TEXT" and attributes is not None:
        rt, ccs = (float(value) for value in attributes)
        if not all(math.isfinite(value) for value in (rt, ccs)):
            raise ValueError("Nonfinite predicted attributes")
        molecular = (f"Specialist-predicted retention time: {rt:.10g} minutes. "
                     f"Specialist-predicted collision cross section: {ccs:.10g} square angstroms.")
        sequence = (molecular_row or row)["sequence"]
        if not sequence:
            raise ValueError("Empty sequence")
        composition = ", ".join(f"{aa}:{sequence.count(aa) / len(sequence):.3f}"
                                for aa in "ACDEFGHIKLMNPQRSTVWY")
        molecular += f" Sequence length: {len(sequence)} residues. Amino acid fractions: {composition}."
    else:
        molecular = "Molecular property estimates unavailable."
    question_text = TEMPLATES[question][index].format(**thresholds)
    return (f"{context}\n{molecular}\nQuestion key: {question}.\n{question_text}\n"
            "Answer with exactly one lowercase word: yes or no.")


def build_examples(rows, seed, epoch=None):
    examples = [(row, question) for row in rows for question in QUESTIONS]
    if epoch is not None:
        random.Random(f"{seed}:epoch:{epoch}").shuffle(examples)
    return examples


class Assets:
    def __init__(self, scope, config, train_rows):
        features = np.load(resolve(scope, config.get("features_path", "cache/features.npz")), allow_pickle=False)
        attributes = np.load(resolve(scope, config.get("attributes_path", "data/attributes.npz")), allow_pickle=False)
        self.ids = features["ids"].astype(str).tolist()
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Duplicate feature ids")
        self.index = {ident: i for i, ident in enumerate(self.ids)}
        if attributes["ids"].astype(str).tolist() != self.ids:
            raise ValueError("Attribute and feature identity order differs")
        self.p15 = np.asarray(features["p15"], dtype=np.float32)
        self.p30 = np.asarray(features["p30"], dtype=np.float32)
        self.attrs = np.asarray(attributes["predictions"], dtype=np.float64)
        self.splits = attributes["split"].astype(str).tolist()
        self.control_mapping = json.loads(resolve(scope, config.get(
            "control_mapping_path", "data/control_mapping.json")).read_text(encoding="utf-8"))
        count = len(self.ids)
        if (self.p15.shape != (count, 8, 640) or self.p30.shape != (count, 8, 640)
                or self.attrs.shape != (count, 2) or len(self.splits) != count):
            raise ValueError("Asset dimensions differ from frozen contract")
        if not all(np.isfinite(value).all() for value in (self.p15, self.p30, self.attrs)):
            raise ValueError("Asset contains nonfinite values")
        train_indices = [self.index[row["id"]] for row in train_rows]
        if not train_indices or any(self.splits[i] != "train" for i in train_indices):
            raise ValueError("TRAIN memory mean must use only TRAIN identities")
        self.mean = {"p15": self.p15[train_indices].mean(axis=0, dtype=np.float64).astype(np.float32),
                     "p30": self.p30[train_indices].mean(axis=0, dtype=np.float64).astype(np.float32)}

    def memory(self, ids, mode, arm, torch, device):
        if mode == "mean" or arm == "TEXT":
            values = {key: np.repeat(value[None], len(ids), axis=0) for key, value in self.mean.items()}
        else:
            indices = [self.index[ident] for ident in ids]
            values = {"p15": self.p15[indices], "p30": self.p30[indices]}
        return {key: torch.as_tensor(value, dtype=torch.float32, device=device).detach()
                for key, value in values.items()}

    def attributes(self, ident):
        return self.attrs[self.index[ident]]

    def donors(self, rows):
        result = {}
        for row in rows:
            split = row["split"]
            if self.splits[self.index[row["id"]]] != split:
                raise ValueError("Attribute split differs from row split")
            donor = self.control_mapping[split][row["id"]]
            if donor == row["id"] or self.splits[self.index[donor]] != split:
                raise ValueError("Control mapping must derange within split")
            result[row["id"]] = donor
        if len(set(result.values())) != len(result):
            raise ValueError("Control donors must form a permutation")
        if set(result) != set(result.values()):
            raise ValueError("Control donors must cover the same evaluation rows")
        return result


def encode_prompt(tokenizer, prompt):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    return list(ids)


def left_pad(sequences, pad_id, torch, device):
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, sequence in enumerate(sequences):
        ids[i, width - len(sequence):] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[i, width - len(sequence):] = 1
    return ids, mask


def greedy_generate(model, tokenizer, prompts, memories, device, torch,
                    max_new_tokens=4, off=False):
    sequences = [encode_prompt(tokenizer, prompt) for prompt in prompts]
    ids, mask = left_pad(sequences, tokenizer.pad_token_id, torch, device)
    ends = torch.full((len(prompts),), ids.shape[1] - 1, dtype=torch.long, device=device)
    done = [False] * len(prompts)
    outputs = [[] for _ in prompts]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            result = model(ids, mask, ends, memories, off=off)
            tokens = result["logits"][:, -1].argmax(-1)
            for i, token in enumerate(tokens.tolist()):
                if not done[i]:
                    if token == tokenizer.eos_token_id:
                        done[i] = True
                    else:
                        outputs[i].append(token)
            if all(done):
                break
            # Finished rows continue with EOS to preserve contiguous active text;
            # generated payload stops at their first EOS.
            for i, stopped in enumerate(done):
                if stopped:
                    tokens[i] = tokenizer.eos_token_id
            ids = torch.cat((ids, tokens[:, None]), dim=1)
            mask = torch.cat((mask, torch.ones_like(tokens[:, None])), dim=1)
    return [tokenizer.decode(tokens, skip_special_tokens=False,
                              clean_up_tokenization_spaces=False) for tokens in outputs]


def evaluate(model, tokenizer, rows, assets, config, arm, mode, torch, device,
             output_path=None, limit=None):
    from qa_metrics import score
    examples = build_examples(rows, 0)
    if limit is not None:
        examples = examples[:limit]
    mapping = assets.donors(rows)
    rows_by_id = {row["id"]: row for row in rows}
    batch_size = int(config.get("training", {}).get("generation_batch_size", 8))
    records = []
    started = time.monotonic()
    model.eval()
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        ids, prompts = [], []
        for row, question in batch:
            donor = mapping[row["id"]] if mode == "shuffled" else row["id"]
            attrs = assets.attributes(donor) if arm == "TEXT" and mode != "removed" else None
            prompts.append(make_prompt(row, question, config["thresholds"], arm, attrs,
                                       method_context=config.get("method_context", "Fixed experimental method from the frozen dataset."),
                                       molecular_row=rows_by_id[donor]))
            ids.append(donor)
        memory = assets.memory(ids, mode, arm, torch, device)
        raw = greedy_generate(model, tokenizer, prompts, memory, device, torch,
                              config.get("training", {}).get("max_new_tokens", 4), off=mode == "off")
        for (row, question), answer, donor in zip(batch, raw, ids):
            records.append({"id": row["id"], "question": question, "raw": answer,
                            "condition": mode, "donor_id": donor})
    if output_path:
        write_jsonl(output_path, records)
    if limit is None:
        summary = score(rows, records, config["thresholds"])
    else:
        summary = {"train_only_generation_examples": len(records)}
    summary["wall_seconds"] = time.monotonic() - started
    summary["generated_rows"] = len(records)
    if mode == "shuffled":
        summary["shuffle_seed"] = config.get("shuffle_seed", 991)
        summary["frozen_mapping_used"] = True
    return summary


def load_model(config, arm, torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from qa_model import SharedMemoryLM
    path = config["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                              local_files_only=True, attn_implementation="sdpa")
    base.config.use_cache = False
    base.gradient_checkpointing_disable()
    model = SharedMemoryLM(base, variant=arm, hidden_dim=base.config.hidden_size,
                           **config.get("connector", {})).to("cuda")
    return model, tokenizer


def validate_contract(config, train, validation):
    training = config.get("training", {})
    if int(training.get("epochs", 6)) != 6 or int(training.get("micro_batch_size", 1)) != 1:
        raise ValueError("Frozen experiment requires six epochs and microbatch one")
    if int(training.get("gradient_accumulation_steps", 8)) != 8:
        raise ValueError("Frozen experiment requires effective batch eight")
    if len({row["id"] for row in train + validation}) != len(train) + len(validation):
        raise ValueError("Duplicate or overlapping TRAIN/validation identities")
    for q in ("rt", "ccs"):
        median = float(np.median([float(row[q]) for row in train]))
        if not math.isclose(median, float(config["thresholds"][q]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{q} threshold differs from TRAIN median")
    if {row.get("group_id", row["id"]) for row in train} & {row.get("group_id", row["id"]) for row in validation}:
        raise ValueError("TRAIN/validation peptide groups overlap")


def run(scope, arm, seed, phase, config_path=None):
    import torch
    import torch.nn.functional as functional
    from qa_metrics import label
    scope = Path(scope).resolve()
    config_path = resolve(scope, config_path or "config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if arm not in ("M0", "M3", "TEXT") or seed not in config.get("seeds", [17, 42, 2026]):
        raise ValueError("Unknown frozen arm or seed")
    run_dir = resolve(scope, config.get("runs_root", "runs")) / f"{arm}__seed{seed}" / phase
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        manifest_path = resolve(scope, config.get("frozen_manifest", "frozen_manifest.json"))
        frozen = (verify_frozen(scope, config_path, config) if phase == "formal" or manifest_path.exists()
                  else {"phase": "train_only_technical_smoke", "config_sha256": sha256(config_path),
                        "trainer_sha256": sha256(__file__)})
        atomic_json(run_dir / "progress.json", {"phase": "loading", "arm": arm, "seed": seed})
        data_root = resolve(scope, config.get("data_root", "data"))
        train = read_jsonl(data_root / "train.jsonl")
        validation = read_jsonl(data_root / "validation.jsonl") if phase == "formal" else []
        validate_contract(config, train, validation)
        assets = Assets(scope, config, train)
        if config.get("model_asset_receipt"):
            model_assets = json.loads(resolve(scope, config["model_asset_receipt"]).read_text())
            for key, metadata in model_assets.items():
                for entry in metadata["files"]:
                    path = Path(config[key]) / entry["file"]
                    stat = path.stat()
                    if stat.st_size != entry["bytes"] or stat.st_mtime_ns != entry["mtime_ns"]:
                        raise ValueError("External model asset changed since full-hash qualification: " + entry["file"])
            if model_assets["model_path"]["revision"] != config["model_revision"]:
                raise ValueError("Qualified Qwen revision differs")
        torch.set_num_threads(2)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        model, tokenizer = load_model(config, arm, torch)
        device = next(model.parameters()).device
        training = config.get("training", {})
        own = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(own, lr=float(training.get("learning_rate", 3e-4)),
                                     weight_decay=float(training.get("weight_decay", .01)))
        capacity = model.capacity_report()
        atomic_json(run_dir / "run_contract.json", {"arm": arm, "seed": seed, "phase": phase,
                    "frozen": frozen, "capacity": capacity, "backbone_frozen": all(
                        not p.requires_grad for p in model.base_lm.parameters()),
                    "backbone": config["model_path"], "torch_version": torch.__version__,
                    "torch_initial_seed": torch.initial_seed(),
                    "train_memory_mean_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in assets.mean.items()}})
        accumulation = int(training.get("gradient_accumulation_steps", 8))
        updates = 0
        examples_seen = 0
        best = -math.inf
        best_epoch = None
        epochs = 1 if phase == "smoke" else 6
        history = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            examples = build_examples(train, seed, epoch)
            if phase == "smoke":
                examples = examples[:4 * accumulation]
                if len(examples) < 4 * accumulation:
                    raise ValueError("Smoke requires at least 32 TRAIN question examples")
            losses, gates, gradients = [], [], []
            for offset, (row, question) in enumerate(examples):
                attrs = assets.attributes(row["id"]) if arm == "TEXT" else None
                prompt = make_prompt(row, question, config["thresholds"], arm, attrs,
                                     epoch=epoch, method_context=config.get("method_context", "Fixed experimental method from the frozen dataset."))
                prompt_ids = encode_prompt(tokenizer, prompt)
                answer = label(row, question, config["thresholds"])
                answer_ids = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
                tokens = prompt_ids + answer_ids
                if len(tokens) > int(training.get("max_sequence_length", 1024)):
                    raise ValueError("Prompt exceeds frozen context budget; silent truncation forbidden")
                ids = torch.tensor([tokens], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                end = torch.tensor([len(prompt_ids) - 1], dtype=torch.long, device=device)
                memory = assets.memory([row["id"]], "normal", arm, torch, device)
                result = model(ids, mask, end, memory)
                answer_logits = result["logits"][:, len(prompt_ids) - 1:-1].float()
                targets = torch.tensor([answer_ids], dtype=torch.long, device=device)
                loss = functional.cross_entropy(answer_logits.reshape(-1, answer_logits.shape[-1]), targets.reshape(-1))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                (loss / accumulation).backward()
                losses.append(float(loss.detach()))
                gates.append(float(result["gate"].detach()))
                examples_seen += 1
                if (offset + 1) % accumulation == 0 or offset + 1 == len(examples):
                    tail = (offset + 1) % accumulation
                    if tail:
                        for parameter in own:
                            if parameter.grad is not None:
                                parameter.grad.mul_(accumulation / tail)
                    norm = torch.nn.utils.clip_grad_norm_(own, float(training.get("max_grad_norm", 1.0)), error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    gradients.append(float(norm))
                    if any(p.requires_grad or p.grad is not None for p in model.base_lm.parameters()):
                        raise ValueError("Frozen LM gained gradients")
                    atomic_json(run_dir / "progress.json", {"phase": "training", "arm": arm, "seed": seed,
                                "epoch_one_based": epoch + 1, "examples_seen": examples_seen,
                                "optimizer_updates": updates, "mean_ce": float(np.mean(losses)),
                                "gate": gates[-1], "last_grad_norm": gradients[-1],
                                "wall_seconds": time.monotonic() - started})
            log = {"epoch_one_based": epoch + 1, "mean_ce": float(np.mean(losses)),
                   "mean_gate": float(np.mean(gates)), "mean_grad_norm": float(np.mean(gradients)),
                   "optimizer_updates": updates, "examples_seen": examples_seen}
            if phase == "smoke":
                log["train_only_generation"] = evaluate(model, tokenizer, train, assets, config, arm,
                    "normal", torch, device, run_dir / "train_generations.jsonl", limit=8)
            else:
                metrics = evaluate(model, tokenizer, validation, assets, config, arm, "normal", torch,
                                   device, run_dir / f"validation_epoch_{epoch + 1:02d}.jsonl")
                log["validation"] = metrics
                if not math.isfinite(metrics["primary"]):
                    raise ValueError("Nonfinite validation primary metric")
                if metrics["primary"] > best:
                    best, best_epoch = metrics["primary"], epoch + 1
                    temporary = run_dir / "best_adapter.pt.tmp"
                    torch.save(model.trainable_state_dict(), temporary)
                    os.replace(temporary, run_dir / "best_adapter.pt")
            history.append(log)
            atomic_json(run_dir / "training_history.json", history)
        if phase == "formal":
            checkpoint = run_dir / "best_adapter.pt"
            selected = {"selected_epoch": best_epoch, "validation_primary": best,
                        "selection_rule": "max validation mean(BA_rt,BA_ccs); earliest exact tie",
                        "epochs_completed": 6, "checkpoint_sha256": sha256(checkpoint),
                        "optimizer_updates": updates, "training_examples_seen": examples_seen,
                        "test_opened": False, "frozen": frozen}
            atomic_json(run_dir / "selected_checkpoint.json", selected)
            # Opening TEST occurs strictly after the selected checkpoint receipt.
            test = read_jsonl(data_root / "test.jsonl")
            earlier_ids = {row["id"] for row in train + validation}
            earlier_groups = {row.get("group_id", row["id"]) for row in train + validation}
            if (len({row["id"] for row in test}) != len(test)
                    or earlier_ids & {row["id"] for row in test}
                    or earlier_groups & {row.get("group_id", row["id"]) for row in test}):
                raise ValueError("Frozen TEST has duplicate or overlapping peptides")
            model.load_trainable_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            if sha256(checkpoint) != selected["checkpoint_sha256"]:
                raise ValueError("Selected checkpoint changed before TEST")
            conditions = ("normal", "shuffled", "removed") if arm == "TEXT" else ("normal", "mean", "shuffled", "off")
            scores = {}
            for mode in conditions:
                atomic_json(run_dir / "progress.json", {"phase": "test", "condition": mode,
                            "arm": arm, "seed": seed, "checkpoint_sha256": selected["checkpoint_sha256"]})
                scores[mode] = evaluate(model, tokenizer, test, assets, config, arm, mode, torch,
                                        device, run_dir / f"test_{mode}.jsonl")
            atomic_json(run_dir / "test_summary.json", scores)
            if sha256(checkpoint) != selected["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed during TEST")
        artifacts = {}
        for path in sorted(run_dir.iterdir()):
            if path.suffix == ".jsonl" or path.name in ("selected_checkpoint.json", "best_adapter.pt", "training_history.json", "test_summary.json", "run_contract.json"):
                artifacts[path.name] = {"sha256": sha256(path)}
                if path.suffix == ".jsonl":
                    artifacts[path.name]["rows"] = len(read_jsonl(path))
        final = {"status": "complete", "arm": arm, "seed": seed, "phase": phase,
                 "epochs_completed": epochs, "optimizer_updates": updates,
                 "training_examples_seen": examples_seen, "wall_seconds": time.monotonic() - started,
                 "capacity": capacity, "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "max_gpu_reserved_bytes": torch.cuda.max_memory_reserved(), "frozen": frozen,
                 "artifacts": artifacts}
        atomic_json(run_dir / "completion.json", final)
        atomic_json(run_dir / "progress.json", final)
        return final
    except BaseException as error:
        atomic_json(run_dir / "failure.json", {"status": "failed", "exception": type(error).__name__,
                    "message": str(error), "traceback": traceback.format_exc(),
                    "wall_seconds": time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", required=True)
    parser.add_argument("--arm", choices=("M0", "M3", "TEXT"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    run(args.scope, args.arm, args.seed, args.phase, args.config)


if __name__ == "__main__":
    main()

"""Frozen ESM2 H33 -> published MotifAE features; sequence-only, private cache.

The sparse encoder follows CHAOHOU-97/MotifAE at
8dd6a7588d30726fef98c789dac3042370e829c5/model/dictionary.py.
No feature normalization, fitting, label selection, or network downloads occur here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn

STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
ACTIVATION_DIM = 1280
DICTIONARY_SIZE = 40960
MOTIF_MD5 = "cabb07008f26705f07e18344607a2d51"
MOTIF_BYTES = 419601512
ARRAY_KEYS = {"h33", "csr_data", "csr_indices", "csr_indptr", "csr_shape"}
STATE_KEYS = {"bias", "encoder.weight", "encoder.bias", "decoder.weight"}


def file_hash(path: Path | str, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    """Identity hashes the exact sequence, never an accession or annotation."""
    validate_sequence(sequence)
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def validate_sequence(sequence: str) -> None:
    if not isinstance(sequence, str) or not (1 <= len(sequence) <= 1022):
        raise ValueError("A complete sequence of 1..1022 residues is required")
    if not set(sequence) <= STANDARD_AA:
        raise ValueError("Sequence must already contain standard uppercase amino acids")


def json_text(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def write_frozen_json(path: Path, data: dict) -> None:
    encoded = json_text(data).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Refusing to replace frozen receipt: {path}")
        return
    with path.open("xb") as stream:
        stream.write(encoded)


class MotifAutoEncoder(nn.Module):
    """Exact published affine-ReLU encoder and affine decoder, no tied refit."""

    def __init__(self, activation_dim: int, dictionary_size: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dictionary_size
        self.bias = nn.Parameter(torch.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dictionary_size, bias=True)
        self.decoder = nn.Linear(dictionary_size, activation_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x - self.bias))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval()
    model.requires_grad_(False)
    return model


def load_motif_model(path: Path, device: torch.device) -> tuple[MotifAutoEncoder, dict]:
    """Published direct state dict only; never enable arbitrary pickle loading."""
    if path.stat().st_size != MOTIF_BYTES or file_hash(path, "md5") != MOTIF_MD5:
        raise RuntimeError("Published MotifAE checkpoint byte length / MD5 mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    wrapper = "direct_state_dict"
    if isinstance(state, (dict, OrderedDict)) and set(state) == {"state_dict"}:
        state, wrapper = state["state_dict"], "single_state_dict_wrapper"
    if not isinstance(state, (dict, OrderedDict)) or set(state) != STATE_KEYS:
        raise ValueError("Unexpected MotifAE state keys; inspection required before adaptation")
    shapes = {
        "bias": (ACTIVATION_DIM,),
        "encoder.weight": (DICTIONARY_SIZE, ACTIVATION_DIM),
        "encoder.bias": (DICTIONARY_SIZE,),
        "decoder.weight": (ACTIVATION_DIM, DICTIONARY_SIZE),
    }
    for key, shape in shapes.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Unexpected tensor shape: {key}")
        if value.dtype != torch.float32 or not torch.isfinite(value).all().item():
            raise ValueError(f"Checkpoint tensor must be finite FP32: {key}")
    with torch.device("meta"):
        model = MotifAutoEncoder(ACTIVATION_DIM, DICTIONARY_SIZE)
    model.load_state_dict(state, strict=True, assign=True)
    freeze_model(model).to(device)
    return model, {"state_layout": wrapper, "state_keys": sorted(state),
                   "activation_dim": ACTIVATION_DIM, "dictionary_size": DICTIONARY_SIZE,
                   "dtype": "float32", "all_parameters_frozen": True}


def safe_esm_checkpoint(path: Path) -> dict:
    """Official ESM cfg uses only the standard argparse.Namespace allowlist."""
    with torch.serialization.safe_globals([argparse.Namespace]):
        value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not {"cfg", "model"} <= set(value):
        raise ValueError("Unexpected original fair-esm checkpoint structure")
    if not isinstance(value["model"], (dict, OrderedDict)):
        raise ValueError("Missing ESM parameter mapping")
    return value


def load_esm_model(path: Path, device: torch.device):
    import esm

    if getattr(esm, "__version__", None) != "2.0.0":
        raise RuntimeError("This extractor requires the verified fair-esm 2.0.0 runtime")
    regression = path.with_name(path.stem + "-contact-regression.pt")
    if not regression.is_file():
        raise FileNotFoundError(f"Local contact regression checkpoint missing: {regression}")
    model_data = safe_esm_checkpoint(path)
    regression_data = torch.load(regression, map_location="cpu", weights_only=True)
    if not isinstance(regression_data, dict) or "model" not in regression_data:
        raise ValueError("Unexpected contact regression checkpoint structure")
    # This fair-esm function consumes existing objects; unlike its local/hub
    # convenience loaders it does not deserialize files or access the network.
    model, alphabet = esm.pretrained.load_model_and_alphabet_core(
        path.stem, model_data, regression_data)
    del model_data, regression_data
    if model.num_layers != 33 or model.embed_dim != ACTIVATION_DIM:
        raise ValueError("Expected ESM2 t33 650M with 1280-dimensional H33")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("ESM parameters must be FP32")
    freeze_model(model).to(device)
    return model, alphabet


def residue_representation(representations: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """Batch-one BOS + full sequence + EOS, with no truncation or padding."""
    if representations.ndim != 3 or tuple(representations.shape[:2]) != (1, sequence_length + 2):
        raise ValueError("Expected exactly BOS + full sequence + EOS at batch size one")
    residues = representations[0, 1:sequence_length + 1]
    if residues.dtype != torch.float32 or not torch.isfinite(residues).all().item():
        raise ValueError("ESM residue representations must be finite FP32")
    return residues


@torch.inference_mode()
def extract_h33(model, alphabet, sequence: str, device: torch.device) -> torch.Tensor:
    validate_sequence(sequence)
    _, converted, tokens = alphabet.get_batch_converter()([("sequence", sequence)])
    if converted != [sequence] or tuple(tokens.shape) != (1, len(sequence) + 2):
        raise ValueError("ESM tokenizer changed or truncated the sequence")
    output = model(tokens.to(device), repr_layers=[33], return_contacts=False)
    # ESM's layer-33 representation already includes its own final layer norm.
    residues = residue_representation(output["representations"][33], len(sequence))
    if residues.shape[1] != ACTIVATION_DIM:
        raise ValueError("Unexpected ESM hidden width")
    return residues


@torch.inference_mode()
def encode_csr(model: MotifAutoEncoder, h33: torch.Tensor, chunk_size: int = 64) -> tuple[dict, dict]:
    """Encode bounded residue chunks; never retain an L x 40960 GPU matrix."""
    if not 1 <= chunk_size <= 128:
        raise ValueError("Residue chunk size must be within 1..128")
    if h33.ndim != 2 or h33.shape[0] < 1 or h33.shape[1] != model.activation_dim:
        raise ValueError("Unexpected residue hidden representation shape")
    if h33.dtype != torch.float32 or not torch.isfinite(h33).all().item():
        raise ValueError("Hidden representations must be finite FP32")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Feature dictionary must be frozen in evaluation mode")
    all_data, all_indices, all_counts = [], [], []
    squared_error, scalar_count = 0.0, 0
    for start in range(0, len(h33), chunk_size):
        x = h33[start:start + chunk_size]
        z = model.encode(x)
        reconstruction = model.decode(z)
        if not torch.isfinite(z).all().item() or not torch.isfinite(reconstruction).all().item():
            raise ValueError("Nonfinite MotifAE activations / reconstruction")
        active = z > 0
        coordinates = torch.nonzero(active, as_tuple=False)
        all_data.append(z[active].cpu().numpy().astype(np.float32, copy=False))
        all_indices.append(coordinates[:, 1].cpu().numpy().astype(np.int32, copy=False))
        all_counts.append(active.sum(dim=1).cpu().numpy().astype(np.int64, copy=False))
        squared_error += ((reconstruction - x).double().square().sum().item())
        scalar_count += x.numel()
        del z, reconstruction, active, coordinates
    counts = np.concatenate(all_counts)
    indptr64 = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    if indptr64[-1] > np.iinfo(np.int32).max:
        raise ValueError("CSR nonzero count exceeds int32 storage")
    arrays = {
        "h33": h33.cpu().numpy().astype(np.float32, copy=True),
        "csr_data": np.concatenate(all_data),
        "csr_indices": np.concatenate(all_indices),
        "csr_indptr": indptr64.astype(np.int32),
        "csr_shape": np.asarray((len(h33), model.dict_size), dtype=np.int64),
    }
    diagnostics = {
        "residues": len(h33), "features": model.dict_size,
        "reconstruction_mse": squared_error / scalar_count,
        "l0_min": int(counts.min()), "l0_max": int(counts.max()),
        "l0_mean": float(counts.mean()), "nonzero_values": int(indptr64[-1]),
        "zero_feature_rows": int((counts == 0).sum()),
        "h33_all_finite": True, "feature_values_all_finite": True,
        "h33_abs_max": float(np.abs(arrays["h33"]).max()),
        "feature_value_max": float(arrays["csr_data"].max(initial=0)),
    }
    return arrays, diagnostics


def validate_csr(arrays: dict, sequence_length: int, activation_dim=ACTIVATION_DIM,
                 dictionary_size=DICTIONARY_SIZE) -> None:
    if set(arrays) != ARRAY_KEYS:
        raise ValueError("Unexpected private feature cache arrays")
    expected = {"h33": np.dtype("float32"), "csr_data": np.dtype("float32"),
                "csr_indices": np.dtype("int32"), "csr_indptr": np.dtype("int32"),
                "csr_shape": np.dtype("int64")}
    if any(arrays[key].dtype != dtype for key, dtype in expected.items()):
        raise ValueError("Unexpected feature cache dtype")
    if arrays["h33"].shape != (sequence_length, activation_dim):
        raise ValueError("Cached hidden representation shape mismatch")
    if arrays["csr_shape"].shape != (2,) or arrays["csr_shape"].tolist() != [sequence_length, dictionary_size]:
        raise ValueError("Cached dictionary shape mismatch")
    data, indices, indptr = (arrays[key] for key in ("csr_data", "csr_indices", "csr_indptr"))
    if data.ndim != 1 or indices.shape != data.shape or indptr.shape != (sequence_length + 1,):
        raise ValueError("Malformed CSR vectors")
    if indptr[0] != 0 or indptr[-1] != len(data) or np.any(np.diff(indptr) < 0):
        raise ValueError("Malformed CSR row pointers")
    if np.any(indices < 0) or np.any(indices >= dictionary_size) or np.any(data <= 0):
        raise ValueError("Malformed CSR indices / values")
    if not np.isfinite(data).all() or not np.isfinite(arrays["h33"]).all():
        raise ValueError("Nonfinite cached values")
    for row in range(sequence_length):
        if np.any(np.diff(indices[indptr[row]:indptr[row + 1]]) <= 0):
            raise ValueError("CSR columns must be unique and ascending within each row")


def read_panel_sequences(path: Path) -> dict[str, dict[str, str]]:
    """Do not inspect splits, targets, classes, coordinates or motif annotations."""
    sequences = {"full": {}, "short": {}}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for view, field in (("full", "parent_sequence"), ("short", "short_sequence")):
            sequence = row[field]
            digest = sequence_sha256(sequence)
            previous = sequences[view].setdefault(digest, sequence)
            if previous != sequence:
                raise RuntimeError("Sequence digest collision")
    if not sequences["full"] or not sequences["short"]:
        raise ValueError("Empty extraction panel")
    return {view: dict(sorted(items.items())) for view, items in sequences.items()}


def source_pins(scope: Path, esm_checkpoint: Path, motif_checkpoint: Path) -> dict:
    paths = {
        "panel": scope / "data/panel.jsonl", "feature_source": Path(__file__),
        "esm_checkpoint": esm_checkpoint, "motif_checkpoint": motif_checkpoint,
        "esm_contact_regression": esm_checkpoint.with_name(
            esm_checkpoint.stem + "-contact-regression.pt"),
        "reference_embedding": scope / "assets/reference_embedding.npz",
        "reference_sequence": scope / "assets/reference_sequence.txt",
    }
    return {name: {"sha256": file_hash(path), "bytes": path.stat().st_size}
            for name, path in paths.items()}


def file_identity(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def read_cache(path: Path, sequence: str, pins: dict) -> dict:
    receipt_path = path.with_suffix(".json")
    if not receipt_path.is_file():
        raise RuntimeError(f"Cache exists without frozen receipt: {path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["source_pins"] != pins or receipt["sequence_sha256"] != sequence_sha256(sequence):
        raise RuntimeError("Existing cache source / sequence pin mismatch")
    if receipt["npz_sha256"] != file_hash(path):
        raise RuntimeError("Existing feature cache checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_csr(arrays, len(sequence))
    return receipt


def save_cache(path: Path, sequence: str, arrays: dict, diagnostics: dict, pins: dict) -> dict:
    validate_csr(arrays, len(sequence))
    if path.exists() or path.with_suffix(".json").exists():
        raise RuntimeError("Feature outputs are immutable; reuse verified cache or inspect incomplete output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.partial")
    if temporary.exists():
        raise RuntimeError(f"Incomplete prior feature write retained: {temporary}")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    # Scope lock serializes writers; preserve a failed partial instead of overwriting.
    if path.exists():
        raise RuntimeError("Another feature writer created the destination")
    temporary.rename(path)
    receipt = {"schema_version": 1, "private_data": True, "sequence_sha256": sequence_sha256(sequence),
               "source_pins": pins, "npz_sha256": file_hash(path), "npz_bytes": path.stat().st_size,
               "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                          for key, value in arrays.items()}, "diagnostics": diagnostics}
    write_frozen_json(path.with_suffix(".json"), receipt)
    return receipt


@contextmanager
def scope_lock(scope: Path):
    """Node-local advisory lock, automatically released on worker exit."""
    import fcntl

    lock_path = scope / "cache/.feature_extractor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def configure_runtime() -> torch.device:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible.strip() in {"-1", ""}:
        raise RuntimeError("Launcher must assign exactly one CUDA_VISIBLE_DEVICES identity")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one assigned CUDA device must be visible")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("Launcher must set deterministic CUBLAS_WORKSPACE_CONFIG before Python starts")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return torch.device("cuda:0")


def reference_smoke(scope: Path, esm_model, alphabet, motif_model, device, pins: dict) -> dict:
    sequence = (scope / "assets/reference_sequence.txt").read_text(encoding="utf-8").strip()
    validate_sequence(sequence)
    if len(sequence) != 719:
        raise ValueError("Published reference must contain exactly 719 residues")
    with np.load(scope / "assets/reference_embedding.npz", allow_pickle=False) as archive:
        if "repr" not in archive.files:
            raise ValueError("Published reference is missing repr")
        reference = archive["repr"]
    if reference.shape != (719, ACTIVATION_DIM) or reference.dtype != np.float32:
        raise ValueError("Published reference repr must be 719x1280 float32")
    if not np.isfinite(reference).all():
        raise ValueError("Nonfinite published reference")
    actual = extract_h33(esm_model, alphabet, sequence, device)
    actual_np = actual.cpu().numpy()
    difference = actual_np.astype(np.float64) - reference.astype(np.float64)
    absolute_max = float(np.abs(difference).max())
    passed = bool(np.allclose(actual_np, reference, atol=1e-3, rtol=0))
    dot = float(np.sum(actual_np.astype(np.float64) * reference))
    denominator = float(np.linalg.norm(actual_np.astype(np.float64)) * np.linalg.norm(reference.astype(np.float64)))
    arrays, diagnostics = encode_csr(motif_model, actual)
    validate_csr(arrays, len(sequence))
    receipt = {"schema_version": 1, "stage": "official_reference_smoke", "passed": passed,
               "source_pins": pins, "sequence_sha256": sequence_sha256(sequence),
               "reference_shape": [719, ACTIVATION_DIM], "absolute_tolerance": 1e-3,
               "relative_tolerance": 0, "max_absolute_difference": absolute_max,
               "rmse": float(np.sqrt(np.mean(difference ** 2))),
               "cosine_similarity": dot / denominator if denominator else None,
               "motif_diagnostics": diagnostics, "extra_l2_normalization": False,
               "esm_layer": 33, "esm_final_layer_norm": "original fair-esm model output",
               "trained_parameters": 0, "evaluation_labels_consumed": False}
    write_frozen_json(scope / "cache/reference_smoke.json", receipt)
    if not passed:
        raise RuntimeError("ESM output differs from the official reference beyond frozen 1e-3 absolute tolerance")
    return receipt


def run(args) -> dict:
    scope = args.scope.resolve()
    device = configure_runtime()
    with scope_lock(scope):
        if not 1 <= args.chunk_size <= 128:
            raise ValueError("Residue chunk size must be within 1..128")
        large_inputs = (args.esm_checkpoint, args.motif_checkpoint,
                        args.esm_checkpoint.with_name(args.esm_checkpoint.stem + "-contact-regression.pt"))
        original_identities = {path: file_identity(path) for path in large_inputs}
        pins = source_pins(scope, args.esm_checkpoint, args.motif_checkpoint)
        pins["encoding_protocol"] = {"residue_chunk_size": args.chunk_size,
                                     "dtype": "float32", "extra_normalization": False}
        prior_path = scope / "cache/extraction_source_pins.json"
        write_frozen_json(prior_path, pins)
        motif_model, model_receipt = load_motif_model(args.motif_checkpoint, device)
        esm_model, alphabet = load_esm_model(args.esm_checkpoint, device)
        write_frozen_json(scope / "cache/model_receipt.json", model_receipt)
        smoke_path = scope / "cache/reference_smoke.json"
        if smoke_path.exists():
            smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
            if smoke.get("source_pins") != pins or smoke.get("passed") is not True:
                raise RuntimeError("Existing smoke receipt is failed or belongs to different frozen sources")
        else:
            smoke = reference_smoke(scope, esm_model, alphabet, motif_model, device, pins)
        if args.stage == "smoke":
            return smoke
        sequences = read_panel_sequences(scope / "data/panel.jsonl")
        outputs = {}
        for view in ("full", "short"):
            for digest, sequence in sequences[view].items():
                path = scope / "cache" / view / (digest + ".npz")
                if path.exists():
                    receipt = read_cache(path, sequence, pins)
                else:
                    h33 = extract_h33(esm_model, alphabet, sequence, device)
                    arrays, diagnostics = encode_csr(motif_model, h33, args.chunk_size)
                    receipt = save_cache(path, sequence, arrays, diagnostics, pins)
                    del h33, arrays
                relative = path.relative_to(scope).as_posix()
                outputs[relative] = {"sha256": receipt["npz_sha256"], "bytes": receipt["npz_bytes"],
                                     "receipt_sha256": file_hash(path.with_suffix(".json")),
                                     "sequence_sha256": digest, "residues": len(sequence)}
                print(json.dumps({"event": "sequence_cache_verified", "view": view,
                                  "done": len(outputs), "expected": sum(map(len, sequences.values()))}), flush=True)
        # Detect source changes before claiming a complete extraction. The large
        # model files were hashed once above, then checked for file replacement.
        if file_hash(scope / "data/panel.jsonl") != pins["panel"]["sha256"]:
            raise RuntimeError("Panel changed during extraction")
        if file_hash(Path(__file__)) != pins["feature_source"]["sha256"]:
            raise RuntimeError("Feature source changed during extraction")
        if any(file_identity(path) != identity for path, identity in original_identities.items()):
            raise RuntimeError("Model asset identity / size / modification time changed during extraction")
        manifest = {"schema_version": 1, "status": "complete", "private_data": True,
                    "source_pins": pins, "model": model_receipt,
                    "coverage": {view: {"unique_sequences": len(items),
                                        "residues": sum(map(len, items.values())),
                                        "every_residue_encoded": True, "truncated_sequences": 0}
                                 for view, items in sequences.items()},
                    "output_count": len(outputs), "outputs": outputs,
                    "feature_input_fields": ["parent_sequence", "short_sequence"],
                    "evaluation_labels_consumed": False, "normalization": "none_added",
                    "all_parameters_frozen": True, "training_updates": 0}
        write_frozen_json(scope / "cache/features_manifest.json", manifest)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--esm-checkpoint", type=Path, required=True)
    parser.add_argument("--motif-checkpoint", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "extract"), required=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()
    result = run(args)
    print(json_text({"stage": args.stage, "status": result.get("status", "passed")}))


if __name__ == "__main__":
    main()

# Tensor checkpoints

38/38 safetensors files verified; 69,046,380 file bytes and 68,999,828 raw tensor bytes.
This project exports31 selected reader states,6 untrained paired initializations, and one fixed projection.

| Directory/disposition | Count | Status |
|---|---:|---|
| selected/accepted_formal |30| Formally accepted originalV1 states |
| selected/rejected_not_formal |1| Complete attempt rejected by original validator; quarantined |
| initial |6| Seeds17/42/2026 with alpha0 or3; untrained, no labels |
| fixed |1| Seed314159 Gaussian640x128 projection |

Five unstarted plan positions have no manufactured checkpoint.
The rejected state's artifact.complete and self-review.accepted labels do not override standard_valid_for_acceptance=false; its metrics do not enter formal pool results.

[Complete hashes, tensor shapes/dtypes/byte checksums, selected epochs and original scalar metrics](../checkpoints/manifest.json).
Source locations are relative provenance locators in the original experiment workspace, not personal absolute paths or download URLs.
Own-run scalar metrics and exact acceptance dispositions are retained; group-level source rows and runtime metadata are omitted.

Each reader has11 tensors/464001 float32 elements. Every exported tensor was reloaded on CPU and checked for exact tensor and raw-byte equality; no cast or quantization.
Headers contain only artifact identity/disposition, source SHA and the tensor-only marker.
Original pickled registration, hostname, process IDs, absolute paths and runtime source metadata are excluded.
Original.pt files remain unchanged. CPU publication serialization does not restart the scientific time window.

Load with safetensors.torch.load_file(path, device='cpu') using the matching original architecture.
Pretrained Qwen/ESM or external author weights are outside this tensor archive.

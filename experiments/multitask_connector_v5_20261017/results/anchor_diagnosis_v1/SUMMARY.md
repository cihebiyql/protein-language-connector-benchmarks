# TRAIN-only anchor diagnosis

Two existing failed Qwen2.5-1.5B M3 qualification checkpoints were loaded without training. Each task used its first TRAIN row. Neither validation nor test data was opened. Physical GPU 3 was used under an 8 GiB allocator ceiling, 64 GiB host RSS ceiling, and 600-second independent process-group guard.

| Task | Comparison | Connector on head max difference | Connector off head max difference |
|---|---|---:|---:|
| Meltome | Original length T versus appended T+2 | 0.05488532781600952 | 0.01938539743423462 |
| Meltome | Same T+2 shape, future suffix `[1,1]` versus `[2,2]` | 0 | 0 |
| Meltome | Same shape, pre-answer prefix positive control | 0.0742616057395935 | 0.05989265441894531 |
| ELM | Original length T versus appended T+2 | 0.04260987043380737 | 0.02776026725769043 |
| ELM | Same T+2 shape, future suffix `[1,1]` versus `[2,2]` | 0 | 0 |
| ELM | Same shape, pre-answer prefix positive control | 0.04093611240386963 | 0.04766610264778137 |

For the same-shape future replacement, every prompt-hidden value and both view/bank route tensors were also bitwise equal for both tasks. The positive control changed all of these outputs and was rejected at the unchanged 1e-5 comparison tolerance.

The original shape-changing check therefore confounded future-token causality with shape-dependent BF16 arithmetic: the frozen backbone with the connector disabled also changed across T versus T+2. This evidence supports comparing different synthetic future contents at an identical shape, while retaining the original shape-changing values as diagnostics. It does not support loosening the tolerance or changing training/evaluation targets.

Both original checkpoint file hashes and all loaded trainable tensors were unchanged. Source-file hashes were identical at the beginning and end. Exact vectors, routes, dtype, token counts, original failure records, comparisons, and software versions are retained in `diagnosis.json`.

Runtime: PyTorch 2.6.0+cu124, Transformers 4.51.3, CUDA 12.4; peak allocated GPU memory 3,157,221,888 bytes, reserved 3,355,443,200 bytes. The successful supervisor reported exit 0, no remaining group members, and peak group RSS 5,979,226,112 bytes.

The first launch had a command assembly error before importing the model; its separate failed-launch receipt is preserved. The corrected second launch is the source of all diagnostic results above.

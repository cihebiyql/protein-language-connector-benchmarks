"""Apply the contracted PyTorch allocator cap before importing a GPU worker."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
from datetime import datetime, timezone


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-memory-budget-bytes', required=True, type=int)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--module', choices=('src.mm_extract', 'src.mm_train'), required=True)
    args, rest = parser.parse_known_args(argv)
    if rest[:1] == ['--']:
        rest = rest[1:]
    if args.gpu_memory_budget_bytes <= 0 or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Explicit GPU assignment and positive allocator budget required')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    total = torch.cuda.get_device_properties(0).total_memory
    if args.gpu_memory_budget_bytes > total:
        raise ValueError('Requested GPU allocator budget exceeds physical GPU memory')
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_budget_bytes / total, device=0)
    receipt_path = Path(args.receipt)
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    receipt = dict(started_at=datetime.now(timezone.utc).isoformat(),
                   visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                   allocator_budget_bytes=args.gpu_memory_budget_bytes,
                   physical_total_bytes=total, module=args.module,
                   determinism=True, tf32=False, cpu_threads=2,
                   scope='PyTorch allocator; CUDA context/library overhead reported separately by supervisor resource probes')
    old_argv = sys.argv
    try:
        sys.argv = [args.module] + rest
        runpy.run_module(args.module, run_name='__main__', alter_sys=True)
    finally:
        sys.argv = old_argv
        receipt.update(ended_at=datetime.now(timezone.utc).isoformat(),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        with receipt_path.open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Profile double buffer implementation."""

import torch
import torch.distributed as dist
import os
import argparse
from tp_overlap_double_buffer import (
    DoubleBufferOverlapRowParallelLinear,
    setup_distributed,
    cleanup_distributed,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--in_features", type=int, default=4096)
    parser.add_argument("--out_features", type=int, default=12288)
    parser.add_argument("--num_chunks", type=int, default=4)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    setup_distributed(rank, world_size)
    device = f"cuda:{rank}"

    if rank == 0:
        print(f"Profiling double buffer overlap...")

    # Create input
    torch.manual_seed(42 + rank)
    input_tensor = torch.randn(
        args.batch_size, args.seq_len, args.in_features,
        device=device, dtype=torch.float32
    )

    # Create layer
    layer = DoubleBufferOverlapRowParallelLinear(
        in_features=args.in_features,
        out_features=args.out_features,
        num_chunks=args.num_chunks,
        process_group=None,
        device=device,
    )

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = layer(input_tensor)
        torch.cuda.synchronize()

    # Profiled iterations
    with torch.no_grad():
        for i in range(10):
            _ = layer(input_tensor)
        torch.cuda.synchronize()

    if rank == 0:
        print(f"Profiling complete")

    cleanup_distributed()


if __name__ == "__main__":
    main()

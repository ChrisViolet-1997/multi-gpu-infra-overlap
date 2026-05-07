#!/usr/bin/env python3
"""
Profile script for analyzing overlap performance with nsys.

Usage:
    # Profile baseline
    nsys profile -o baseline_profile --trace=cuda,nvtx,osrt \
        torchrun --nproc_per_node=2 profile_overlap.py --mode baseline

    # Profile overlap
    nsys profile -o overlap_profile --trace=cuda,nvtx,osrt \
        torchrun --nproc_per_node=2 profile_overlap.py --mode overlap
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Optional
import os
import argparse
from tp_overlap_poc import (
    BaselineRowParallelLinear,
    OverlapRowParallelLinear,
    setup_distributed,
    cleanup_distributed,
)


def profile_layer(
    layer: nn.Module,
    input_tensor: torch.Tensor,
    mode: str,
    num_warmup: int = 5,
    num_iterations: int = 10,
):
    """Profile a layer with NVTX markers for nsys visualization."""

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = layer(input_tensor)
        torch.cuda.synchronize()

    # Profiled iterations with NVTX markers
    with torch.no_grad():
        for i in range(num_iterations):
            torch.cuda.nvtx.range_push(f"{mode}_iteration_{i}")
            output = layer(input_tensor)
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description="Profile overlap implementation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "overlap"],
        required=True,
        help="Which implementation to profile",
    )
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
        print(f"Profiling {args.mode} mode...")
        print(f"Config: BS={args.batch_size}, SeqLen={args.seq_len}, "
              f"In={args.in_features}, Out={args.out_features}, Chunks={args.num_chunks}")

    # Create input
    torch.manual_seed(42 + rank)
    input_tensor = torch.randn(
        args.batch_size, args.seq_len, args.in_features,
        device=device, dtype=torch.float32
    )

    # Create layer based on mode
    if args.mode == "baseline":
        layer = BaselineRowParallelLinear(
            in_features=args.in_features,
            out_features=args.out_features,
            process_group=None,
            device=device,
        )
    else:
        layer = OverlapRowParallelLinear(
            in_features=args.in_features,
            out_features=args.out_features,
            num_chunks=args.num_chunks,
            process_group=None,
            device=device,
        )

    # Profile
    profile_layer(layer, input_tensor, args.mode)

    if rank == 0:
        print(f"Profiling complete for {args.mode} mode")

    cleanup_distributed()


if __name__ == "__main__":
    main()

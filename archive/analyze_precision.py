#!/usr/bin/env python3
"""Analyze precision differences between baseline and overlap implementations."""

import torch
import torch.distributed as dist
import os
import sys


def analyze_precision():
    """Detailed precision analysis."""
    from tp_overlap_poc import setup_distributed, cleanup_distributed

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    setup_distributed(rank, world_size)

    batch_size = 4
    seq_len = 128
    in_features = 512
    out_features = 512
    device = f"cuda:{rank}"

    torch.manual_seed(42)
    weight = torch.randn(out_features, in_features, device=device, dtype=torch.float32)

    torch.manual_seed(42 + rank)
    input_tensor = torch.randn(
        batch_size, seq_len, in_features, device=device, dtype=torch.float32
    )

    if rank == 0:
        print("=" * 80)
        print("PRECISION ANALYSIS")
        print("=" * 80)

    # Test 1: Baseline (full GEMM + all_reduce)
    with torch.no_grad():
        output_baseline = torch.matmul(input_tensor.view(-1, in_features), weight.t())
        dist.all_reduce(output_baseline, op=dist.ReduceOp.SUM)

    # Test 2: Chunked GEMM without streams (to isolate chunking effect)
    total_tokens = batch_size * seq_len
    x_flat = input_tensor.view(-1, in_features)

    for num_chunks in [2, 4, 8]:
        chunk_size = (total_tokens + num_chunks - 1) // num_chunks
        output_chunked = torch.empty(total_tokens, out_features, device=device, dtype=torch.float32)

        # Compute chunks sequentially without streams
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_tokens)

            if start_idx >= total_tokens:
                break

            x_chunk = x_flat[start_idx:end_idx]
            output_chunk = output_chunked[start_idx:end_idx]

            # Compute and immediately reduce
            torch.matmul(x_chunk, weight.t(), out=output_chunk)
            dist.all_reduce(output_chunk, op=dist.ReduceOp.SUM)

        max_diff = torch.max(torch.abs(output_baseline - output_chunked)).item()
        mean_diff = torch.mean(torch.abs(output_baseline - output_chunked)).item()

        if rank == 0:
            print(f"\nChunks: {num_chunks}")
            print(f"  Max diff:  {max_diff:.2e}")
            print(f"  Mean diff: {mean_diff:.2e}")

    # Test 3: Check if it's a stream synchronization issue
    if rank == 0:
        print("\n" + "=" * 80)
        print("Testing with explicit synchronization...")
        print("=" * 80)

    from tp_overlap_poc import OverlapRowParallelLinear

    for num_chunks in [2, 4, 8]:
        overlap_layer = OverlapRowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            num_chunks=num_chunks,
            process_group=None,
            device=device,
        )
        overlap_layer.weight.data.copy_(weight)

        with torch.no_grad():
            output_overlap = overlap_layer(input_tensor)

        output_overlap_flat = output_overlap.view(-1, out_features)
        max_diff = torch.max(torch.abs(output_baseline - output_overlap_flat)).item()
        mean_diff = torch.mean(torch.abs(output_baseline - output_overlap_flat)).item()

        if rank == 0:
            print(f"\nOverlap with {num_chunks} chunks:")
            print(f"  Max diff:  {max_diff:.2e}")
            print(f"  Mean diff: {mean_diff:.2e}")

    if rank == 0:
        print("\n" + "=" * 80)

    cleanup_distributed()


if __name__ == "__main__":
    if not dist.is_available() or not torch.cuda.is_available():
        print("Error: Requires PyTorch with CUDA and distributed support")
        sys.exit(1)

    analyze_precision()

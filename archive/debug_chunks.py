#!/usr/bin/env python3
"""Debug script to understand the 2-chunk failure."""

import torch
import torch.distributed as dist
import os
import sys


def debug_two_chunks():
    """Test 2-chunk case in detail."""
    from tp_overlap_poc import (
        BaselineRowParallelLinear,
        OverlapRowParallelLinear,
        setup_distributed,
        cleanup_distributed,
    )

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    setup_distributed(rank, world_size)

    batch_size = 4
    seq_len = 256
    in_features = 512
    out_features = 512
    device = f"cuda:{rank}"

    torch.manual_seed(42)
    weight = torch.randn(out_features, in_features, device=device, dtype=torch.float32)

    baseline_layer = BaselineRowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        process_group=None,
        device=device,
    )
    baseline_layer.weight.data.copy_(weight)

    torch.manual_seed(42 + rank)
    input_tensor = torch.randn(
        batch_size, seq_len, in_features, device=device, dtype=torch.float32
    )

    with torch.no_grad():
        baseline_output = baseline_layer(input_tensor)

    # Test with 2 chunks
    overlap_layer = OverlapRowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        num_chunks=2,
        process_group=None,
        device=device,
    )
    overlap_layer.weight.data.copy_(weight)

    with torch.no_grad():
        overlap_output = overlap_layer(input_tensor)

    if rank == 0:
        print("=" * 80)
        print("DEBUG: 2-CHUNK CASE")
        print("=" * 80)
        print(f"Input shape: {input_tensor.shape}")
        print(f"Baseline output shape: {baseline_output.shape}")
        print(f"Overlap output shape: {overlap_output.shape}")
        print()

        diff = baseline_output - overlap_output
        max_diff = torch.max(torch.abs(diff)).item()
        mean_diff = torch.mean(torch.abs(diff)).item()

        print(f"Max absolute difference:  {max_diff:.2e}")
        print(f"Mean absolute difference: {mean_diff:.2e}")
        print()

        # Check if specific regions have issues
        total_tokens = batch_size * seq_len
        chunk_size = (total_tokens + 2 - 1) // 2

        print(f"Total tokens: {total_tokens}")
        print(f"Chunk size: {chunk_size}")
        print()

        # Flatten and check chunks
        baseline_flat = baseline_output.view(-1, out_features)
        overlap_flat = overlap_output.view(-1, out_features)

        chunk0_diff = torch.max(torch.abs(baseline_flat[:chunk_size] - overlap_flat[:chunk_size])).item()
        chunk1_diff = torch.max(torch.abs(baseline_flat[chunk_size:] - overlap_flat[chunk_size:])).item()

        print(f"Chunk 0 max diff: {chunk0_diff:.2e}")
        print(f"Chunk 1 max diff: {chunk1_diff:.2e}")
        print()

        # Sample values
        print("Sample baseline values (first 5 of chunk 0):")
        print(baseline_flat[0, :5])
        print("Sample overlap values (first 5 of chunk 0):")
        print(overlap_flat[0, :5])
        print()

        print("Sample baseline values (first 5 of chunk 1):")
        print(baseline_flat[chunk_size, :5])
        print("Sample overlap values (first 5 of chunk 1):")
        print(overlap_flat[chunk_size, :5])
        print("=" * 80)

    cleanup_distributed()


if __name__ == "__main__":
    if not dist.is_available() or not torch.cuda.is_available():
        print("Error: Requires PyTorch with CUDA and distributed support")
        sys.exit(1)

    debug_two_chunks()

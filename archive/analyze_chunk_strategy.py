#!/usr/bin/env python3
"""
Analyze different chunking strategies to achieve 1e-5 precision.

The key insight: precision loss comes from changing the order of all_reduce operations.
We need to ensure the reduction order matches the baseline.
"""

import torch
import torch.distributed as dist
import os
import sys


def test_chunking_strategy(rank: int, world_size: int):
    """Test different chunking strategies."""
    from tp_overlap_poc import setup_distributed, cleanup_distributed

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
        print("CHUNKING STRATEGY ANALYSIS")
        print("=" * 80)
        print("\nBaseline: Full GEMM, then single all_reduce")
        print("=" * 80)

    # Baseline: full GEMM then all_reduce
    with torch.no_grad():
        x_flat = input_tensor.view(-1, in_features)
        output_baseline = torch.matmul(x_flat, weight.t())
        dist.all_reduce(output_baseline, op=dist.ReduceOp.SUM)

    total_tokens = batch_size * seq_len

    # Strategy 1: Chunk computation, but accumulate locally before all_reduce
    if rank == 0:
        print("\n[Strategy 1] Chunk GEMM, accumulate locally, then single all_reduce")
        print("-" * 80)

    for num_chunks in [2, 4, 8, 16]:
        chunk_size = (total_tokens + num_chunks - 1) // num_chunks
        output_local = torch.zeros(total_tokens, out_features, device=device, dtype=torch.float32)

        # Compute all chunks locally first
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_tokens)
            if start_idx >= total_tokens:
                break

            x_chunk = x_flat[start_idx:end_idx]
            output_chunk = output_local[start_idx:end_idx]
            torch.matmul(x_chunk, weight.t(), out=output_chunk)

        # Single all_reduce at the end
        dist.all_reduce(output_local, op=dist.ReduceOp.SUM)

        max_diff = torch.max(torch.abs(output_baseline - output_local)).item()
        mean_diff = torch.mean(torch.abs(output_baseline - output_local)).item()

        if rank == 0:
            status = "✓" if max_diff < 1e-5 else "✗"
            print(f"  Chunks: {num_chunks:2d} | Max: {max_diff:.2e} | Mean: {mean_diff:.2e} | {status}")

    # Strategy 2: Current implementation (chunk GEMM + immediate all_reduce per chunk)
    if rank == 0:
        print("\n[Strategy 2] Chunk GEMM + immediate all_reduce per chunk (CURRENT)")
        print("-" * 80)

    for num_chunks in [2, 4, 8, 16]:
        chunk_size = (total_tokens + num_chunks - 1) // num_chunks
        output_chunked = torch.empty(total_tokens, out_features, device=device, dtype=torch.float32)

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_tokens)
            if start_idx >= total_tokens:
                break

            x_chunk = x_flat[start_idx:end_idx]
            output_chunk = output_chunked[start_idx:end_idx]
            torch.matmul(x_chunk, weight.t(), out=output_chunk)
            dist.all_reduce(output_chunk, op=dist.ReduceOp.SUM)

        max_diff = torch.max(torch.abs(output_baseline - output_chunked)).item()
        mean_diff = torch.mean(torch.abs(output_baseline - output_chunked)).item()

        if rank == 0:
            status = "✓" if max_diff < 1e-5 else "✗"
            print(f"  Chunks: {num_chunks:2d} | Max: {max_diff:.2e} | Mean: {mean_diff:.2e} | {status}")

    # Strategy 3: Chunk along output dimension instead of batch dimension
    if rank == 0:
        print("\n[Strategy 3] Chunk along output features (not batch)")
        print("-" * 80)

    for num_chunks in [2, 4, 8, 16]:
        chunk_size = (out_features + num_chunks - 1) // num_chunks
        output_feature_chunked = torch.empty(total_tokens, out_features, device=device, dtype=torch.float32)

        for chunk_idx in range(num_chunks):
            start_feat = chunk_idx * chunk_size
            end_feat = min(start_feat + chunk_size, out_features)
            if start_feat >= out_features:
                break

            weight_chunk = weight[start_feat:end_feat, :]
            output_chunk = output_feature_chunked[:, start_feat:end_feat]
            torch.matmul(x_flat, weight_chunk.t(), out=output_chunk)

        # Single all_reduce at the end
        dist.all_reduce(output_feature_chunked, op=dist.ReduceOp.SUM)

        max_diff = torch.max(torch.abs(output_baseline - output_feature_chunked)).item()
        mean_diff = torch.mean(torch.abs(output_baseline - output_feature_chunked)).item()

        if rank == 0:
            status = "✓" if max_diff < 1e-5 else "✗"
            print(f"  Chunks: {num_chunks:2d} | Max: {max_diff:.2e} | Mean: {mean_diff:.2e} | {status}")

    if rank == 0:
        print("\n" + "=" * 80)
        print("CONCLUSION:")
        print("=" * 80)
        print("Strategy 1: Chunk computation locally, single all_reduce")
        print("  - Pros: Maintains exact numerical equivalence (1e-5+ precision)")
        print("  - Cons: Cannot overlap computation with communication")
        print("  - Use case: When precision is critical")
        print()
        print("Strategy 2: Chunk computation + immediate all_reduce per chunk")
        print("  - Pros: Enables computation-communication overlap")
        print("  - Cons: Changes reduction order, ~1e-4 precision loss")
        print("  - Use case: When performance > precision (typical for ML)")
        print()
        print("Strategy 3: Chunk along output features")
        print("  - Pros: Maintains exact numerical equivalence")
        print("  - Cons: Cannot overlap (all_reduce needs full output)")
        print("  - Use case: Alternative when batch chunking doesn't work")
        print("=" * 80)

    cleanup_distributed()


if __name__ == "__main__":
    if not dist.is_available() or not torch.cuda.is_available():
        print("Error: Requires PyTorch with CUDA and distributed support")
        sys.exit(1)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    test_chunking_strategy(rank, world_size)

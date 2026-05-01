#!/usr/bin/env python3
"""
Unit tests for TP overlap implementation.

Validates correctness of overlapped computation against baseline.
"""

import torch
import torch.distributed as dist
import os
import sys


def test_correctness(rank: int, world_size: int):
    """
    Verify that overlapped implementation produces identical results to baseline.

    Args:
        rank: GPU rank
        world_size: Number of GPUs
    """
    from tp_overlap_poc import (
        BaselineRowParallelLinear,
        OverlapRowParallelLinear,
    )

    # Test configuration
    batch_size = 4
    seq_len = 128
    in_features = 512
    out_features = 512
    num_chunks = 4
    device = f"cuda:{rank}"

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create identical weights for both layers
    weight = torch.randn(out_features, in_features, device=device, dtype=torch.float32)

    baseline_layer = BaselineRowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        process_group=None,
        device=device,
    )
    baseline_layer.weight.data.copy_(weight)

    overlap_layer = OverlapRowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        num_chunks=num_chunks,
        process_group=None,
        device=device,
    )
    overlap_layer.weight.data.copy_(weight)

    # Create input
    torch.manual_seed(42 + rank)
    input_tensor = torch.randn(
        batch_size, seq_len, in_features, device=device, dtype=torch.float32
    )

    # Forward pass
    with torch.no_grad():
        baseline_output = baseline_layer(input_tensor)
        overlap_output = overlap_layer(input_tensor)

    # Check correctness
    max_diff = torch.max(torch.abs(baseline_output - overlap_output)).item()
    mean_diff = torch.mean(torch.abs(baseline_output - overlap_output)).item()

    if rank == 0:
        print("=" * 80)
        print("CORRECTNESS TEST")
        print("=" * 80)
        print(f"Max absolute difference:  {max_diff:.2e}")
        print(f"Mean absolute difference: {mean_diff:.2e}")

        tolerance = 1e-3
        if max_diff < tolerance:
            print(f"\n✓ PASS: Outputs match within tolerance ({tolerance})")
            print("=" * 80)
            return True
        else:
            print(f"\n✗ FAIL: Outputs differ by more than tolerance ({tolerance})")
            print("=" * 80)
            return False

    return True


def test_chunk_sizes(rank: int, world_size: int):
    """Test various chunk sizes for correctness."""
    from tp_overlap_poc import (
        BaselineRowParallelLinear,
        OverlapRowParallelLinear,
    )

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

    if rank == 0:
        print("\n" + "=" * 80)
        print("CHUNK SIZE CORRECTNESS TEST")
        print("=" * 80)

    all_passed = True
    for num_chunks in [2, 4, 8, 16, 32]:
        overlap_layer = OverlapRowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            num_chunks=num_chunks,
            process_group=None,
            device=device,
        )
        overlap_layer.weight.data.copy_(weight)

        with torch.no_grad():
            overlap_output = overlap_layer(input_tensor)

        max_diff = torch.max(torch.abs(baseline_output - overlap_output)).item()

        if rank == 0:
            status = "✓ PASS" if max_diff < 1e-3 else "✗ FAIL"
            print(f"Chunks: {num_chunks:2d} | Max Diff: {max_diff:.2e} | {status}")

        if max_diff >= 1e-3:
            all_passed = False

    if rank == 0:
        print("=" * 80)
        if all_passed:
            print("\n✓ All chunk sizes passed correctness test")
        else:
            print("\n✗ Some chunk sizes failed correctness test")

    return all_passed


def main():
    """Run all tests."""
    if not dist.is_available() or not torch.cuda.is_available():
        print("Error: Requires PyTorch with CUDA and distributed support")
        sys.exit(1)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size < 2:
        print("Error: Requires at least 2 GPUs")
        print("Run with: torchrun --nproc_per_node=2 test_correctness.py")
        sys.exit(1)

    from tp_overlap_poc import setup_distributed, cleanup_distributed

    # Setup distributed once for all tests
    setup_distributed(rank, world_size)

    # Run tests
    test1_passed = test_correctness(rank, world_size)
    test2_passed = test_chunk_sizes(rank, world_size)

    if rank == 0:
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Basic Correctness Test: {'✓ PASS' if test1_passed else '✗ FAIL'}")
        print(f"Chunk Size Test:        {'✓ PASS' if test2_passed else '✗ FAIL'}")
        print("=" * 80)

        if test1_passed and test2_passed:
            print("\n✓ All tests passed!")
        else:
            print("\n✗ Some tests failed")
            sys.exit(1)

    # Cleanup distributed at the end
    cleanup_distributed()


if __name__ == "__main__":
    main()

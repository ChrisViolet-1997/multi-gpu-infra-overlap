#!/usr/bin/env python3
"""
Correctness Verification for Chunked Overlap Computation.

For each operator, compare output of chunk=N (overlap) against chunk=1
(no overlap, sequential baseline). Both use the same weights and input.

Usage:
    torchrun --nproc_per_node=2 verify_correctness.py
"""

import torch
import torch.distributed as dist
import os
import sys

sys.path.insert(0, '/root/autodl-tmp/multi-gpu-infra-overlap/graph_implementation/core')
sys.path.insert(0, '/root/autodl-tmp/multi-gpu-infra-overlap/qwen3_integration')

from tp_overlap_cudagraph import CUDAGraphDoubleBufferOverlapRowParallelLinear


def setup_distributed():
    """Initialize distributed process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def verify_operator(
    name: str,
    in_features: int,
    out_features: int,
    num_chunks: int,
    batch_size: int = 16,
    seq_len: int = 1024,
    rank: int = 0,
):
    """
    Verify correctness of chunked overlap vs sequential baseline.

    Returns (passed, max_diff).
    """
    device = "cuda"
    input_shape = (batch_size, seq_len, in_features)

    # Reference: chunk=1 (no overlap possible, purely sequential)
    ref_layer = CUDAGraphDoubleBufferOverlapRowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        num_chunks=1,
        static_input_shape=input_shape,
        enable_graph=True,
        device=device,
    )

    # Test: chunk=N (overlap enabled)
    test_layer = CUDAGraphDoubleBufferOverlapRowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        num_chunks=num_chunks,
        static_input_shape=input_shape,
        enable_graph=True,
        device=device,
    )

    # Synchronize weights: copy ref weights to test, then broadcast both from rank 0
    with torch.no_grad():
        # First broadcast ref weights so all ranks have the same
        dist.broadcast(ref_layer.weight.data, src=0)
        # Copy to test layer
        test_layer.weight.copy_(ref_layer.weight)

    # Create identical input on all ranks (broadcast from rank 0)
    x = torch.randn(input_shape, device=device)
    dist.broadcast(x, src=0)

    # Run reference (chunk=1)
    with torch.no_grad():
        ref_output = ref_layer(x).clone()

    torch.cuda.synchronize()
    dist.barrier()

    # Run test (chunk=N)
    with torch.no_grad():
        test_output = test_layer(x).clone()

    torch.cuda.synchronize()
    dist.barrier()

    # Compare
    max_diff = (ref_output - test_output).abs().max().item()
    passed = max_diff < 1e-4

    return passed, max_diff


def main():
    rank = setup_distributed()
    world_size = dist.get_world_size()

    if rank == 0:
        print("=" * 80)
        print("CORRECTNESS VERIFICATION: Chunked Overlap vs Sequential Baseline")
        print(f"World size: {world_size}")
        print("=" * 80)

    # Qwen3-8B operator configurations
    operators = [
        ("q_proj",    4096, 4096),
        ("k_proj",    4096, 1024),
        ("v_proj",    4096, 1024),
        ("o_proj",    4096, 4096),
        ("gate_proj", 4096, 12288),
        ("up_proj",   4096, 12288),
        ("down_proj", 12288, 4096),
    ]

    # Chunk values to test per operator
    chunk_tests = [2, 4, 8]

    all_passed = True
    results = []

    for op_name, in_feat, out_feat in operators:
        for num_chunks in chunk_tests:
            # Skip if out_features not divisible by num_chunks
            if out_feat % num_chunks != 0:
                continue

            dist.barrier()
            passed, max_diff = verify_operator(
                name=op_name,
                in_features=in_feat,
                out_features=out_feat,
                num_chunks=num_chunks,
                rank=rank,
            )

            status = "PASS" if passed else "FAIL"
            results.append((op_name, num_chunks, passed, max_diff))

            if not passed:
                all_passed = False

            if rank == 0:
                print(f"  [{status}] {op_name:10s} chunks={num_chunks:2d}  "
                      f"max_diff={max_diff:.2e}")

    if rank == 0:
        print("\n" + "=" * 80)
        if all_passed:
            print("ALL TESTS PASSED")
        else:
            print("SOME TESTS FAILED")
            for op_name, chunks, passed, diff in results:
                if not passed:
                    print(f"  FAILED: {op_name} chunks={chunks} diff={diff:.2e}")
        print("=" * 80)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

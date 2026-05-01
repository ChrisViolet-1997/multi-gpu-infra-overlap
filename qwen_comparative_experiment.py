#!/usr/bin/env python3
"""
Qwen-32B Tensor Parallel Row-Parallel Comparative Experiment

This script provides a rigorous comparison between baseline and overlap implementations
using exact Qwen-32B model dimensions for Row-Parallel layers.

Qwen-32B Architecture:
    - hidden_size = 5120
    - intermediate_size = 13824 (MLP Down-projection)
    - num_attention_heads = 40
    - head_dim = 128
    - Attention Out-projection: [batch, seq_len, 5120] @ [5120, 5120]

Experiment Design:
    1. Baseline: Full GEMM → Blocking All-Reduce
    2. Overlap: Chunked GEMM with pipelined async All-Reduce
    3. Validation: torch.allclose verification
    4. Scaling: Vary sequence length from 512 to 4096
"""

import torch
import torch.distributed as dist
import os
from typing import Tuple, Optional
import time


# ============================================================================
# QWEN-32B MODEL CONFIGURATION
# ============================================================================

class Qwen32BConfig:
    """Qwen-32B model dimensions for Row-Parallel layers."""
    hidden_size = 5120
    intermediate_size = 13824  # MLP Down-proj
    num_attention_heads = 40
    head_dim = 128

    # Row-Parallel layer shapes
    # MLP Down-proj: [*, intermediate_size] @ [intermediate_size, hidden_size]
    # Attention Out-proj: [*, hidden_size] @ [hidden_size, hidden_size]


# ============================================================================
# BASELINE IMPLEMENTATION (No Overlap)
# ============================================================================

def forward_baseline(
    x: torch.Tensor,
    weight: torch.Tensor,
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Baseline Row-Parallel forward pass without overlap optimization.

    Steps:
        1. Compute partial_y = x @ weight.T (local GEMM on each GPU)
        2. Perform blocking all_reduce to sum across TP group
        3. Return final result

    Communication Stall:
        GPU is idle during all_reduce operation.

    Args:
        x: Input tensor [batch_size, seq_len, in_features]
        weight: Weight matrix [out_features, in_features] (sharded across TP)
        process_group: Distributed process group for all_reduce

    Returns:
        Output tensor [batch_size, seq_len, out_features] after all_reduce
    """
    # Step 1: Local GEMM computation
    # Each GPU computes: partial_y = x @ weight_local.T
    partial_y = torch.matmul(x, weight.t())

    # Step 2: Blocking all_reduce (COMMUNICATION STALL)
    # GPU waits here until all GPUs complete their all_reduce
    dist.all_reduce(partial_y, op=dist.ReduceOp.SUM, group=process_group)

    return partial_y


# ============================================================================
# OVERLAP IMPLEMENTATION (Computation-Communication Pipelining)
# ============================================================================

def forward_overlap(
    x: torch.Tensor,
    weight: torch.Tensor,
    num_chunks: int = 4,
    process_group: Optional[dist.ProcessGroup] = None,
    compute_stream: Optional[torch.cuda.Stream] = None,
    comm_stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """
    Optimized Row-Parallel forward pass with computation-communication overlap.

    Pipeline Strategy:
        Chunk 0: [Compute] ──→ [All-Reduce]
        Chunk 1:                [Compute] ──→ [All-Reduce]
        Chunk 2:                                [Compute] ──→ [All-Reduce]
        Chunk 3:                                                [Compute] ──→ [All-Reduce]

        Key: While Chunk i is reducing, Chunk i+1 is computing (OVERLAP)

    Synchronization:
        - compute_events: Ensure GEMM completes before all_reduce starts
        - comm_events: Ensure all_reduce completes before output is used

    Args:
        x: Input tensor [batch_size, seq_len, in_features]
        weight: Weight matrix [out_features, in_features]
        num_chunks: Number of chunks to split computation
        process_group: Distributed process group
        compute_stream: CUDA stream for computation
        comm_stream: CUDA stream for communication

    Returns:
        Output tensor [batch_size, seq_len, out_features] after all_reduce
    """
    device = x.device
    batch_size, seq_len, in_features = x.shape
    out_features = weight.shape[0]

    # Create streams if not provided
    if compute_stream is None:
        compute_stream = torch.cuda.Stream(device=device)
    if comm_stream is None:
        comm_stream = torch.cuda.Stream(device=device)

    # Flatten to 2D for easier chunking: [total_tokens, in_features]
    total_tokens = batch_size * seq_len
    x_flat = x.view(total_tokens, in_features)

    # Calculate chunk size (split along token dimension)
    chunk_size = (total_tokens + num_chunks - 1) // num_chunks

    # Pre-allocate output buffer
    output = torch.empty(total_tokens, out_features, device=device, dtype=x.dtype)

    # Pre-allocate synchronization events
    compute_events = [torch.cuda.Event() for _ in range(num_chunks)]
    comm_events = [torch.cuda.Event() for _ in range(num_chunks)]

    # ========================================================================
    # PIPELINED EXECUTION LOOP
    # ========================================================================
    actual_chunks = 0
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_tokens)

        if start_idx >= total_tokens:
            break

        actual_chunks += 1

        # Extract chunk slices
        x_chunk = x_flat[start_idx:end_idx]
        output_chunk = output[start_idx:end_idx]

        # ====================================================================
        # STAGE 1: COMPUTE (on compute_stream)
        # ====================================================================
        with torch.cuda.stream(compute_stream):
            # Perform local GEMM for this chunk
            torch.matmul(x_chunk, weight.t(), out=output_chunk)

            # Record event: "Computation of chunk_idx is complete"
            compute_events[chunk_idx].record(compute_stream)

        # ====================================================================
        # STAGE 2: COMMUNICATE (on comm_stream, OVERLAPPED with next compute)
        # ====================================================================
        with torch.cuda.stream(comm_stream):
            # Wait for computation to finish before starting communication
            # This ensures we don't reduce incomplete data
            comm_stream.wait_event(compute_events[chunk_idx])

            # Asynchronous all_reduce on comm_stream
            # While this executes, compute_stream can start chunk i+1
            # This is where OVERLAP happens!
            dist.all_reduce(output_chunk, op=dist.ReduceOp.SUM, group=process_group)

            # Record event: "All-reduce of chunk_idx is complete"
            comm_events[chunk_idx].record(comm_stream)

    # ========================================================================
    # FINAL SYNCHRONIZATION
    # ========================================================================
    # Ensure all communication completes before returning output
    for idx in range(actual_chunks):
        comm_events[idx].synchronize()

    # Reshape back to original 3D dimensions
    return output.view(batch_size, seq_len, out_features)


# ============================================================================
# VALIDATION FUNCTION
# ============================================================================

def validate_correctness(
    x: torch.Tensor,
    weight: torch.Tensor,
    num_chunks: int = 4,
    process_group: Optional[dist.ProcessGroup] = None,
    tolerance: float = 1e-4,
) -> Tuple[bool, float]:
    """
    Validate that overlap implementation produces identical results to baseline.

    Args:
        x: Input tensor
        weight: Weight matrix
        num_chunks: Number of chunks for overlap version
        process_group: Distributed process group
        tolerance: Maximum allowed difference

    Returns:
        (is_correct, max_difference)
    """
    with torch.no_grad():
        # Run baseline
        baseline_output = forward_baseline(x.clone(), weight, process_group)

        # Run overlap
        overlap_output = forward_overlap(
            x.clone(), weight, num_chunks, process_group
        )

        # Compute difference
        max_diff = torch.max(torch.abs(baseline_output - overlap_output)).item()
        is_correct = max_diff < tolerance

        return is_correct, max_diff


# ============================================================================
# BENCHMARKING FUNCTION
# ============================================================================

def benchmark_forward(
    forward_fn,
    x: torch.Tensor,
    weight: torch.Tensor,
    num_warmup: int = 10,
    num_iterations: int = 50,
    **kwargs,
) -> float:
    """
    Benchmark a forward function using CUDA events for accurate GPU timing.

    Args:
        forward_fn: Function to benchmark
        x: Input tensor
        weight: Weight matrix
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations
        **kwargs: Additional arguments for forward_fn

    Returns:
        Average latency in milliseconds
    """
    with torch.no_grad():
        # Warmup phase
        for _ in range(num_warmup):
            _ = forward_fn(x, weight, **kwargs)
        torch.cuda.synchronize()

        # Timed phase using CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iterations):
            _ = forward_fn(x, weight, **kwargs)
        end_event.record()

        torch.cuda.synchronize()

        # Calculate average latency
        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = total_time_ms / num_iterations

        return avg_latency_ms


# ============================================================================
# DISTRIBUTED SETUP
# ============================================================================

def setup_distributed(rank: int, world_size: int):
    """Initialize PyTorch distributed with NCCL backend."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed process group."""
    dist.destroy_process_group()


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_comparative_experiment(rank: int, world_size: int):
    """
    Run comprehensive comparative experiment with Qwen-32B dimensions.

    Experiment Matrix:
        - Layer Types: MLP Down-proj, Attention Out-proj
        - Sequence Lengths: 512, 1024, 2048, 4096
        - Chunk Sizes: 4 (default)

    Args:
        rank: GPU rank
        world_size: Number of GPUs
    """
    setup_distributed(rank, world_size)

    device = f"cuda:{rank}"
    config = Qwen32BConfig()

    # Experiment parameters
    batch_size = 1  # Typical inference batch size
    sequence_lengths = [512, 1024, 2048, 4096]
    num_chunks = 4

    # Layer configurations to test
    layer_configs = [
        {
            "name": "MLP Down-Projection",
            "in_features": config.intermediate_size,
            "out_features": config.hidden_size,
        },
        {
            "name": "Attention Out-Projection",
            "in_features": config.hidden_size,
            "out_features": config.hidden_size,
        },
    ]

    if rank == 0:
        print("=" * 100)
        print("QWEN-32B TENSOR PARALLEL ROW-PARALLEL COMPARATIVE EXPERIMENT")
        print("=" * 100)
        print(f"\nConfiguration:")
        print(f"  World Size: {world_size} GPUs")
        print(f"  Batch Size: {batch_size}")
        print(f"  Num Chunks: {num_chunks}")
        print(f"  Sequence Lengths: {sequence_lengths}")
        print("=" * 100)

    # Run experiments for each layer type
    for layer_config in layer_configs:
        layer_name = layer_config["name"]
        in_features = layer_config["in_features"]
        out_features = layer_config["out_features"]

        if rank == 0:
            print(f"\n{'=' * 100}")
            print(f"LAYER: {layer_name}")
            print(f"Shape: [batch, seq_len, {in_features}] @ [{out_features}, {in_features}]")
            print(f"{'=' * 100}")
            print(f"\n{'BatchSize':<12} {'SeqLen':<10} {'Baseline(ms)':<15} {'Overlap(ms)':<15} "
                  f"{'Speedup':<12} {'Hidden%':<12} {'Validation':<12}")
            print("-" * 100)

        # Create weight matrix (same for all sequence lengths)
        torch.manual_seed(42 + rank)
        weight = torch.randn(
            out_features, in_features, device=device, dtype=torch.float32
        )

        # Test each sequence length
        for seq_len in sequence_lengths:
            # Create input tensor
            torch.manual_seed(42 + rank + seq_len)
            x = torch.randn(
                batch_size, seq_len, in_features, device=device, dtype=torch.float32
            )

            # Validate correctness
            is_correct, max_diff = validate_correctness(
                x, weight, num_chunks, process_group=None, tolerance=1e-4
            )

            # Benchmark baseline
            baseline_latency = benchmark_forward(
                forward_baseline,
                x,
                weight,
                num_warmup=10,
                num_iterations=50,
                process_group=None,
            )

            # Benchmark overlap
            overlap_latency = benchmark_forward(
                forward_overlap,
                x,
                weight,
                num_warmup=10,
                num_iterations=50,
                num_chunks=num_chunks,
                process_group=None,
            )

            # Calculate metrics
            speedup = (baseline_latency / overlap_latency - 1.0) * 100  # Percentage improvement
            hidden_ratio = ((baseline_latency - overlap_latency) / baseline_latency) * 100

            validation_status = "✓ PASS" if is_correct else f"✗ FAIL (diff={max_diff:.2e})"

            if rank == 0:
                print(f"{batch_size:<12} {seq_len:<10} {baseline_latency:<15.3f} "
                      f"{overlap_latency:<15.3f} {speedup:<12.1f}% {hidden_ratio:<12.1f}% "
                      f"{validation_status:<12}")

        if rank == 0:
            print("-" * 100)

    if rank == 0:
        print("\n" + "=" * 100)
        print("EXPERIMENT COMPLETE")
        print("=" * 100)
        print("\nKey Metrics:")
        print("  - Speedup: Percentage improvement over baseline")
        print("  - Hidden%: Percentage of communication latency hidden by overlap")
        print("  - Validation: Numerical correctness check (tolerance=1e-4)")
        print("\nObservations:")
        print("  - Longer sequences → More overlap benefit (larger chunks)")
        print("  - Speedup depends on compute/communication ratio")
        print("  - NVLink provides better overlap than PCIe")
        print("=" * 100)

    cleanup_distributed()


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for comparative experiment.

    Usage:
        # Run on 2 GPUs:
        torchrun --nproc_per_node=2 qwen_comparative_experiment.py

        # Run on 4 GPUs:
        torchrun --nproc_per_node=4 qwen_comparative_experiment.py

        # Run on 8 GPUs:
        torchrun --nproc_per_node=8 qwen_comparative_experiment.py
    """
    if not dist.is_available():
        raise RuntimeError("PyTorch distributed is not available")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # Get rank and world_size from environment (set by torchrun)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size < 2:
        raise ValueError(
            "This experiment requires at least 2 GPUs.\n"
            "Run with: torchrun --nproc_per_node=2 qwen_comparative_experiment.py"
        )

    run_comparative_experiment(rank, world_size)


if __name__ == "__main__":
    main()

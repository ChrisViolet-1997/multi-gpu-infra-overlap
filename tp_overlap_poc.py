#!/usr/bin/env python3
"""
Tensor Parallel Row-Parallel Layer with Computation-Communication Overlap PoC

This script demonstrates how to hide all_reduce communication latency by overlapping
it with subsequent GEMM computation through chunked execution and multi-stream pipelining.

Architecture:
    - Split large GEMM (Y = A × B) along M dimension into K chunks
    - Use separate CUDA streams for computation and communication
    - Pipeline: compute chunk i+1 while all_reduce chunk i is in flight
    - Synchronize using CUDA events to maintain correctness without blocking

Author: Senior AI Infrastructure Engineer
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Optional, Tuple
import time
import os


class BaselineRowParallelLinear(nn.Module):
    """
    Standard Row-Parallel Linear layer without overlap optimization.

    Forward pass:
        1. Compute full GEMM: Y_local = X @ W_local^T
        2. Wait for GEMM to complete
        3. Perform blocking all_reduce across TP group
        4. Return result

    This creates a communication stall where GPU is idle during all_reduce.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: Optional[dist.ProcessGroup] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.process_group = process_group

        # Each GPU holds a shard of the weight matrix along output dimension
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass with communication stall.

        Args:
            x: Input tensor [batch_size, seq_len, in_features]

        Returns:
            Output tensor [batch_size, seq_len, out_features] after all_reduce
        """
        # Step 1: Compute full local GEMM
        # Each GPU computes Y_local = X @ W_local^T
        output = torch.matmul(x, self.weight.t())

        # Step 2: Blocking all_reduce - GPU waits here (COMMUNICATION STALL)
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.process_group)

        return output


class OverlapRowParallelLinear(nn.Module):
    """
    Optimized Row-Parallel Linear with Computation-Communication Overlap.

    Key Innovation:
        - Split GEMM into K chunks along batch dimension
        - Use compute_stream for matrix multiplications
        - Use comm_stream for all_reduce operations
        - Pipeline: while chunk i is reducing, compute chunk i+1

    Synchronization Strategy:
        - CUDA events ensure chunk i computation completes before its all_reduce starts
        - Events ensure chunk i all_reduce completes before chunk i is used downstream
        - No global synchronization until final output assembly
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_chunks: int = 4,
        process_group: Optional[dist.ProcessGroup] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_chunks = num_chunks
        self.process_group = process_group
        self.device = device

        # Weight matrix (same as baseline)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device, dtype=torch.float32)
        )

        # Create dedicated CUDA streams for overlap
        self.compute_stream = torch.cuda.Stream(device=device)
        self.comm_stream = torch.cuda.Stream(device=device)

        # Pre-allocate CUDA events for synchronization
        # We need 2 events per chunk: compute_done and comm_done
        self.compute_events = [
            torch.cuda.Event() for _ in range(num_chunks)
        ]
        self.comm_events = [
            torch.cuda.Event() for _ in range(num_chunks)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overlapped forward pass using chunked GEMM and pipelined all_reduce.

        Pipeline Schedule:
            Chunk 0: [Compute] -> [All-Reduce]
            Chunk 1:              [Compute] -> [All-Reduce]
            Chunk 2:                           [Compute] -> [All-Reduce]
            ...

        Args:
            x: Input tensor [batch_size, seq_len, in_features]

        Returns:
            Output tensor [batch_size, seq_len, out_features] after all_reduce
        """
        batch_size, seq_len, _ = x.shape
        total_tokens = batch_size * seq_len

        # Reshape to 2D for easier chunking: [total_tokens, in_features]
        x_flat = x.view(-1, self.in_features)

        # Calculate chunk size (split along token dimension)
        chunk_size = (total_tokens + self.num_chunks - 1) // self.num_chunks

        # Pre-allocate output buffer
        output = torch.empty(
            total_tokens, self.out_features, device=self.device, dtype=x.dtype
        )

        # === PIPELINED EXECUTION ===
        for chunk_idx in range(self.num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_tokens)

            if start_idx >= total_tokens:
                break

            # Extract chunk from input
            x_chunk = x_flat[start_idx:end_idx]
            output_chunk = output[start_idx:end_idx]

            # --- STAGE 1: COMPUTE CHUNK ---
            with torch.cuda.stream(self.compute_stream):
                # Compute local GEMM for this chunk
                torch.matmul(x_chunk, self.weight.t(), out=output_chunk)

                # Record event: "Computation of chunk_idx is complete"
                self.compute_events[chunk_idx].record(self.compute_stream)

            # --- STAGE 2: ALL-REDUCE CHUNK (OVERLAPPED) ---
            with torch.cuda.stream(self.comm_stream):
                # Wait for computation to finish before starting communication
                self.comm_stream.wait_event(self.compute_events[chunk_idx])

                # Asynchronous all_reduce on comm_stream
                # While this runs, compute_stream can start chunk i+1
                dist.all_reduce(
                    output_chunk, op=dist.ReduceOp.SUM, group=self.process_group
                )

                # Record event: "All-reduce of chunk_idx is complete"
                self.comm_events[chunk_idx].record(self.comm_stream)

        # === FINAL SYNCHRONIZATION ===
        # Ensure all communication completes before returning
        for event in self.comm_events[:self.num_chunks]:
            event.synchronize()

        # Reshape back to original dimensions
        return output.view(batch_size, seq_len, self.out_features)


def setup_distributed(rank: int, world_size: int):
    """
    Initialize PyTorch distributed process group with NCCL backend.

    Args:
        rank: GPU rank (0 to world_size-1)
        world_size: Total number of GPUs
    """
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


def benchmark_layer(
    layer: nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 10,
    num_iterations: int = 50,
) -> float:
    """
    Benchmark a layer's forward pass using CUDA events for accurate GPU timing.

    Args:
        layer: The layer to benchmark
        input_tensor: Input tensor for forward pass
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations

    Returns:
        Average latency in milliseconds
    """
    # Warmup phase to stabilize GPU clocks and cache
    for _ in range(num_warmup):
        _ = layer(input_tensor)
    torch.cuda.synchronize()

    # Timed phase using CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iterations):
        _ = layer(input_tensor)
    end_event.record()

    torch.cuda.synchronize()

    # Calculate average latency
    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / num_iterations

    return avg_latency_ms


def run_benchmark(rank: int, world_size: int):
    """
    Main benchmark function comparing baseline vs overlap implementations.

    Args:
        rank: GPU rank
        world_size: Total number of GPUs
    """
    setup_distributed(rank, world_size)

    # === CONFIGURATION ===
    batch_size = 8
    seq_len = 2048
    in_features = 4096
    out_features = 4096
    num_chunks = 4

    device = f"cuda:{rank}"

    if rank == 0:
        print("=" * 80)
        print("TENSOR PARALLEL ROW-PARALLEL OVERLAP BENCHMARK")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  - World Size: {world_size} GPUs")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - Sequence Length: {seq_len}")
        print(f"  - Input Features: {in_features}")
        print(f"  - Output Features: {out_features}")
        print(f"  - Num Chunks: {num_chunks}")
        print("=" * 80)

    # Create input tensor (same for both layers)
    torch.manual_seed(42 + rank)
    input_tensor = torch.randn(
        batch_size, seq_len, in_features, device=device, dtype=torch.float32
    )

    # === BASELINE LAYER ===
    baseline_layer = BaselineRowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        process_group=None,  # Use default group
        device=device,
    )

    if rank == 0:
        print("\n[1/2] Benchmarking Baseline (No Overlap)...")

    baseline_latency = benchmark_layer(
        baseline_layer, input_tensor, num_warmup=10, num_iterations=50
    )

    # === OVERLAP LAYER ===
    overlap_layer = OverlapRowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        num_chunks=num_chunks,
        process_group=None,
        device=device,
    )

    if rank == 0:
        print("[2/2] Benchmarking Overlap (Pipelined)...")

    overlap_latency = benchmark_layer(
        overlap_layer, input_tensor, num_warmup=10, num_iterations=50
    )

    # === RESULTS ===
    if rank == 0:
        speedup = baseline_latency / overlap_latency
        hidden_latency_pct = ((baseline_latency - overlap_latency) / baseline_latency) * 100

        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Baseline Latency:        {baseline_latency:.3f} ms")
        print(f"Overlap Latency:         {overlap_latency:.3f} ms")
        print(f"Speedup:                 {speedup:.2f}x")
        print(f"Hidden Latency:          {hidden_latency_pct:.1f}%")
        print("=" * 80)

        if speedup > 1.0:
            print(f"\n✓ SUCCESS: Achieved {speedup:.2f}x speedup through overlap!")
            print(f"  Communication latency hidden: {hidden_latency_pct:.1f}%")
        else:
            print("\n⚠ WARNING: No speedup observed. Possible reasons:")
            print("  - Computation time >> communication time (compute-bound)")
            print("  - Insufficient chunk granularity")
            print("  - Hardware limitations (PCIe vs NVLink)")

    cleanup_distributed()


def main():
    """
    Entry point for multi-GPU benchmark.

    Usage:
        # Single-node, 2 GPUs:
        torchrun --nproc_per_node=2 tp_overlap_poc.py

        # Single-node, 4 GPUs:
        torchrun --nproc_per_node=4 tp_overlap_poc.py
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
            "This benchmark requires at least 2 GPUs. "
            "Run with: torchrun --nproc_per_node=2 tp_overlap_poc.py"
        )

    run_benchmark(rank, world_size)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Optimized Overlap with Double Buffering

Key insight: The wait_event in the original code prevents overlap because:
- Chunk i's compute waits for chunk i-1's comm to finish
- This creates sequential execution: [Compute_0][Comm_0][Compute_1][Comm_1]...

Problem: We can't remove the wait because of data hazard:
- Chunk i-1's comm is reading from output buffer
- Chunk i's compute would write to the same buffer
- This causes a race condition (dirty write)

Solution: Double buffering
- Use 2 output buffers that alternate
- Chunk i writes to buffer A while chunk i-1's comm reads from buffer B
- No data hazard, true overlap achieved!
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Optional
import os


class DoubleBufferOverlapRowParallelLinear(nn.Module):
    """
    Row-Parallel Linear with Double Buffer for True Overlap.

    Architecture:
        - 2 output buffers (buffer_0 and buffer_1)
        - Chunk i uses buffer[i % 2]
        - While comm reads from buffer A, compute writes to buffer B
        - Achieves true overlap without data hazards
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

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device, dtype=torch.float32)
        )

        # Separate streams for compute and communication
        self.compute_stream = torch.cuda.Stream(device=device)
        self.comm_stream = torch.cuda.Stream(device=device)

        # Events for synchronization
        self.compute_events = [torch.cuda.Event() for _ in range(num_chunks)]
        self.comm_events = [torch.cuda.Event() for _ in range(num_chunks)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with double buffering for true overlap.

        Timeline (with double buffer):
            Chunk 0: [Compute -> Buffer_0] -> [Comm reads Buffer_0]
            Chunk 1:                          [Compute -> Buffer_1] (overlaps with Chunk 0 Comm)
                                              -> [Comm reads Buffer_1]
            Chunk 2:                                                  [Compute -> Buffer_0] (overlaps with Chunk 1 Comm)

        Key: No wait between chunks! Each chunk uses a different buffer.
        """
        batch_size, seq_len, _ = x.shape
        total_tokens = batch_size * seq_len

        x_flat = x.view(-1, self.in_features)
        chunk_size = (total_tokens + self.num_chunks - 1) // self.num_chunks

        # Allocate TWO output buffers for double buffering
        buffer_0 = torch.empty(
            total_tokens, self.out_features, device=self.device, dtype=x.dtype
        )
        buffer_1 = torch.empty(
            total_tokens, self.out_features, device=self.device, dtype=x.dtype
        )
        buffers = [buffer_0, buffer_1]

        # Final output buffer (will copy from double buffers)
        output = torch.empty(
            total_tokens, self.out_features, device=self.device, dtype=x.dtype
        )

        actual_chunks = 0
        for chunk_idx in range(self.num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_tokens)

            if start_idx >= total_tokens:
                break

            actual_chunks += 1

            # Select buffer for this chunk (alternating)
            buffer_idx = chunk_idx % 2
            current_buffer = buffers[buffer_idx]

            x_chunk = x_flat[start_idx:end_idx]
            buffer_chunk = current_buffer[start_idx:end_idx]
            output_chunk = output[start_idx:end_idx]

            # --- STAGE 1: COMPUTE CHUNK ---
            with torch.cuda.stream(self.compute_stream):
                # REMOVED: No wait for previous comm!
                # Different chunks use different buffers, so no data hazard

                # Compute GEMM into the current buffer
                torch.matmul(x_chunk, self.weight.t(), out=buffer_chunk)

                # Record: computation done
                self.compute_events[chunk_idx].record(self.compute_stream)

            # --- STAGE 2: ALL-REDUCE CHUNK (OVERLAPPED) ---
            with torch.cuda.stream(self.comm_stream):
                # Wait for THIS chunk's computation to finish
                self.comm_stream.wait_event(self.compute_events[chunk_idx])

                # All-reduce on the buffer
                # While this runs, next chunk can compute into the OTHER buffer!
                dist.all_reduce(
                    buffer_chunk, op=dist.ReduceOp.SUM, group=self.process_group
                )

                # Record: communication done
                self.comm_events[chunk_idx].record(self.comm_stream)

            # --- STAGE 3: COPY TO OUTPUT (AFTER COMM COMPLETES) ---
            # We need to copy from buffer to output after comm completes
            # This ensures the final output has all chunks in the right place
            with torch.cuda.stream(self.comm_stream):
                # Already on comm_stream, so this happens after all_reduce
                output_chunk.copy_(buffer_chunk)

        # === FINAL SYNCHRONIZATION ===
        # Wait for all communication to complete
        for i in range(actual_chunks):
            self.comm_events[i].synchronize()

        return output.view(batch_size, seq_len, self.out_features)


# Import baseline and original overlap for comparison
import sys
sys.path.insert(0, '/root/autodl-tmp/multi-gpu-infra-overlap')
from tp_overlap_poc import (
    BaselineRowParallelLinear,
    OverlapRowParallelLinear,
    setup_distributed,
    cleanup_distributed,
    benchmark_layer,
)


def run_benchmark(
    rank: int,
    world_size: int,
    batch_size: int = 1,
    seq_len: int = 2048,
    out_features: int = 12288,
    in_features: int = 4096,
    num_chunks: int = 4,
):
    """Benchmark baseline vs original overlap vs double buffer overlap."""
    setup_distributed(rank, world_size)

    device = f"cuda:{rank}"

    if rank == 0:
        print("=" * 80)
        print("DOUBLE BUFFER OVERLAP BENCHMARK")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  - World Size: {world_size} GPUs")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - Sequence Length: {seq_len}")
        print(f"  - Input Features: {in_features}")
        print(f"  - Output Features: {out_features}")
        print(f"  - Num Chunks: {num_chunks}")
        print("=" * 80)

    # Create input
    torch.manual_seed(42 + rank)
    input_tensor = torch.randn(
        batch_size, seq_len, in_features, device=device, dtype=torch.float32
    )

    # === BASELINE ===
    baseline_layer = BaselineRowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        process_group=None,
        device=device,
    )

    if rank == 0:
        print("\n[1/3] Benchmarking Baseline (No Overlap)...")

    baseline_latency = benchmark_layer(
        baseline_layer, input_tensor, num_warmup=10, num_iterations=50
    )

    # === ORIGINAL OVERLAP (with wait) ===
    overlap_layer = OverlapRowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        num_chunks=num_chunks,
        process_group=None,
        device=device,
    )

    if rank == 0:
        print("[2/3] Benchmarking Original Overlap (with wait_event)...")

    overlap_latency = benchmark_layer(
        overlap_layer, input_tensor, num_warmup=10, num_iterations=50
    )

    # === DOUBLE BUFFER OVERLAP ===
    double_buffer_layer = DoubleBufferOverlapRowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        num_chunks=num_chunks,
        process_group=None,
        device=device,
    )

    if rank == 0:
        print("[3/3] Benchmarking Double Buffer Overlap...")

    double_buffer_latency = benchmark_layer(
        double_buffer_layer, input_tensor, num_warmup=10, num_iterations=50
    )

    # === RESULTS ===
    if rank == 0:
        speedup_original = baseline_latency / overlap_latency
        speedup_double_buffer = baseline_latency / double_buffer_latency

        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Baseline Latency:              {baseline_latency:.3f} ms")
        print(f"Original Overlap Latency:      {overlap_latency:.3f} ms")
        print(f"Double Buffer Overlap Latency: {double_buffer_latency:.3f} ms")
        print(f"\nOriginal Overlap Speedup:      {speedup_original:.2f}x")
        print(f"Double Buffer Speedup:         {speedup_double_buffer:.2f}x")
        print("=" * 80)

        if speedup_double_buffer > speedup_original:
            improvement = (speedup_double_buffer - speedup_original) / speedup_original * 100
            print(f"\n✓ Double Buffer is {improvement:.1f}% faster than Original Overlap!")

        if speedup_double_buffer > 1.1:
            print(f"\n✓ SUCCESS: Double Buffer achieved {speedup_double_buffer:.2f}x speedup!")
        else:
            print(f"\n⚠ Limited speedup. Possible reasons:")
            print(f"  - Communication time >> computation time")
            print(f"  - Try larger batch_size or seq_len")

    cleanup_distributed()


def main():
    if not dist.is_available():
        raise RuntimeError("PyTorch distributed is not available")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size < 2:
        raise ValueError(
            "This benchmark requires at least 2 GPUs. "
            "Run with: torchrun --nproc_per_node=2 tp_overlap_double_buffer.py"
        )

    # Get configuration from environment
    batch_size = int(os.environ.get("BATCH_SIZE", 1))
    seq_len = int(os.environ.get("SEQ_LEN", 2048))
    out_features = int(os.environ.get("OUT_FEATURES", 12288))
    in_features = int(os.environ.get("IN_FEATURES", 4096))
    num_chunks = int(os.environ.get("NUM_CHUNKS", 4))

    run_benchmark(
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
        seq_len=seq_len,
        out_features=out_features,
        in_features=in_features,
        num_chunks=num_chunks,
    )


if __name__ == "__main__":
    main()

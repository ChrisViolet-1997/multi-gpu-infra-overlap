#!/usr/bin/env python3
"""
Standard RowParallelLinear with Chunking for Computation-Communication Overlap

Key concepts:
1. RowParallelLinear: Each rank stores [out_features, in_features/world_size] weights
2. Input sharding: Each rank processes its own input shard
3. Chunking: Split output dimension into chunks for overlap
4. All-reduce: Sum partial results across ranks

Timeline with chunking:
    Chunk 0: [Compute] -> [All-reduce] (overlaps with Chunk 1 compute)
    Chunk 1:              [Compute] -> [All-reduce] (overlaps with Chunk 2 compute)
    ...
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Optional


class DoubleBufferOverlapRowParallelLinear(nn.Module):
    """
    Standard RowParallelLinear with chunking for computation-communication overlap.

    Architecture:
        - Weight: [out_features, in_features/world_size] per rank
        - Input: Full [batch, seq, in_features], extract own shard
        - Chunking: Split output dimension for overlap
        - All-reduce: Sum partial results
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

        # Get TP world size
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # RowParallelLinear: shard input dimension
        assert in_features % self.world_size == 0, \
            f"in_features ({in_features}) must be divisible by world_size ({self.world_size})"
        self.in_features_per_rank = in_features // self.world_size

        # Weight: [out_features, in_features_per_rank]
        self.weight = nn.Parameter(
            torch.randn(out_features, self.in_features_per_rank, device=device, dtype=torch.float32)
        )

        # Calculate chunk size (in output dimension)
        assert out_features % num_chunks == 0, \
            f"out_features ({out_features}) must be divisible by num_chunks ({num_chunks})"
        self.chunk_size = out_features // num_chunks

        # Separate streams for compute and communication
        self.compute_stream = torch.cuda.Stream(device=device)
        self.comm_stream = torch.cuda.Stream(device=device)

        # Events for synchronization
        self.compute_events = [torch.cuda.Event() for _ in range(num_chunks)]
        self.comm_events = [torch.cuda.Event() for _ in range(num_chunks)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with chunking for overlap.

        Args:
            x: Input [batch, seq, in_features] (full, not sharded)

        Returns:
            output: [batch, seq, out_features] (full, after all-reduce)
        """
        batch_size, seq_len, _ = x.shape
        total_tokens = batch_size * seq_len

        # Extract this rank's input shard
        start_idx = self.rank * self.in_features_per_rank
        end_idx = (self.rank + 1) * self.in_features_per_rank
        x_shard = x[:, :, start_idx:end_idx]
        x_flat = x_shard.reshape(-1, self.in_features_per_rank)

        # Output buffer
        output = torch.empty(total_tokens, self.out_features, device=self.device, dtype=x.dtype)

        # Process each chunk
        for chunk_idx in range(self.num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = (chunk_idx + 1) * self.chunk_size

            # Get weight chunk
            weight_chunk = self.weight[chunk_start:chunk_end, :]

            # Get output chunk
            output_chunk = output[:, chunk_start:chunk_end]

            # --- STAGE 1: COMPUTE ---
            with torch.cuda.stream(self.compute_stream):
                # Compute: Y_local = X_shard @ W_chunk.T
                torch.matmul(x_flat, weight_chunk.t(), out=output_chunk)

                # Record: computation done
                self.compute_events[chunk_idx].record(self.compute_stream)

            # --- STAGE 2: ALL-REDUCE (OVERLAPPED) ---
            with torch.cuda.stream(self.comm_stream):
                # Wait for this chunk's computation
                self.comm_stream.wait_event(self.compute_events[chunk_idx])

                # All-reduce: Y = sum(Y_local) across ranks
                if self.process_group is not None:
                    dist.all_reduce(output_chunk, op=dist.ReduceOp.SUM, group=self.process_group)

                # Record: communication done
                self.comm_events[chunk_idx].record(self.comm_stream)

        # Wait for all communication to complete
        for i in range(self.num_chunks):
            self.comm_events[i].synchronize()

        return output.view(batch_size, seq_len, self.out_features)


def main():
    """Test the RowParallelLinear implementation."""
    print("=" * 80)
    print("ROWPARALLELLINEAR WITH CHUNKING TEST")
    print("=" * 80)

    # Setup distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    # Configuration
    batch_size = 32
    seq_len = 1
    in_features = 4096
    out_features = 4096
    num_chunks = 4

    if local_rank == 0:
        print(f"\nConfiguration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Seq len: {seq_len}")
        print(f"  In features: {in_features}")
        print(f"  Out features: {out_features}")
        print(f"  Num chunks: {num_chunks}")
        print(f"  World size: {world_size}")

    # Create layer
    layer = DoubleBufferOverlapRowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        num_chunks=num_chunks,
        device=device,
    )

    if local_rank == 0:
        print(f"\nLayer info:")
        print(f"  Weight shape: {layer.weight.shape}")
        print(f"  Expected: [{out_features}, {in_features // world_size}]")

    # Create input
    x = torch.randn(batch_size, seq_len, in_features, device=device, requires_grad=False)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = layer(x)
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for _ in range(50):
            output = layer(x)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event) / 50

    if local_rank == 0:
        print(f"\nResults:")
        print(f"  Output shape: {output.shape}")
        print(f"  Average time: {elapsed_time:.3f} ms")

    dist.destroy_process_group()


if __name__ == "__main__":
    import os
    main()

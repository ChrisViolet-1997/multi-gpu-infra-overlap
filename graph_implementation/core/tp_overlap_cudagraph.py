#!/usr/bin/env python3
"""
CUDAGraph-optimized version of DoubleBufferOverlapRowParallelLinear.

This module provides a graph-accelerated version that eliminates kernel launch overhead.
The original implementation remains unchanged.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional
import warnings


class CUDAGraphDoubleBufferOverlapRowParallelLinear(nn.Module):
    """
    CUDAGraph-optimized TP overlap linear layer.

    Key differences from original:
    - Pre-allocates all buffers during initialization
    - Captures forward pass as CUDAGraph
    - Eliminates per-call kernel launch overhead

    Constraints:
    - Input shape must be fixed (batch_size, seq_len, in_features)
    - Cannot change num_chunks after graph capture
    - Requires warmup before graph capture
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_chunks: int = 4,
        process_group: Optional[dist.ProcessGroup] = None,
        device: str = "cuda",
        # CUDAGraph specific parameters
        static_input_shape: Optional[tuple] = None,  # (batch, seq, in_features)
        enable_graph: bool = True,
        sequential: bool = False,  # If True, disable overlap (compute then communicate)
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_chunks = num_chunks
        self.process_group = process_group
        self.device = device
        self.enable_graph = enable_graph
        self.static_input_shape = static_input_shape
        self.sequential = sequential  # Shadow execution mode

        # Get TP world size
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # RowParallelLinear: shard input dimension
        assert in_features % self.world_size == 0, \
            f"in_features ({in_features}) must be divisible by world_size ({self.world_size})"
        self.in_features_per_rank = in_features // self.world_size

        # Weight parameter: [out_features, in_features_per_rank]
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_rank, device=device)
        )
        nn.init.xavier_uniform_(self.weight)

        # Calculate chunk size
        assert out_features % num_chunks == 0, \
            f"out_features ({out_features}) must be divisible by num_chunks ({num_chunks})"
        self.chunk_size = out_features // num_chunks

        # Create separate CUDA stream for communication (always create for overlap)
        self.comm_stream = torch.cuda.Stream()

        # Pre-allocate buffers if static shape is provided
        self.static_buffers = None
        self.graph = None
        self.graph_captured = False

        if static_input_shape is not None and enable_graph:
            self._preallocate_buffers(static_input_shape)

    def _preallocate_buffers(self, input_shape: tuple):
        """Pre-allocate all buffers for CUDAGraph."""
        batch, seq, _ = input_shape

        # Single output buffer (chunks write directly to their slices)
        buffer_shape = (batch, seq, self.out_features)
        self.static_buffers = {
            'final_output': torch.empty(buffer_shape, device=self.device),
            'x_shard': torch.empty(batch, seq, self.in_features_per_rank, device=self.device),
        }

        # Pre-allocate contiguous buffers for each chunk (for communication)
        self.chunk_buffers = []
        for i in range(self.num_chunks):
            chunk_buffer = torch.empty(batch * seq, self.chunk_size, device=self.device)
            self.chunk_buffers.append(chunk_buffer)

    def _capture_graph(self, x: torch.Tensor):
        """Capture computation graphs for each chunk separately."""
        if self.graph_captured:
            return

        if self.static_buffers is None:
            warnings.warn("Cannot capture graph without static buffers. Call with static_input_shape.")
            return

        print(f"Capturing CUDAGraph for shape {x.shape}, chunks={self.num_chunks}...")

        batch, seq, _ = x.shape

        # Extract this rank's input shard
        start_idx = self.rank * self.in_features_per_rank
        end_idx = (self.rank + 1) * self.in_features_per_rank
        x_shard = x[:, :, start_idx:end_idx]

        # Copy to static buffer
        self.static_buffers['x_shard'].copy_(x_shard)

        x_flat = self.static_buffers['x_shard'].view(-1, self.in_features_per_rank)
        output_flat = self.static_buffers['final_output'].view(-1, self.out_features)

        # Capture each chunk's computation as a separate graph
        self.compute_graphs = []

        for i in range(self.num_chunks):
            chunk_start = i * self.chunk_size
            chunk_end = (i + 1) * self.chunk_size
            weight_chunk = self.weight[chunk_start:chunk_end, :]
            output_chunk = output_flat[:, chunk_start:chunk_end]

            # Warmup
            for _ in range(3):
                torch.matmul(x_flat, weight_chunk.t(), out=output_chunk)
            torch.cuda.synchronize()

            # Capture this chunk's computation
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                torch.matmul(x_flat, weight_chunk.t(), out=output_chunk)

            self.compute_graphs.append(graph)

        torch.cuda.synchronize()
        self.graph_captured = True
        print(f"✓ Captured {self.num_chunks} computation graphs successfully")

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        Actual forward implementation (used for both eager and graph mode).

        RowParallelLinear:
        - Input: [batch, seq, in_features] (full)
        - Extract shard: [batch, seq, in_features_per_rank]
        - Weight: [out_features, in_features_per_rank]
        - Local output: [batch, seq, out_features]
        - All-reduce: sum across ranks
        """
        batch, seq, _ = x.shape

        # Use pre-allocated buffers if available
        if self.static_buffers is not None:
            final_output = self.static_buffers['final_output']
        else:
            # Fallback to dynamic allocation (eager mode)
            final_output = torch.empty(batch, seq, self.out_features, device=self.device)

        # Extract this rank's input shard
        start_idx = self.rank * self.in_features_per_rank
        end_idx = (self.rank + 1) * self.in_features_per_rank
        x_shard = x[:, :, start_idx:end_idx]

        # Reshape for chunked processing
        x_flat = x_shard.view(-1, self.in_features_per_rank)
        output_flat = final_output.view(-1, self.out_features)

        if self.sequential:
            # Sequential mode: compute all chunks first, then communicate
            # Step 1: Compute all chunks
            for i in range(self.num_chunks):
                chunk_start = i * self.chunk_size
                chunk_end = (i + 1) * self.chunk_size
                weight_chunk = self.weight[chunk_start:chunk_end, :]
                output_chunk = output_flat[:, chunk_start:chunk_end]
                torch.matmul(x_flat, weight_chunk.t(), out=output_chunk)

            # Step 2: Then do all communication
            for i in range(self.num_chunks):
                chunk_start = i * self.chunk_size
                chunk_end = (i + 1) * self.chunk_size
                output_chunk = output_flat[:, chunk_start:chunk_end].contiguous()
                dist.all_reduce(output_chunk, group=self.process_group)
                output_flat[:, chunk_start:chunk_end].copy_(output_chunk)
        else:
            # Overlap mode: use separate stream for communication
            if self.comm_stream is not None:
                # Eager mode with stream-based overlap
                # Allocate contiguous buffers for communication if needed
                batch_seq = x_flat.shape[0]
                comm_buffers = []
                for i in range(self.num_chunks):
                    chunk_start = i * self.chunk_size
                    chunk_end = (i + 1) * self.chunk_size
                    weight_chunk = self.weight[chunk_start:chunk_end, :]
                    output_chunk = output_flat[:, chunk_start:chunk_end]

                    # Compute chunk i in default stream
                    torch.matmul(x_flat, weight_chunk.t(), out=output_chunk)

                    # Copy to contiguous buffer for communication
                    comm_buf = output_chunk.contiguous()
                    comm_buffers.append((comm_buf, chunk_start, chunk_end))

                    # Record event after compute + copy
                    event = torch.cuda.Event()
                    event.record()

                    # Launch async communication in separate stream
                    with torch.cuda.stream(self.comm_stream):
                        self.comm_stream.wait_event(event)
                        dist.all_reduce(comm_buf, group=self.process_group)

                # Wait for all communications to complete
                torch.cuda.current_stream().wait_stream(self.comm_stream)

                # Copy results back
                for comm_buf, chunk_start, chunk_end in comm_buffers:
                    output_flat[:, chunk_start:chunk_end].copy_(comm_buf)
            else:
                # CUDAGraph mode: async ops without explicit streams
                comm_handles = []

                for i in range(self.num_chunks):
                    chunk_start = i * self.chunk_size
                    chunk_end = (i + 1) * self.chunk_size
                    weight_chunk = self.weight[chunk_start:chunk_end, :]
                    output_chunk = output_flat[:, chunk_start:chunk_end]

                    # Compute chunk i
                    torch.matmul(x_flat, weight_chunk.t(), out=output_chunk)

                    # Launch async communication for chunk i
                    handle = dist.all_reduce(output_chunk, group=self.process_group, async_op=True)
                    comm_handles.append(handle)

                # Wait for all communications to complete
                for handle in comm_handles:
                    handle.wait()

        return final_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional CUDAGraph acceleration.

        Args:
            x: Input tensor [batch, seq, in_features]

        Returns:
            output: Output tensor [batch, seq, out_features]
        """
        # Check if we can use graph
        if self.enable_graph and self.static_input_shape is not None:
            # Verify input shape matches
            if x.shape != self.static_input_shape:
                raise RuntimeError(
                    f"Input shape {x.shape} does not match static shape {self.static_input_shape}. "
                    f"CUDAGraph requires fixed input shapes."
                )

            # Capture graph on first call
            if not self.graph_captured:
                self._capture_graph(x)

            # Execute with graph: computation in graph, communication outside
            batch, seq, _ = x.shape

            # Extract this rank's input shard and copy to static buffer
            start_idx = self.rank * self.in_features_per_rank
            end_idx = (self.rank + 1) * self.in_features_per_rank
            x_shard = x[:, :, start_idx:end_idx]
            self.static_buffers['x_shard'].copy_(x_shard)

            output_flat = self.static_buffers['final_output'].view(-1, self.out_features)

            if self.sequential:
                # Sequential mode: compute all, then communicate all
                for i in range(self.num_chunks):
                    self.compute_graphs[i].replay()

                # Copy to contiguous buffers
                for i in range(self.num_chunks):
                    chunk_start = i * self.chunk_size
                    chunk_end = (i + 1) * self.chunk_size
                    output_chunk_view = output_flat[:, chunk_start:chunk_end]
                    self.chunk_buffers[i].copy_(output_chunk_view)

                # Communicate
                for i in range(self.num_chunks):
                    dist.all_reduce(self.chunk_buffers[i], group=self.process_group)

                # Copy back
                for i in range(self.num_chunks):
                    chunk_start = i * self.chunk_size
                    chunk_end = (i + 1) * self.chunk_size
                    output_flat[:, chunk_start:chunk_end].copy_(self.chunk_buffers[i])
            else:
                # Overlap mode: use separate stream for communication
                for i in range(self.num_chunks):
                    # Replay computation graph for chunk i
                    self.compute_graphs[i].replay()

                    # Copy chunk to contiguous buffer for communication
                    chunk_start = i * self.chunk_size
                    chunk_end = (i + 1) * self.chunk_size
                    output_chunk_view = output_flat[:, chunk_start:chunk_end]
                    self.chunk_buffers[i].copy_(output_chunk_view)

                    # Record event AFTER copy completes (not just after compute)
                    compute_done = torch.cuda.Event()
                    compute_done.record()

                    # Launch async communication in separate stream
                    with torch.cuda.stream(self.comm_stream):
                        # Wait for computation AND copy to finish
                        self.comm_stream.wait_event(compute_done)
                        # Launch communication
                        dist.all_reduce(self.chunk_buffers[i], group=self.process_group)

                # Wait for all communications to complete
                torch.cuda.current_stream().wait_stream(self.comm_stream)

                # Copy results back to final_output
                for i in range(self.num_chunks):
                    chunk_start = i * self.chunk_size
                    chunk_end = (i + 1) * self.chunk_size
                    output_flat[:, chunk_start:chunk_end].copy_(self.chunk_buffers[i])

            return self.static_buffers['final_output']

        # Fallback to eager mode
        return self._forward_impl(x)


def main():
    """Test CUDAGraph implementation."""
    print("=" * 80)
    print("CUDAGRAPH DOUBLE BUFFER OVERLAP TEST")
    print("=" * 80)

    # Test configuration
    batch_size = 32
    seq_len = 1
    in_features = 4096
    out_features = 4096
    num_chunks = 4

    device = "cuda"

    # Create layer with static shape
    layer = CUDAGraphDoubleBufferOverlapRowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        num_chunks=num_chunks,
        static_input_shape=(batch_size, seq_len, in_features),
        enable_graph=True,
        device=device,
    )

    # Create input
    x = torch.randn(batch_size, seq_len, in_features, device=device)

    print(f"\nInput shape: {x.shape}")
    print(f"Chunks: {num_chunks}")

    # First forward (will capture graph)
    print("\nFirst forward (capturing graph)...")
    with torch.no_grad():
        y = layer(x)

    print(f"Output shape: {y.shape}")

    # Subsequent forwards (will replay graph)
    print("\nSubsequent forwards (replaying graph)...")
    with torch.no_grad():
        for i in range(5):
            y = layer(x)

    print("\n✓ All forwards completed successfully")
    print("=" * 80)


if __name__ == "__main__":
    main()

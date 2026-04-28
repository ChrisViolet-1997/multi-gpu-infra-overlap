#!/usr/bin/env python3
"""
Advanced TP Overlap Analysis with Profiling and Visualization

This script extends the basic PoC with:
- Detailed CUDA profiling traces
- Per-chunk latency breakdown
- Communication/Computation ratio analysis
- Optimal chunk size recommendation
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import List, Dict, Tuple
import os
import json


class ProfiledOverlapLinear(nn.Module):
    """
    Enhanced OverlapRowParallelLinear with detailed profiling capabilities.

    Tracks:
        - Per-chunk compute time
        - Per-chunk communication time
        - Overlap efficiency
        - Stream utilization
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_chunks: int = 4,
        process_group=None,
        device: str = "cuda",
        enable_profiling: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_chunks = num_chunks
        self.process_group = process_group
        self.device = device
        self.enable_profiling = enable_profiling

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device, dtype=torch.float32)
        )

        # Streams
        self.compute_stream = torch.cuda.Stream(device=device)
        self.comm_stream = torch.cuda.Stream(device=device)

        # Events for synchronization
        self.compute_events = [torch.cuda.Event() for _ in range(num_chunks)]
        self.comm_events = [torch.cuda.Event() for _ in range(num_chunks)]

        # Profiling events (with timing enabled)
        if enable_profiling:
            self.compute_start_events = [
                torch.cuda.Event(enable_timing=True) for _ in range(num_chunks)
            ]
            self.compute_end_events = [
                torch.cuda.Event(enable_timing=True) for _ in range(num_chunks)
            ]
            self.comm_start_events = [
                torch.cuda.Event(enable_timing=True) for _ in range(num_chunks)
            ]
            self.comm_end_events = [
                torch.cuda.Event(enable_timing=True) for _ in range(num_chunks)
            ]

        self.profile_data = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional profiling."""
        batch_size, seq_len, _ = x.shape
        total_tokens = batch_size * seq_len
        x_flat = x.view(-1, self.in_features)
        chunk_size = (total_tokens + self.num_chunks - 1) // self.num_chunks

        output = torch.empty(
            total_tokens, self.out_features, device=self.device, dtype=x.dtype
        )

        for chunk_idx in range(self.num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_tokens)

            if start_idx >= total_tokens:
                break

            x_chunk = x_flat[start_idx:end_idx]
            output_chunk = output[start_idx:end_idx]

            # Compute stage
            with torch.cuda.stream(self.compute_stream):
                if self.enable_profiling:
                    self.compute_start_events[chunk_idx].record(self.compute_stream)

                torch.matmul(x_chunk, self.weight.t(), out=output_chunk)

                if self.enable_profiling:
                    self.compute_end_events[chunk_idx].record(self.compute_stream)

                self.compute_events[chunk_idx].record(self.compute_stream)

            # Communication stage
            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(self.compute_events[chunk_idx])

                if self.enable_profiling:
                    self.comm_start_events[chunk_idx].record(self.comm_stream)

                dist.all_reduce(
                    output_chunk, op=dist.ReduceOp.SUM, group=self.process_group
                )

                if self.enable_profiling:
                    self.comm_end_events[chunk_idx].record(self.comm_stream)

                self.comm_events[chunk_idx].record(self.comm_stream)

        # Synchronize
        for event in self.comm_events[:self.num_chunks]:
            event.synchronize()

        # Collect profiling data
        if self.enable_profiling:
            self._collect_profile_data()

        return output.view(batch_size, seq_len, self.out_features)

    def _collect_profile_data(self):
        """Extract timing information from profiling events."""
        chunk_profiles = []
        for i in range(self.num_chunks):
            compute_time = self.compute_start_events[i].elapsed_time(
                self.compute_end_events[i]
            )
            comm_time = self.comm_start_events[i].elapsed_time(
                self.comm_end_events[i]
            )
            chunk_profiles.append(
                {"chunk_id": i, "compute_ms": compute_time, "comm_ms": comm_time}
            )
        self.profile_data.append(chunk_profiles)

    def get_average_profile(self) -> Dict:
        """Calculate average profiling statistics."""
        if not self.profile_data:
            return {}

        total_compute = 0.0
        total_comm = 0.0
        num_samples = len(self.profile_data)

        for sample in self.profile_data:
            for chunk in sample:
                total_compute += chunk["compute_ms"]
                total_comm += chunk["comm_ms"]

        avg_compute_per_chunk = total_compute / (num_samples * self.num_chunks)
        avg_comm_per_chunk = total_comm / (num_samples * self.num_chunks)

        return {
            "avg_compute_per_chunk_ms": avg_compute_per_chunk,
            "avg_comm_per_chunk_ms": avg_comm_per_chunk,
            "compute_comm_ratio": avg_compute_per_chunk / avg_comm_per_chunk
            if avg_comm_per_chunk > 0
            else float("inf"),
            "total_compute_ms": total_compute / num_samples,
            "total_comm_ms": total_comm / num_samples,
        }

    def reset_profiling(self):
        """Clear profiling data."""
        self.profile_data = []


def analyze_chunk_sweep(
    rank: int,
    world_size: int,
    chunk_sizes: List[int] = [2, 4, 8, 16, 32],
):
    """
    Sweep over different chunk sizes to find optimal configuration.

    Args:
        rank: GPU rank
        world_size: Number of GPUs
        chunk_sizes: List of chunk counts to test
    """
    from tp_overlap_poc import setup_distributed, cleanup_distributed, benchmark_layer

    setup_distributed(rank, world_size)

    batch_size = 8
    seq_len = 2048
    in_features = 4096
    out_features = 4096
    device = f"cuda:{rank}"

    input_tensor = torch.randn(
        batch_size, seq_len, in_features, device=device, dtype=torch.float32
    )

    results = []

    if rank == 0:
        print("=" * 80)
        print("CHUNK SIZE SWEEP ANALYSIS")
        print("=" * 80)

    for num_chunks in chunk_sizes:
        layer = ProfiledOverlapLinear(
            in_features=in_features,
            out_features=out_features,
            num_chunks=num_chunks,
            process_group=None,
            device=device,
            enable_profiling=True,
        )

        latency = benchmark_layer(layer, input_tensor, num_warmup=5, num_iterations=20)
        profile = layer.get_average_profile()

        results.append(
            {
                "num_chunks": num_chunks,
                "latency_ms": latency,
                "profile": profile,
            }
        )

        if rank == 0:
            print(f"\nChunks: {num_chunks:2d} | Latency: {latency:.3f} ms")
            if profile:
                print(f"  Compute/Chunk: {profile['avg_compute_per_chunk_ms']:.3f} ms")
                print(f"  Comm/Chunk:    {profile['avg_comm_per_chunk_ms']:.3f} ms")
                print(f"  Ratio:         {profile['compute_comm_ratio']:.2f}x")

    if rank == 0:
        # Find optimal chunk size
        best_result = min(results, key=lambda x: x["latency_ms"])
        print("\n" + "=" * 80)
        print(f"OPTIMAL CONFIGURATION: {best_result['num_chunks']} chunks")
        print(f"Best Latency: {best_result['latency_ms']:.3f} ms")
        print("=" * 80)

        # Save results to JSON
        with open("chunk_sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to: chunk_sweep_results.json")

    cleanup_distributed()


def main():
    """Entry point for advanced analysis."""
    if not dist.is_available() or not torch.cuda.is_available():
        raise RuntimeError("Requires PyTorch with CUDA and distributed support")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size < 2:
        raise ValueError("Requires at least 2 GPUs")

    # Run chunk size sweep
    analyze_chunk_sweep(rank, world_size, chunk_sizes=[2, 4, 8, 16, 32])


if __name__ == "__main__":
    main()

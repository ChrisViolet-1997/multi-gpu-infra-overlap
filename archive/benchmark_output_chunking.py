#!/usr/bin/env python3
"""
Benchmark output feature chunking with different chunk numbers.
Measures both accuracy and GPU utilization.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Optional
import os
import sys
import subprocess
import time
import threading


class BaselineRowParallelLinear(nn.Module):
    """Baseline implementation without overlap."""

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

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(x, self.weight.t())
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.process_group)
        return output


class OutputFeatureChunkedLinear(nn.Module):
    """
    Row-Parallel Linear with output feature chunking for high precision overlap.

    Chunks along output feature dimension to maintain numerical precision
    while enabling computation-communication overlap.
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

        # Weight matrix
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device, dtype=torch.float32)
        )

        # Create dedicated CUDA streams
        self.compute_stream = torch.cuda.Stream(device=device)
        self.comm_stream = torch.cuda.Stream(device=device)

        # Pre-allocate CUDA events
        self.compute_events = [torch.cuda.Event() for _ in range(num_chunks)]
        self.comm_events = [torch.cuda.Event() for _ in range(num_chunks)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with output feature chunking.

        Pipeline:
            Chunk 0: [Compute features 0:k] -> [All-Reduce]
            Chunk 1: Wait(Comm_0) -> [Compute features k:2k] -> [All-Reduce]
            ...
        """
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.in_features)
        total_tokens = x_flat.shape[0]

        # Calculate chunk size along output features
        chunk_size = (self.out_features + self.num_chunks - 1) // self.num_chunks

        # Pre-allocate output buffer
        output = torch.empty(
            total_tokens, self.out_features, device=self.device, dtype=x.dtype
        )

        # Pre-allocate contiguous buffers for each chunk
        chunk_buffers = []
        for chunk_idx in range(self.num_chunks):
            start_feat = chunk_idx * chunk_size
            end_feat = min(start_feat + chunk_size, self.out_features)
            if start_feat >= self.out_features:
                break
            chunk_feat_size = end_feat - start_feat
            chunk_buffers.append(
                torch.empty(total_tokens, chunk_feat_size, device=self.device, dtype=x.dtype)
            )

        # === PIPELINED EXECUTION ===
        actual_chunks = len(chunk_buffers)
        for chunk_idx in range(actual_chunks):
            start_feat = chunk_idx * chunk_size
            end_feat = min(start_feat + chunk_size, self.out_features)

            # Extract weight chunk and use pre-allocated buffer
            weight_chunk = self.weight[start_feat:end_feat, :]
            chunk_buffer = chunk_buffers[chunk_idx]

            # --- STAGE 1: COMPUTE CHUNK ---
            with torch.cuda.stream(self.compute_stream):
                # Wait for previous chunk's communication to complete
                if chunk_idx > 0:
                    self.compute_stream.wait_event(self.comm_events[chunk_idx - 1])

                # Compute GEMM into contiguous buffer
                torch.matmul(x_flat, weight_chunk.t(), out=chunk_buffer)

                # Record event: computation complete
                self.compute_events[chunk_idx].record(self.compute_stream)

            # --- STAGE 2: ALL-REDUCE CHUNK (OVERLAPPED) ---
            with torch.cuda.stream(self.comm_stream):
                # Wait for computation to finish
                self.comm_stream.wait_event(self.compute_events[chunk_idx])

                # All-reduce on contiguous buffer
                dist.all_reduce(
                    chunk_buffer, op=dist.ReduceOp.SUM, group=self.process_group
                )

                # Record event: communication complete
                self.comm_events[chunk_idx].record(self.comm_stream)

        # === FINAL SYNCHRONIZATION & ASSEMBLY ===
        for i in range(actual_chunks):
            self.comm_events[i].synchronize()

        # Copy chunks back to output
        for chunk_idx in range(actual_chunks):
            start_feat = chunk_idx * chunk_size
            end_feat = min(start_feat + chunk_size, self.out_features)
            output[:, start_feat:end_feat].copy_(chunk_buffers[chunk_idx])

        return output.view(batch_size, seq_len, self.out_features)


class GPUMonitor:
    """Monitor GPU utilization in background."""

    def __init__(self, rank: int, interval: float = 0.1):
        self.rank = rank
        self.interval = interval
        self.utilizations = []
        self.running = False
        self.thread = None

    def _monitor(self):
        """Background monitoring loop."""
        while self.running:
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                        f"--id={self.rank}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                if result.returncode == 0:
                    util = float(result.stdout.strip())
                    self.utilizations.append(util)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        """Start monitoring."""
        self.running = True
        self.utilizations = []
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring and return average utilization."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

        if len(self.utilizations) > 0:
            # Remove first few samples (warmup)
            valid_samples = self.utilizations[5:] if len(self.utilizations) > 5 else self.utilizations
            return sum(valid_samples) / len(valid_samples) if valid_samples else 0.0
        return 0.0


def setup_distributed(rank: int, world_size: int):
    """Initialize distributed process group."""
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


def benchmark_with_monitoring(
    layer: nn.Module,
    input_tensor: torch.Tensor,
    rank: int,
    num_warmup: int = 10,
    num_iterations: int = 50,
):
    """Benchmark with GPU utilization monitoring."""
    monitor = GPUMonitor(rank, interval=0.05)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = layer(input_tensor)
        torch.cuda.synchronize()

    # Start monitoring
    monitor.start()

    # Timed benchmark
    with torch.no_grad():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iterations):
            _ = layer(input_tensor)
        end_event.record()

        torch.cuda.synchronize()

    # Stop monitoring
    avg_utilization = monitor.stop()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / num_iterations

    return avg_latency_ms, avg_utilization


def run_experiment(rank: int, world_size: int):
    """Run the full experiment."""
    setup_distributed(rank, world_size)

    # Configuration
    batch_size = 8
    seq_len = 2048
    in_features = 4096
    out_features = 4096
    device = f"cuda:{rank}"

    if rank == 0:
        print("=" * 100)
        print("OUTPUT FEATURE CHUNKING: ACCURACY & GPU UTILIZATION BENCHMARK")
        print("=" * 100)
        print(f"Configuration:")
        print(f"  - GPUs: {world_size}")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - Sequence Length: {seq_len}")
        print(f"  - Input Features: {in_features}")
        print(f"  - Output Features: {out_features}")
        print("=" * 100)

    # Create input
    torch.manual_seed(42 + rank)
    input_tensor = torch.randn(
        batch_size, seq_len, in_features, device=device, dtype=torch.float32
    )

    # Create baseline
    torch.manual_seed(42)
    weight = torch.randn(out_features, in_features, device=device, dtype=torch.float32)

    baseline_layer = BaselineRowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        process_group=None,
        device=device,
    )
    baseline_layer.weight.data.copy_(weight)

    # Get baseline output for accuracy comparison
    with torch.no_grad():
        baseline_output = baseline_layer(input_tensor)

    # Benchmark baseline
    if rank == 0:
        print("\n[Baseline] No Overlap")
        print("-" * 100)

    baseline_latency, baseline_util = benchmark_with_monitoring(
        baseline_layer, input_tensor, rank, num_warmup=10, num_iterations=50
    )

    if rank == 0:
        print(f"  Latency: {baseline_latency:.3f} ms | GPU Util: {baseline_util:.1f}%")

    # Test different chunk numbers
    chunk_configs = [2, 4, 8, 16]
    results = []

    for num_chunks in chunk_configs:
        # Check if chunk size is valid
        chunk_size = (out_features + num_chunks - 1) // num_chunks

        if rank == 0:
            print(f"\n[Chunked] {num_chunks} chunks (chunk size: {chunk_size})")
            print("-" * 100)

        # Create chunked layer
        chunked_layer = OutputFeatureChunkedLinear(
            in_features=in_features,
            out_features=out_features,
            num_chunks=num_chunks,
            process_group=None,
            device=device,
        )
        chunked_layer.weight.data.copy_(weight)

        # Test accuracy
        with torch.no_grad():
            chunked_output = chunked_layer(input_tensor)

        max_diff = torch.max(torch.abs(baseline_output - chunked_output)).item()
        mean_diff = torch.mean(torch.abs(baseline_output - chunked_output)).item()

        # Benchmark performance
        chunked_latency, chunked_util = benchmark_with_monitoring(
            chunked_layer, input_tensor, rank, num_warmup=10, num_iterations=50
        )

        speedup = baseline_latency / chunked_latency
        util_improvement = chunked_util - baseline_util

        if rank == 0:
            accuracy_status = "✓ PASS" if max_diff < 1e-5 else "✗ FAIL"
            print(f"  Accuracy:  Max Diff: {max_diff:.2e} | Mean Diff: {mean_diff:.2e} | {accuracy_status}")
            print(f"  Latency:   {chunked_latency:.3f} ms (Speedup: {speedup:.2f}x)")
            print(f"  GPU Util:  {chunked_util:.1f}% (Δ: {util_improvement:+.1f}%)")

        results.append({
            "chunks": num_chunks,
            "chunk_size": chunk_size,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "latency": chunked_latency,
            "speedup": speedup,
            "gpu_util": chunked_util,
            "util_improvement": util_improvement,
        })

    # Summary
    if rank == 0:
        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)
        print(f"{'Chunks':<8} {'Chunk Size':<12} {'Max Diff':<12} {'Latency':<12} {'Speedup':<10} {'GPU Util':<12} {'Util Δ':<10}")
        print("-" * 100)
        print(f"{'Base':<8} {'-':<12} {'-':<12} {baseline_latency:>8.3f} ms {'-':<10} {baseline_util:>8.1f}% {'-':<10}")

        for r in results:
            acc_mark = "✓" if r["max_diff"] < 1e-5 else "✗"
            print(
                f"{r['chunks']:<8} {r['chunk_size']:<12} {r['max_diff']:<9.2e} {acc_mark:<2} "
                f"{r['latency']:>8.3f} ms {r['speedup']:>8.2f}x {r['gpu_util']:>8.1f}% {r['util_improvement']:>+8.1f}%"
            )

        print("=" * 100)

        # Find best configuration
        valid_results = [r for r in results if r["max_diff"] < 1e-5]
        if valid_results:
            best = max(valid_results, key=lambda x: x["speedup"])
            print(f"\n✓ Best Config: {best['chunks']} chunks")
            print(f"  - Speedup: {best['speedup']:.2f}x")
            print(f"  - GPU Utilization: {best['gpu_util']:.1f}% (Δ: {best['util_improvement']:+.1f}%)")
            print(f"  - Accuracy: {best['max_diff']:.2e} (< 1e-5)")
        else:
            print("\n✗ No configuration achieved 1e-5 accuracy")

        print("=" * 100)

    cleanup_distributed()


def main():
    """Entry point."""
    if not dist.is_available():
        raise RuntimeError("PyTorch distributed is not available")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size != 2:
        raise ValueError(
            "This experiment requires exactly 2 GPUs. "
            "Run with: torchrun --nproc_per_node=2 benchmark_output_chunking.py"
        )

    run_experiment(rank, world_size)


if __name__ == "__main__":
    main()

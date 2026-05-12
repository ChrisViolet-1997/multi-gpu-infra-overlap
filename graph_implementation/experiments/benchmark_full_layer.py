#!/usr/bin/env python3
"""
Full Qwen3-8B Decoder Layer Benchmark.

Compares different chunk configurations using CUDAGraphQwen3DecoderLayer
in eager mode (static_input_shape=None) so individual operators still use
their per-chunk CUDAGraphs + stream overlap.

Usage:
    torchrun --nproc_per_node=2 benchmark_full_layer.py
"""

import torch
import torch.distributed as dist
import os
import sys
import time
import statistics

sys.path.insert(0, '/root/autodl-tmp/multi-gpu-infra-overlap/graph_implementation/core')
sys.path.insert(0, '/root/autodl-tmp/multi-gpu-infra-overlap/qwen3_integration')

from qwen3_layer_cudagraph import CUDAGraphQwen3DecoderLayer
from adaptive_chunk_selector import Qwen3ChunkConfig


def setup_distributed():
    """Initialize distributed process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def create_layer(chunk_config, batch_size=16, seq_len=1024):
    """Create a decoder layer with given chunk config."""
    device = "cuda"
    hidden_size = 4096
    input_shape = (batch_size, seq_len, hidden_size)

    layer = CUDAGraphQwen3DecoderLayer(
        hidden_size=hidden_size,
        intermediate_size=12288,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        chunk_config=chunk_config,
        device=device,
        # Disable layer-level graph capture; operators use their own graphs
        static_input_shape=None,
    )
    return layer, input_shape


def benchmark_layer(
    chunk_config,
    batch_size: int = 16,
    seq_len: int = 1024,
    warmup_iters: int = 5,
    measure_iters: int = 20,
    num_repeats: int = 3,
):
    """
    Benchmark a full decoder layer with given chunk config.

    Returns median latency in ms.
    """
    layer, input_shape = create_layer(chunk_config, batch_size, seq_len)

    # Broadcast all weights from rank 0
    for param in layer.parameters():
        dist.broadcast(param.data, src=0)

    # Create input
    x = torch.randn(input_shape, device="cuda")
    dist.broadcast(x, src=0)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = layer(x)
    torch.cuda.synchronize()

    # Measure
    repeat_medians = []
    for _ in range(num_repeats):
        times = []
        for _ in range(measure_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = layer(x)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        repeat_medians.append(statistics.median(times))

    # Cleanup
    del layer
    torch.cuda.empty_cache()

    return statistics.median(repeat_medians)


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--config", type=str, default=None,
                        help="Load optimal config from JSON file")
    args, _ = parser.parse_known_args()

    rank = setup_distributed()
    world_size = dist.get_world_size()
    total_tokens = args.batch_size * args.seq_len

    if rank == 0:
        print("=" * 80)
        print("FULL QWEN3-8B DECODER LAYER BENCHMARK")
        print(f"World size: {world_size}, BS={args.batch_size}, SeqLen={args.seq_len} ({total_tokens} tokens)")
        print("Measurement: 5 warmup + 20 iters, 3 repeats, median")
        print("Layer runs in eager mode; operators use per-chunk CUDAGraphs")
        print("=" * 80)

    # Load optimal config from JSON or use default
    if args.config:
        with open(args.config, "r") as f:
            config_data = json.load(f)
        opt = config_data["optimal_chunks"]
        optimal_chunk_config = Qwen3ChunkConfig(
            q_proj_chunks=opt["q_proj_chunks"],
            k_proj_chunks=opt["k_proj_chunks"],
            v_proj_chunks=opt["v_proj_chunks"],
            o_proj_chunks=opt["o_proj_chunks"],
            gate_proj_chunks=opt["gate_proj_chunks"],
            up_proj_chunks=opt["up_proj_chunks"],
            down_proj_chunks=opt["down_proj_chunks"],
        )
        if rank == 0:
            print(f"Loaded optimal config from: {args.config}")
    else:
        optimal_chunk_config = Qwen3ChunkConfig(
            q_proj_chunks=16, k_proj_chunks=8, v_proj_chunks=8,
            o_proj_chunks=16, gate_proj_chunks=16, up_proj_chunks=16,
            down_proj_chunks=16,
        )

    # Define configurations to compare
    configs = {
        "baseline (chunk=1)": Qwen3ChunkConfig(
            q_proj_chunks=1, k_proj_chunks=1, v_proj_chunks=1,
            o_proj_chunks=1, gate_proj_chunks=1, up_proj_chunks=1,
            down_proj_chunks=1,
        ),
        "static chunk=2": Qwen3ChunkConfig(
            q_proj_chunks=2, k_proj_chunks=2, v_proj_chunks=2,
            o_proj_chunks=2, gate_proj_chunks=2, up_proj_chunks=2,
            down_proj_chunks=2,
        ),
        "static chunk=4": Qwen3ChunkConfig(
            q_proj_chunks=4, k_proj_chunks=4, v_proj_chunks=4,
            o_proj_chunks=4, gate_proj_chunks=4, up_proj_chunks=4,
            down_proj_chunks=4,
        ),
        "static chunk=8": Qwen3ChunkConfig(
            q_proj_chunks=8, k_proj_chunks=8, v_proj_chunks=8,
            o_proj_chunks=8, gate_proj_chunks=8, up_proj_chunks=8,
            down_proj_chunks=8,
        ),
        # Optimal config from grid search
        "optimal (per-op)": optimal_chunk_config,
    }

    results = {}
    baseline_ms = None

    for config_name, chunk_config in configs.items():
        dist.barrier()

        if rank == 0:
            print(f"\nBenchmarking: {config_name}...")

        latency_ms = benchmark_layer(
            chunk_config,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )
        results[config_name] = latency_ms

        if "baseline" in config_name:
            baseline_ms = latency_ms

    # Print results
    if rank == 0:
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"  {'Config':<25s}  {'Latency (ms)':>12s}  {'Speedup':>8s}")
        print(f"  {'─' * 25}  {'─' * 12}  {'─' * 8}")

        for config_name, latency_ms in results.items():
            speedup = ""
            if baseline_ms is not None and baseline_ms > 0:
                sp = (baseline_ms - latency_ms) / baseline_ms * 100
                speedup = f"{sp:+.1f}%"
            print(f"  {config_name:<25s}  {latency_ms:>12.3f}  {speedup:>8s}")

        print("=" * 80)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

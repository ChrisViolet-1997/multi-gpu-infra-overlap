#!/usr/bin/env python3
"""
Per-Operator Grid Search for Optimal Chunk Count.

Tests each operator independently with all valid chunk divisors.
Measures latency with 10 warmup + 30 iterations, 3 repeats, uses median.

Usage:
    torchrun --nproc_per_node=2 grid_search_per_operator.py
"""

import torch
import torch.distributed as dist
import os
import sys
import time
import statistics

sys.path.insert(0, '/root/autodl-tmp/multi-gpu-infra-overlap/graph_implementation/core')
sys.path.insert(0, '/root/autodl-tmp/multi-gpu-infra-overlap/qwen3_integration')

from tp_overlap_cudagraph import CUDAGraphDoubleBufferOverlapRowParallelLinear


def setup_distributed():
    """Initialize distributed process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def benchmark_operator(
    in_features: int,
    out_features: int,
    num_chunks: int,
    batch_size: int = 16,
    seq_len: int = 1024,
    warmup_iters: int = 10,
    measure_iters: int = 30,
    num_repeats: int = 3,
):
    """
    Benchmark a single operator configuration.

    Returns median latency in ms.
    """
    device = "cuda"
    input_shape = (batch_size, seq_len, in_features)

    layer = CUDAGraphDoubleBufferOverlapRowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        num_chunks=num_chunks,
        static_input_shape=input_shape,
        enable_graph=True,
        device=device,
    )

    # Broadcast weights for consistency
    dist.broadcast(layer.weight.data, src=0)

    # Create input
    x = torch.randn(input_shape, device=device)
    dist.broadcast(x, src=0)

    # Warmup (also triggers graph capture)
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = layer(x)
    torch.cuda.synchronize()

    # Measure across repeats
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

    return statistics.median(repeat_medians)


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--output", type=str, default=None,
                        help="Save optimal config to JSON file")
    args, _ = parser.parse_known_args()

    rank = setup_distributed()
    world_size = dist.get_world_size()
    total_tokens = args.batch_size * args.seq_len

    if rank == 0:
        print("=" * 80)
        print("PER-OPERATOR GRID SEARCH: Finding Optimal Chunk Count")
        print(f"World size: {world_size}, BS={args.batch_size}, SeqLen={args.seq_len} ({total_tokens} tokens)")
        print("Measurement: 10 warmup + 30 iters, 3 repeats, median")
        print("=" * 80)

    # Qwen3-8B operator configs: (name, in_features, out_features, chunk_options)
    operators = [
        ("q_proj",    4096,  4096,  [1, 2, 4, 8, 16]),
        ("k_proj",    4096,  1024,  [1, 2, 4, 8, 16]),
        ("v_proj",    4096,  1024,  [1, 2, 4, 8, 16]),
        ("o_proj",    4096,  4096,  [1, 2, 4, 8, 16]),
        ("gate_proj", 4096,  12288, [1, 2, 3, 4, 6, 8, 12, 16]),
        ("up_proj",   4096,  12288, [1, 2, 3, 4, 6, 8, 12, 16]),
        ("down_proj", 12288, 4096,  [1, 2, 4, 8, 16]),
    ]

    optimal_config = {}

    for op_name, in_feat, out_feat, chunk_options in operators:
        if rank == 0:
            print(f"\n{'─' * 70}")
            print(f"  {op_name} ({in_feat} -> {out_feat})")
            print(f"{'─' * 70}")
            print(f"  {'Chunks':>8s}  {'Latency (ms)':>12s}  {'Speedup':>8s}")

        baseline_ms = None
        best_ms = float('inf')
        best_chunks = 1

        for num_chunks in chunk_options:
            # Verify divisibility
            if out_feat % num_chunks != 0:
                continue

            dist.barrier()
            latency_ms = benchmark_operator(
                in_features=in_feat,
                out_features=out_feat,
                num_chunks=num_chunks,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
            )

            if num_chunks == 1:
                baseline_ms = latency_ms

            speedup = ""
            if baseline_ms is not None and baseline_ms > 0:
                sp = (baseline_ms - latency_ms) / baseline_ms * 100
                speedup = f"{sp:+.1f}%"

            if latency_ms < best_ms:
                best_ms = latency_ms
                best_chunks = num_chunks

            if rank == 0:
                print(f"  {num_chunks:>8d}  {latency_ms:>12.3f}  {speedup:>8s}")

        optimal_config[op_name] = best_chunks

        if rank == 0:
            print(f"  >> Best: chunks={best_chunks} ({best_ms:.3f} ms)")

    if rank == 0:
        print("\n" + "=" * 80)
        print("OPTIMAL CONFIGURATION")
        print("=" * 80)
        print("\nQwen3ChunkConfig(")
        for op_name, chunks in optimal_config.items():
            print(f"    {op_name}_chunks={chunks},")
        print(")")
        print("=" * 80)

        # Save to JSON if output path specified
        if args.output:
            config_data = {
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "total_tokens": total_tokens,
                "world_size": world_size,
                "optimal_chunks": {f"{k}_chunks": v for k, v in optimal_config.items()},
            }
            with open(args.output, "w") as f:
                json.dump(config_data, f, indent=2)
            print(f"\nConfig saved to: {args.output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Grid Search for Optimal Chunk Count per Qwen3 Operator

This script exhaustively tests chunk counts [2, 4, 8, 16] for each operator
to find the configuration with minimum latency.
"""

import torch
import torch.distributed as dist
import os
from typing import Dict, Tuple

import sys
sys.path.insert(0, '/root/autodl-tmp/multi-gpu-infra-overlap')
from tp_overlap_double_buffer import DoubleBufferOverlapRowParallelLinear
from tp_overlap_poc import setup_distributed, cleanup_distributed


def benchmark_operator(
    in_features: int,
    out_features: int,
    num_chunks: int,
    batch_size: int,
    seq_len: int,
    device: str,
    num_warmup: int = 5,
    num_iterations: int = 20,
    num_repeats: int = 5,
) -> float:
    """
    Benchmark a single operator configuration.

    Runs the benchmark num_repeats times and returns the average.

    Returns:
        Average latency in milliseconds
    """
    latencies = []

    for repeat in range(num_repeats):
        total_tokens = batch_size * seq_len

        # Create layer
        layer = DoubleBufferOverlapRowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            num_chunks=num_chunks,
            process_group=None,
            device=device,
        )

        # Create input
        input_tensor = torch.randn(
            batch_size, seq_len, in_features,
            device=device, dtype=torch.float32
        )

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = layer(input_tensor)
            torch.cuda.synchronize()

        # Benchmark
        with torch.no_grad():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(num_iterations):
                _ = layer(input_tensor)
            end_event.record()
            torch.cuda.synchronize()

            total_time_ms = start_event.elapsed_time(end_event)
            avg_latency_ms = total_time_ms / num_iterations

        latencies.append(avg_latency_ms)

        # Cleanup
        del layer
        del input_tensor
        torch.cuda.empty_cache()

    # Return average of all repeats
    return sum(latencies) / len(latencies)


def grid_search_operator(
    name: str,
    in_features: int,
    out_features: int,
    batch_size: int,
    seq_len: int,
    device: str,
    chunk_options: list = [1, 2, 4, 8, 16],
) -> Tuple[int, float, Dict[int, float]]:
    """
    Grid search over chunk counts for a single operator.

    Returns:
        (best_chunks, best_latency, all_results)
    """
    results = {}

    for num_chunks in chunk_options:
        latency = benchmark_operator(
            in_features=in_features,
            out_features=out_features,
            num_chunks=num_chunks,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )
        results[num_chunks] = latency

    # Find best
    best_chunks = min(results.keys(), key=lambda k: results[k])
    best_latency = results[best_chunks]

    return best_chunks, best_latency, results


def run_grid_search(
    rank: int,
    world_size: int,
    batch_size: int = 1,
    seq_len: int = 2048,
):
    """Run grid search for all Qwen3 operators."""
    setup_distributed(rank, world_size)

    device = f"cuda:{rank}"

    if rank == 0:
        print("=" * 100)
        print("GRID SEARCH: OPTIMAL CHUNK COUNT PER OPERATOR")
        print("=" * 100)
        print(f"Configuration:")
        print(f"  - World Size: {world_size} GPUs")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - Sequence Length: {seq_len}")
        print(f"  - Chunk Options: [1, 2, 4, 8, 16]")
        print(f"  - Repeats per config: 5 (to reduce variance)")
        print("=" * 100)

    # Define all operators
    operators = {
        "q_proj": (4096, 4096),
        "k_proj": (4096, 1024),
        "v_proj": (4096, 1024),
        "o_proj": (4096, 4096),
        "gate_proj": (4096, 12288),
        "up_proj": (4096, 12288),
        "down_proj": (12288, 4096),
    }

    optimal_config = {}
    all_results = {}

    for i, (op_name, (in_feat, out_feat)) in enumerate(operators.items(), 1):
        if rank == 0:
            print(f"\n[{i}/7] Testing {op_name} ({in_feat} → {out_feat})...")

        best_chunks, best_latency, results = grid_search_operator(
            name=op_name,
            in_features=in_feat,
            out_features=out_feat,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )

        optimal_config[op_name] = best_chunks
        all_results[op_name] = results

        if rank == 0:
            print(f"  Results: {results}")
            print(f"  ✓ Best: {best_chunks} chunks ({best_latency:.3f} ms)")

    if rank == 0:
        print("\n" + "=" * 100)
        print("OPTIMAL CONFIGURATION")
        print("=" * 100)
        print(f"{'Operator':<15} {'Dimensions':<20} {'Optimal Chunks':<15} {'Latency (ms)':<15}")
        print("-" * 100)

        for op_name, (in_feat, out_feat) in operators.items():
            best_chunks = optimal_config[op_name]
            best_latency = all_results[op_name][best_chunks]
            dims = f"{in_feat} → {out_feat}"
            print(f"{op_name:<15} {dims:<20} {best_chunks:<15} {best_latency:<15.3f}")

        print("=" * 100)

        # Print detailed comparison
        print("\nDETAILED RESULTS (Latency in ms)")
        print("=" * 100)
        print(f"{'Operator':<15} {'1 chunk':<12} {'2 chunks':<12} {'4 chunks':<12} {'8 chunks':<12} {'16 chunks':<12} {'Best':<10}")
        print("-" * 100)

        for op_name in operators.keys():
            results = all_results[op_name]
            best_chunks = optimal_config[op_name]

            row = f"{op_name:<15}"
            for chunks in [1, 2, 4, 8, 16]:
                latency = results[chunks]
                marker = " *" if chunks == best_chunks else ""
                row += f" {latency:>10.3f}{marker:<2}"
            row += f" {best_chunks:<10}"
            print(row)

        print("=" * 100)

        # Save configuration to file
        config_str = "# Optimal Chunk Configuration for Qwen3 Operators\n\n"
        config_str += "optimal_chunks = {\n"
        for op_name, chunks in optimal_config.items():
            config_str += f"    '{op_name}': {chunks},\n"
        config_str += "}\n"

        with open("optimal_qwen3_chunks.py", "w") as f:
            f.write(config_str)

        print("\n✓ Optimal configuration saved to: optimal_qwen3_chunks.py")
        print("=" * 100)

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
            "This script requires at least 2 GPUs. "
            "Run with: torchrun --nproc_per_node=2 grid_search_qwen3_chunks.py"
        )

    batch_size = int(os.environ.get("BATCH_SIZE", 1))
    seq_len = int(os.environ.get("SEQ_LEN", 2048))

    run_grid_search(
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
        seq_len=seq_len,
    )


if __name__ == "__main__":
    main()

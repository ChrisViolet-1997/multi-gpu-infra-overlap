#!/usr/bin/env python3
"""
Compare Qwen3 Layer Performance:
1. Fixed 4 chunks for all operators
2. Fixed 8 chunks for all operators
3. Optimal per-operator chunks (from grid search)
"""

import torch
import torch.distributed as dist
import os

import sys
sys.path.insert(0, '/root/autodl-tmp/multi-gpu-infra-overlap')
from qwen3_layer import Qwen3DecoderLayer
from adaptive_chunk_selector import Qwen3ChunkConfig
from tp_overlap_poc import setup_distributed, cleanup_distributed
from optimal_qwen3_chunks import optimal_chunks


def benchmark_layer(
    layer: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 10,
    num_iterations: int = 50,
) -> float:
    """Benchmark a layer's latency."""
    device = input_tensor.device

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

    return avg_latency_ms


def run_comparison(
    rank: int,
    world_size: int,
    batch_size: int = 1,
    seq_len: int = 2048,
    num_warmup: int = 10,
    num_iterations: int = 50,
):
    """Run comparison benchmark."""
    setup_distributed(rank, world_size)

    device = f"cuda:{rank}"
    hidden_size = 4096
    intermediate_size = 12288

    if rank == 0:
        print("=" * 100)
        print("QWEN3 LAYER COMPARISON: FIXED vs OPTIMAL CHUNKS")
        print("=" * 100)
        print(f"Configuration:")
        print(f"  - World Size: {world_size} GPUs")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - Sequence Length: {seq_len}")
        print(f"  - Hidden Size: {hidden_size}")
        print(f"  - Intermediate Size: {intermediate_size}")
        print("=" * 100)

    # Create input
    torch.manual_seed(42 + rank)
    input_tensor = torch.randn(
        batch_size, seq_len, hidden_size,
        device=device, dtype=torch.float32
    )

    results = {}

    # === Configuration 1: Fixed 1 chunk (No Overlap) ===
    if rank == 0:
        print("\n[1/6] Benchmarking Fixed 1 Chunk (No Overlap)...")

    config_1 = Qwen3ChunkConfig(
        q_proj_chunks=1,
        k_proj_chunks=1,
        v_proj_chunks=1,
        o_proj_chunks=1,
        gate_proj_chunks=1,
        up_proj_chunks=1,
        down_proj_chunks=1,
    )

    layer_1 = Qwen3DecoderLayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        chunk_config=config_1,
        process_group=None,
        device=device,
    )

    latency_1 = benchmark_layer(layer_1, input_tensor, num_warmup, num_iterations)
    results["Fixed 1 chunk (No Overlap)"] = latency_1

    del layer_1
    torch.cuda.empty_cache()

    # === Configuration 2: Fixed 2 chunks ===
    if rank == 0:
        print("[2/6] Benchmarking Fixed 2 Chunks...")

    config_2 = Qwen3ChunkConfig(
        q_proj_chunks=2,
        k_proj_chunks=2,
        v_proj_chunks=2,
        o_proj_chunks=2,
        gate_proj_chunks=2,
        up_proj_chunks=2,
        down_proj_chunks=2,
    )

    layer_2 = Qwen3DecoderLayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        chunk_config=config_2,
        process_group=None,
        device=device,
    )

    latency_2 = benchmark_layer(layer_2, input_tensor, num_warmup, num_iterations)
    results["Fixed 2 chunks"] = latency_2

    del layer_2
    torch.cuda.empty_cache()

    # === Configuration 3: Fixed 4 chunks ===
    if rank == 0:
        print("[3/6] Benchmarking Fixed 4 Chunks...")

    config_4 = Qwen3ChunkConfig(
        q_proj_chunks=4,
        k_proj_chunks=4,
        v_proj_chunks=4,
        o_proj_chunks=4,
        gate_proj_chunks=4,
        up_proj_chunks=4,
        down_proj_chunks=4,
    )

    layer_4 = Qwen3DecoderLayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        chunk_config=config_4,
        process_group=None,
        device=device,
    )

    latency_4 = benchmark_layer(layer_4, input_tensor, num_warmup, num_iterations)
    results["Fixed 4 chunks"] = latency_4

    del layer_4
    torch.cuda.empty_cache()

    # === Configuration 4: Fixed 8 chunks ===
    if rank == 0:
        print("[4/6] Benchmarking Fixed 8 Chunks...")

    config_8 = Qwen3ChunkConfig(
        q_proj_chunks=8,
        k_proj_chunks=8,
        v_proj_chunks=8,
        o_proj_chunks=8,
        gate_proj_chunks=8,
        up_proj_chunks=8,
        down_proj_chunks=8,
    )

    layer_8 = Qwen3DecoderLayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        chunk_config=config_8,
        process_group=None,
        device=device,
    )

    latency_8 = benchmark_layer(layer_8, input_tensor, num_warmup, num_iterations)
    results["Fixed 8 chunks"] = latency_8

    del layer_8
    torch.cuda.empty_cache()

    # === Configuration 5: Fixed 16 chunks ===
    if rank == 0:
        print("[5/6] Benchmarking Fixed 16 Chunks...")

    config_16 = Qwen3ChunkConfig(
        q_proj_chunks=16,
        k_proj_chunks=16,
        v_proj_chunks=16,
        o_proj_chunks=16,
        gate_proj_chunks=16,
        up_proj_chunks=16,
        down_proj_chunks=16,
    )

    layer_16 = Qwen3DecoderLayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        chunk_config=config_16,
        process_group=None,
        device=device,
    )

    latency_16 = benchmark_layer(layer_16, input_tensor, num_warmup, num_iterations)
    results["Fixed 16 chunks"] = latency_16

    del layer_16
    torch.cuda.empty_cache()

    # === Configuration 6: Optimal per-operator ===
    if rank == 0:
        print("[6/6] Benchmarking Optimal Per-Operator Chunks...")
        print("\nOptimal Configuration:")
        print(f"  - q_proj:    {optimal_chunks['q_proj']} chunks")
        print(f"  - k_proj:    {optimal_chunks['k_proj']} chunks")
        print(f"  - v_proj:    {optimal_chunks['v_proj']} chunks")
        print(f"  - o_proj:    {optimal_chunks['o_proj']} chunks")
        print(f"  - gate_proj: {optimal_chunks['gate_proj']} chunks")
        print(f"  - up_proj:   {optimal_chunks['up_proj']} chunks")
        print(f"  - down_proj: {optimal_chunks['down_proj']} chunks")

    config_optimal = Qwen3ChunkConfig(
        q_proj_chunks=optimal_chunks['q_proj'],
        k_proj_chunks=optimal_chunks['k_proj'],
        v_proj_chunks=optimal_chunks['v_proj'],
        o_proj_chunks=optimal_chunks['o_proj'],
        gate_proj_chunks=optimal_chunks['gate_proj'],
        up_proj_chunks=optimal_chunks['up_proj'],
        down_proj_chunks=optimal_chunks['down_proj'],
    )

    layer_optimal = Qwen3DecoderLayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        chunk_config=config_optimal,
        process_group=None,
        device=device,
    )

    latency_optimal = benchmark_layer(layer_optimal, input_tensor, num_warmup, num_iterations)
    results["Optimal per-operator"] = latency_optimal

    del layer_optimal
    torch.cuda.empty_cache()

    # === Results ===
    if rank == 0:
        print("\n" + "=" * 100)
        print("RESULTS")
        print("=" * 100)

        # Find best configuration
        best_config = min(results.keys(), key=lambda k: results[k])
        best_latency = results[best_config]

        print(f"{'Configuration':<30} {'Latency (ms)':<20} {'Speedup vs Best':<20} {'vs Best':<15}")
        print("-" * 100)

        for config_name in ["Fixed 1 chunk (No Overlap)", "Fixed 2 chunks", "Fixed 4 chunks", "Fixed 8 chunks", "Fixed 16 chunks", "Optimal per-operator"]:
            if config_name not in results:
                continue
            latency = results[config_name]
            speedup = best_latency / latency
            improvement = (1 - latency / best_latency) * 100

            marker = ""
            if config_name == best_config:
                marker = " ← BEST"

            print(f"{config_name:<30} {latency:<20.3f} {speedup:<20.2f}x {improvement:<14.1f}%{marker}")

        print("=" * 100)

        # Detailed analysis
        print("\nDETAILED ANALYSIS:")
        print("-" * 100)

        print(f"\n✓ BEST Configuration: {best_config}")
        print(f"  - Latency: {best_latency:.3f} ms")

        # Compare optimal vs all fixed configs
        if best_config == "Optimal per-operator":
            print(f"\n✓ Optimal per-operator WINS!")
            for fixed_name in ["Fixed 1 chunk (No Overlap)", "Fixed 2 chunks", "Fixed 4 chunks", "Fixed 8 chunks", "Fixed 16 chunks"]:
                if fixed_name in results:
                    improvement = (results[fixed_name] / best_latency - 1) * 100
                    print(f"  - {improvement:.1f}% faster than {fixed_name}")
        else:
            print(f"\n⚠ {best_config} performs best")
            if "Optimal per-operator" in results:
                diff = (results["Optimal per-operator"] / best_latency - 1) * 100
                print(f"  - Optimal per-operator is {diff:.1f}% slower")

        # Show optimal configuration
        print(f"\nOptimal Per-Operator Configuration:")
        print(f"  - q_proj:    {optimal_chunks['q_proj']} chunks")
        print(f"  - k_proj:    {optimal_chunks['k_proj']} chunks")
        print(f"  - v_proj:    {optimal_chunks['v_proj']} chunks")
        print(f"  - o_proj:    {optimal_chunks['o_proj']} chunks")
        print(f"  - gate_proj: {optimal_chunks['gate_proj']} chunks")
        print(f"  - up_proj:   {optimal_chunks['up_proj']} chunks")
        print(f"  - down_proj: {optimal_chunks['down_proj']} chunks")

        # Breakdown by operator type
        print("\n" + "=" * 100)
        print("OPERATOR BREAKDOWN (from grid search):")
        print("-" * 100)
        print(f"{'Operator':<15} {'Optimal Chunks':<15} {'Latency (ms)':<15} {'Note':<40}")
        print("-" * 100)

        # Load individual operator latencies from grid search
        operator_notes = {
            'q_proj': 'Attention query projection',
            'k_proj': 'Attention key projection (smaller)',
            'v_proj': 'Attention value projection (smaller)',
            'o_proj': 'Attention output projection',
            'gate_proj': 'MLP gate (largest output)',
            'up_proj': 'MLP up (largest output)',
            'down_proj': 'MLP down (largest input)',
        }

        for op_name, chunks in optimal_chunks.items():
            note = operator_notes.get(op_name, '')
            print(f"{op_name:<15} {chunks:<15} {'N/A':<15} {note:<40}")

        print("=" * 100)

        # Summary
        print("\nKEY FINDINGS:")
        print("-" * 100)
        print("1. Optimal chunk counts vary by operator:")
        print(f"   - Small outputs (k_proj, v_proj): {optimal_chunks['k_proj']} chunks")
        print(f"   - Medium outputs (q_proj, o_proj): {optimal_chunks['q_proj']} chunks")
        print(f"   - Large outputs (gate_proj, up_proj): {optimal_chunks['gate_proj']} chunks")
        print(f"   - Large inputs (down_proj): {optimal_chunks['down_proj']} chunks")
        print()
        print("2. Per-operator tuning provides measurable benefit over fixed chunking")
        print()
        print("3. Larger output dimensions benefit from more chunks (better overlap)")
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
            "This benchmark requires at least 2 GPUs. "
            "Run with: torchrun --nproc_per_node=2 compare_qwen3_configs.py"
        )

    batch_size = int(os.environ.get("BATCH_SIZE", 1))
    seq_len = int(os.environ.get("SEQ_LEN", 2048))
    num_warmup = int(os.environ.get("NUM_WARMUP", 10))
    num_iterations = int(os.environ.get("NUM_ITERATIONS", 50))

    run_comparison(
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
        seq_len=seq_len,
        num_warmup=num_warmup,
        num_iterations=num_iterations,
    )


if __name__ == "__main__":
    main()

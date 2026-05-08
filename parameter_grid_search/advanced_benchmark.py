#!/usr/bin/env python3
"""
Advanced Benchmark with Overlap Ratio Metrics

Measures:
- T_comp: Pure computation time
- T_comm: Pure communication time
- T_total: Total execution time with overlap
- Overlap_Ratio = (T_comp + T_comm - T_total) / min(T_comp, T_comm)
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Dict, Tuple
import os


def measure_separated_compute_comm(
    layer: nn.Module,
    input_tensor: torch.Tensor,
    num_iterations: int = 50,
) -> Tuple[float, float]:
    """
    Measure pure computation and communication time separately (no overlap).

    Returns:
        (avg_comp_time_ms, avg_comm_time_ms)
    """
    batch_size, seq_len, _ = input_tensor.shape
    total_tokens = batch_size * seq_len
    x_flat = input_tensor.view(-1, layer.in_features)
    chunk_size = (total_tokens + layer.num_chunks - 1) // layer.num_chunks

    # === Measure pure computation time ===
    comp_times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            comp_start = torch.cuda.Event(enable_timing=True)
            comp_end = torch.cuda.Event(enable_timing=True)

            comp_start.record()
            # Do all GEMM computations sequentially
            for chunk_idx in range(layer.num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_tokens)
                if start_idx >= total_tokens:
                    break

                x_chunk = x_flat[start_idx:end_idx]
                _ = torch.matmul(x_chunk, layer.weight.t())

            comp_end.record()
            torch.cuda.synchronize()
            comp_times.append(comp_start.elapsed_time(comp_end))

    avg_comp_time_ms = sum(comp_times) / len(comp_times)

    # === Measure pure communication time ===
    # First do one forward pass to get the output
    with torch.no_grad():
        output = layer(input_tensor)

    # Now measure just all-reduce time
    comm_times = []
    output_flat = output.view(-1, layer.out_features)

    with torch.no_grad():
        for _ in range(num_iterations):
            comm_start = torch.cuda.Event(enable_timing=True)
            comm_end = torch.cuda.Event(enable_timing=True)

            comm_start.record()
            # Do all all-reduce operations sequentially
            for chunk_idx in range(layer.num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_tokens)
                if start_idx >= total_tokens:
                    break

                chunk = output_flat[start_idx:end_idx].clone()
                dist.all_reduce(chunk, op=dist.ReduceOp.SUM, group=layer.process_group)

            comm_end.record()
            torch.cuda.synchronize()
            comm_times.append(comm_start.elapsed_time(comm_end))

    avg_comm_time_ms = sum(comm_times) / len(comm_times)

    return avg_comp_time_ms, avg_comm_time_ms


def measure_overlap_metrics(
    layer: nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 10,
    num_iterations: int = 50,
) -> Dict[str, float]:
    """
    Measure detailed overlap metrics using CUDA events.

    Returns:
        Dict with keys:
        - latency_ms: Average total latency (with overlap)
        - comp_time_ms: Pure computation time (no overlap)
        - comm_time_ms: Pure communication time (no overlap)
        - speedup: (T_comp + T_comm) / T_total
        - overlap_ratio: (T_comp + T_comm - T_total) / min(T_comp, T_comm)
        - overlap_percentage: Percentage of time saved by overlap
    """
    device = input_tensor.device

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = layer(input_tensor)
        torch.cuda.synchronize()

    # === Measure T_total (with overlap) ===
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

    # === Measure T_comp and T_comm separately (no overlap) ===
    avg_comp_time_ms, avg_comm_time_ms = measure_separated_compute_comm(
        layer, input_tensor, num_iterations=num_iterations
    )

    # === Calculate metrics ===
    # Speedup: how much faster with overlap vs sequential
    sequential_time = avg_comp_time_ms + avg_comm_time_ms
    speedup = sequential_time / avg_latency_ms

    # Time saved by overlap
    time_saved = sequential_time - avg_latency_ms

    # Overlap ratio: normalized by the shorter operation
    min_time = min(avg_comp_time_ms, avg_comm_time_ms)
    if min_time > 0:
        overlap_ratio = time_saved / min_time
    else:
        overlap_ratio = 0.0

    # Overlap percentage: how much of the total sequential time was saved
    if sequential_time > 0:
        overlap_percentage = (time_saved / sequential_time) * 100
    else:
        overlap_percentage = 0.0

    return {
        "latency_ms": avg_latency_ms,
        "comp_time_ms": avg_comp_time_ms,
        "comm_time_ms": avg_comm_time_ms,
        "speedup": speedup,
        "overlap_ratio": overlap_ratio,
        "overlap_percentage": overlap_percentage,
    }


def grid_search_parameters(
    layer_class,
    input_tensor: torch.Tensor,
    out_features_list: list,
    num_chunks_list: list,
    in_features: int,
    process_group=None,
    device: str = "cuda",
    num_warmup: int = 5,
    num_iterations: int = 20,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Grid search over output_features and num_chunks.

    Args:
        layer_class: Layer class to instantiate
        input_tensor: Input tensor for benchmarking
        out_features_list: List of output feature dimensions to test
        num_chunks_list: List of chunk counts to test
        in_features: Input feature dimension
        process_group: Distributed process group
        device: Device to run on
        num_warmup: Warmup iterations
        num_iterations: Benchmark iterations

    Returns:
        Dict mapping (out_features, num_chunks) -> metrics dict
    """
    results = {}

    total_configs = len(out_features_list) * len(num_chunks_list)
    config_idx = 0

    for out_features in out_features_list:
        for num_chunks in num_chunks_list:
            config_idx += 1

            # Check if chunk size is reasonable
            batch_size, seq_len, _ = input_tensor.shape
            total_tokens = batch_size * seq_len
            chunk_size = total_tokens // num_chunks

            # Skip if chunk is too small (< 16 tokens)
            if chunk_size < 16:
                continue

            # Create layer
            layer = layer_class(
                in_features=in_features,
                out_features=out_features,
                num_chunks=num_chunks,
                process_group=process_group,
                device=device,
            )

            # Measure metrics
            metrics = measure_overlap_metrics(
                layer, input_tensor, num_warmup=num_warmup, num_iterations=num_iterations
            )

            # Add configuration info
            metrics["out_features"] = out_features
            metrics["num_chunks"] = num_chunks
            metrics["chunk_size"] = chunk_size
            metrics["out_per_chunk"] = out_features / num_chunks

            results[(out_features, num_chunks)] = metrics

            # Clean up
            del layer
            torch.cuda.empty_cache()

    return results


def grid_search_in_out_features(
    layer_class,
    batch_size: int,
    seq_len: int,
    in_features_list: list,
    out_features_list: list,
    num_chunks: int = 4,
    process_group=None,
    device: str = "cuda",
    num_warmup: int = 5,
    num_iterations: int = 20,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Grid search over in_features and out_features with fixed num_chunks.

    Args:
        layer_class: Layer class to instantiate
        batch_size: Batch size
        seq_len: Sequence length
        in_features_list: List of input feature dimensions to test
        out_features_list: List of output feature dimensions to test
        num_chunks: Fixed number of chunks
        process_group: Distributed process group
        device: Device to run on
        num_warmup: Warmup iterations
        num_iterations: Benchmark iterations

    Returns:
        Dict mapping (in_features, out_features) -> metrics dict
    """
    results = {}

    total_configs = len(in_features_list) * len(out_features_list)
    config_idx = 0

    for in_features in in_features_list:
        for out_features in out_features_list:
            config_idx += 1

            # Create input tensor
            input_tensor = torch.randn(
                batch_size, seq_len, in_features,
                device=device, dtype=torch.float32
            )

            # Create layer
            layer = layer_class(
                in_features=in_features,
                out_features=out_features,
                num_chunks=num_chunks,
                process_group=process_group,
                device=device,
            )

            # Measure metrics
            metrics = measure_overlap_metrics(
                layer, input_tensor, num_warmup=num_warmup, num_iterations=num_iterations
            )

            # Add configuration info
            metrics["in_features"] = in_features
            metrics["out_features"] = out_features
            metrics["num_chunks"] = num_chunks
            metrics["comp_comm_ratio"] = metrics["comp_time_ms"] / metrics["comm_time_ms"] if metrics["comm_time_ms"] > 0 else 0

            results[(in_features, out_features)] = metrics

            # Clean up
            del layer
            del input_tensor
            torch.cuda.empty_cache()

    return results


def generate_heatmap_data(
    results: Dict[Tuple[int, int], Dict[str, float]],
    metric_key: str = "overlap_ratio"
) -> Tuple[list, list, list]:
    """
    Generate data for heatmap visualization.

    Args:
        results: Grid search results
        metric_key: Which metric to visualize

    Returns:
        (out_features_list, num_chunks_list, values_2d)
    """
    import numpy as np

    # Extract unique values
    out_features_set = sorted(set(k[0] for k in results.keys()))
    num_chunks_set = sorted(set(k[1] for k in results.keys()))

    # Create 2D array
    values = np.zeros((len(num_chunks_set), len(out_features_set)))

    for i, num_chunks in enumerate(num_chunks_set):
        for j, out_features in enumerate(out_features_set):
            key = (out_features, num_chunks)
            if key in results:
                values[i, j] = results[key][metric_key]
            else:
                values[i, j] = np.nan

    return out_features_set, num_chunks_set, values


def print_in_out_results_table(results: Dict[Tuple[int, int], Dict[str, float]], rank: int = 0):
    """Print results for in_features vs out_features grid search."""
    if rank != 0:
        return

    print("\n" + "=" * 140)
    print("IN_FEATURES vs OUT_FEATURES GRID SEARCH RESULTS")
    print("=" * 140)
    print(f"{'In Features':<15} {'Out Features':<15} {'Comp/Comm':<12} {'Latency(ms)':<15} "
          f"{'Comp(ms)':<12} {'Comm(ms)':<12} {'Speedup':<10} {'Overlap%':<12} {'Overlap Ratio':<15}")
    print("-" * 140)

    # Sort by in_features, then out_features
    sorted_keys = sorted(results.keys())

    for key in sorted_keys:
        metrics = results[key]
        in_features = metrics["in_features"]
        out_features = metrics["out_features"]
        comp_comm_ratio = metrics["comp_comm_ratio"]
        latency = metrics["latency_ms"]
        comp = metrics["comp_time_ms"]
        comm = metrics["comm_time_ms"]
        speedup = metrics.get("speedup", 0.0)
        overlap_pct = metrics["overlap_percentage"]
        overlap_ratio = metrics["overlap_ratio"]

        print(f"{in_features:<15} {out_features:<15} {comp_comm_ratio:<12.2f} {latency:<15.3f} "
              f"{comp:<12.3f} {comm:<12.3f} {speedup:<10.2f} {overlap_pct:<12.1f} {overlap_ratio:<15.3f}")

    print("=" * 140)


def print_results_table(results: Dict[Tuple[int, int], Dict[str, float]], rank: int = 0):
    """Print results in a formatted table."""
    if rank != 0:
        return

    print("\n" + "=" * 130)
    print("GRID SEARCH RESULTS")
    print("=" * 130)
    print(f"{'Out Features':<15} {'Chunks':<10} {'Out/Chunk':<12} {'Latency(ms)':<15} "
          f"{'Comp(ms)':<12} {'Comm(ms)':<12} {'Speedup':<10} {'Overlap%':<12} {'Overlap Ratio':<15}")
    print("-" * 130)

    # Sort by out_features, then num_chunks
    sorted_keys = sorted(results.keys())

    for key in sorted_keys:
        metrics = results[key]
        out_features = metrics["out_features"]
        num_chunks = metrics["num_chunks"]
        out_per_chunk = metrics["out_per_chunk"]
        latency = metrics["latency_ms"]
        comp = metrics["comp_time_ms"]
        comm = metrics["comm_time_ms"]
        speedup = metrics.get("speedup", 0.0)
        overlap_pct = metrics["overlap_percentage"]
        overlap_ratio = metrics["overlap_ratio"]

        print(f"{out_features:<15} {num_chunks:<10} {out_per_chunk:<12.1f} {latency:<15.3f} "
              f"{comp:<12.3f} {comm:<12.3f} {speedup:<10.2f} {overlap_pct:<12.1f} {overlap_ratio:<15.3f}")

    print("=" * 130)




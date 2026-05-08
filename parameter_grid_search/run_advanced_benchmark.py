#!/usr/bin/env python3
"""
Advanced Benchmark Runner with Grid Search

Usage:
    # Single configuration
    torchrun --nproc_per_node=2 run_advanced_benchmark.py

    # Grid search
    torchrun --nproc_per_node=2 run_advanced_benchmark.py --grid_search
"""

import torch
import torch.distributed as dist
import os
import argparse
import json
from tp_overlap_double_buffer import (
    DoubleBufferOverlapRowParallelLinear,
    setup_distributed,
    cleanup_distributed,
)
from advanced_benchmark import (
    measure_overlap_metrics,
    grid_search_parameters,
    grid_search_in_out_features,
    print_results_table,
    print_in_out_results_table,
    generate_heatmap_data,
)


def run_single_benchmark(args, rank, world_size):
    """Run benchmark for a single configuration."""
    setup_distributed(rank, world_size)
    device = f"cuda:{rank}"

    if rank == 0:
        print("=" * 80)
        print("ADVANCED BENCHMARK - Single Configuration")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  - World Size: {world_size} GPUs")
        print(f"  - Batch Size: {args.batch_size}")
        print(f"  - Sequence Length: {args.seq_len}")
        print(f"  - Input Features: {args.in_features}")
        print(f"  - Output Features: {args.out_features}")
        print(f"  - Num Chunks: {args.num_chunks}")
        print("=" * 80)

    # Create input
    torch.manual_seed(42 + rank)
    input_tensor = torch.randn(
        args.batch_size, args.seq_len, args.in_features,
        device=device, dtype=torch.float32
    )

    # Create layer
    layer = DoubleBufferOverlapRowParallelLinear(
        in_features=args.in_features,
        out_features=args.out_features,
        num_chunks=args.num_chunks,
        process_group=None,
        device=device,
    )

    # Measure metrics
    metrics = measure_overlap_metrics(
        layer, input_tensor, num_warmup=args.num_warmup, num_iterations=args.num_iterations
    )

    if rank == 0:
        print("\nResults:")
        print(f"  Total Latency (with overlap): {metrics['latency_ms']:.3f} ms")
        print(f"  Computation Time (no overlap): {metrics['comp_time_ms']:.3f} ms")
        print(f"  Communication Time (no overlap): {metrics['comm_time_ms']:.3f} ms")
        print(f"  Sequential Time (Comp + Comm):  {metrics['comp_time_ms'] + metrics['comm_time_ms']:.3f} ms")
        print(f"\n  Speedup:             {metrics['speedup']:.2f}x")
        print(f"  Overlap Percentage:  {metrics['overlap_percentage']:.1f}%")
        print(f"  Overlap Ratio:       {metrics['overlap_ratio']:.3f}")
        print("\nInterpretation:")
        if metrics['speedup'] >= 1.8:
            print("  ✓ Excellent overlap! Near-perfect hiding of communication.")
        elif metrics['speedup'] >= 1.5:
            print("  ✓ Good overlap. Most communication is hidden.")
        elif metrics['speedup'] >= 1.2:
            print("  ✓ Moderate overlap. Partial communication hiding.")
        else:
            print("  ⚠ Limited overlap. Communication dominates or chunks are too small.")
        print("=" * 80)

    cleanup_distributed()


def run_grid_search(args, rank, world_size):
    """Run grid search over parameters."""
    setup_distributed(rank, world_size)
    device = f"cuda:{rank}"

    if rank == 0:
        print("=" * 80)
        print("ADVANCED BENCHMARK - Grid Search")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  - World Size: {world_size} GPUs")
        print(f"  - Batch Size: {args.batch_size}")
        print(f"  - Sequence Length: {args.seq_len}")
        print(f"  - Input Features: {args.in_features}")
        print(f"  - Output Features Range: {args.out_features_min} to {args.out_features_max}")
        print(f"  - Num Chunks Range: {args.num_chunks_min} to {args.num_chunks_max}")
        print("=" * 80)

    # Create input
    torch.manual_seed(42 + rank)
    input_tensor = torch.randn(
        args.batch_size, args.seq_len, args.in_features,
        device=device, dtype=torch.float32
    )

    # Define parameter ranges
    out_features_list = list(range(
        args.out_features_min,
        args.out_features_max + args.out_features_step,
        args.out_features_step
    ))
    num_chunks_list = list(range(args.num_chunks_min, args.num_chunks_max + 1))

    if rank == 0:
        print(f"\nTesting {len(out_features_list)} x {len(num_chunks_list)} = "
              f"{len(out_features_list) * len(num_chunks_list)} configurations...")

    # Run grid search
    results = grid_search_parameters(
        layer_class=DoubleBufferOverlapRowParallelLinear,
        input_tensor=input_tensor,
        out_features_list=out_features_list,
        num_chunks_list=num_chunks_list,
        in_features=args.in_features,
        process_group=None,
        device=device,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
    )

    # Print results
    print_results_table(results, rank)

    # Save results to JSON
    if rank == 0:
        output_file = "grid_search_results.json"
        # Convert tuple keys to strings for JSON
        json_results = {f"{k[0]}_{k[1]}": v for k, v in results.items()}
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {output_file}")

        # Generate heatmap data
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            out_features_vals, num_chunks_vals, overlap_ratios = generate_heatmap_data(
                results, metric_key="overlap_ratio"
            )

            plt.figure(figsize=(12, 8))
            im = plt.imshow(overlap_ratios, aspect='auto', cmap='RdYlGn', origin='lower')
            plt.colorbar(im, label='Overlap Ratio')
            plt.xlabel('Output Features')
            plt.ylabel('Num Chunks')
            plt.title('Overlap Ratio Heatmap\n(Higher is Better)')

            # Set ticks
            plt.xticks(range(len(out_features_vals)), out_features_vals, rotation=45)
            plt.yticks(range(len(num_chunks_vals)), num_chunks_vals)

            # Add text annotations
            for i in range(len(num_chunks_vals)):
                for j in range(len(out_features_vals)):
                    if not np.isnan(overlap_ratios[i, j]):
                        text = plt.text(j, i, f'{overlap_ratios[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontsize=8)

            plt.tight_layout()
            plt.savefig('overlap_heatmap.png', dpi=150)
            print(f"Heatmap saved to overlap_heatmap.png")

        except ImportError:
            print("\nMatplotlib not available. Skipping heatmap generation.")

    cleanup_distributed()


def run_grid_search_in_out(args, rank, world_size):
    """Run grid search over in_features and out_features."""
    setup_distributed(rank, world_size)
    device = f"cuda:{rank}"

    if rank == 0:
        print("=" * 80)
        print("ADVANCED BENCHMARK - Grid Search (In x Out Features)")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  - World Size: {world_size} GPUs")
        print(f"  - Batch Size: {args.batch_size}")
        print(f"  - Sequence Length: {args.seq_len}")
        print(f"  - In Features Range: {args.in_features_min} to {args.in_features_max}")
        print(f"  - Out Features Range: {args.out_features_min} to {args.out_features_max}")
        print(f"  - Num Chunks (fixed): {args.num_chunks}")
        print("=" * 80)

    # Define parameter ranges
    in_features_list = list(range(
        args.in_features_min,
        args.in_features_max + args.in_features_step,
        args.in_features_step
    ))
    out_features_list = list(range(
        args.out_features_min,
        args.out_features_max + args.out_features_step,
        args.out_features_step
    ))

    if rank == 0:
        print(f"\nTesting {len(in_features_list)} x {len(out_features_list)} = "
              f"{len(in_features_list) * len(out_features_list)} configurations...")

    # Run grid search
    results = grid_search_in_out_features(
        layer_class=DoubleBufferOverlapRowParallelLinear,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        in_features_list=in_features_list,
        out_features_list=out_features_list,
        num_chunks=args.num_chunks,
        process_group=None,
        device=device,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
    )

    # Print results
    print_in_out_results_table(results, rank)

    # Save results to JSON
    if rank == 0:
        output_file = "grid_search_in_out_results.json"
        # Convert tuple keys to strings for JSON
        json_results = {f"{k[0]}_{k[1]}": v for k, v in results.items()}
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {output_file}")

        # Generate heatmap for speedup
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Extract data
            in_features_vals = sorted(set(k[0] for k in results.keys()))
            out_features_vals = sorted(set(k[1] for k in results.keys()))

            # Create 2D arrays for different metrics
            speedup_data = np.zeros((len(out_features_vals), len(in_features_vals)))
            comp_comm_ratio_data = np.zeros((len(out_features_vals), len(in_features_vals)))

            for i, out_features in enumerate(out_features_vals):
                for j, in_features in enumerate(in_features_vals):
                    key = (in_features, out_features)
                    if key in results:
                        speedup_data[i, j] = results[key]["speedup"]
                        comp_comm_ratio_data[i, j] = results[key]["comp_comm_ratio"]
                    else:
                        speedup_data[i, j] = np.nan
                        comp_comm_ratio_data[i, j] = np.nan

            # Plot speedup heatmap
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

            # Speedup heatmap
            im1 = ax1.imshow(speedup_data, aspect='auto', cmap='RdYlGn', origin='lower', vmin=1.0, vmax=2.0)
            plt.colorbar(im1, ax=ax1, label='Speedup')
            ax1.set_xlabel('Input Features')
            ax1.set_ylabel('Output Features')
            ax1.set_title('Speedup Heatmap\n(Higher is Better, Max=2.0x)')
            ax1.set_xticks(range(len(in_features_vals)))
            ax1.set_xticklabels(in_features_vals, rotation=45)
            ax1.set_yticks(range(len(out_features_vals)))
            ax1.set_yticklabels(out_features_vals)

            # Add text annotations for speedup
            for i in range(len(out_features_vals)):
                for j in range(len(in_features_vals)):
                    if not np.isnan(speedup_data[i, j]):
                        text = ax1.text(j, i, f'{speedup_data[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontsize=8)

            # Comp/Comm ratio heatmap
            im2 = ax2.imshow(comp_comm_ratio_data, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar(im2, ax=ax2, label='Comp/Comm Ratio')
            ax2.set_xlabel('Input Features')
            ax2.set_ylabel('Output Features')
            ax2.set_title('Computation/Communication Ratio\n(>1: Comp-bound, <1: Comm-bound)')
            ax2.set_xticks(range(len(in_features_vals)))
            ax2.set_xticklabels(in_features_vals, rotation=45)
            ax2.set_yticks(range(len(out_features_vals)))
            ax2.set_yticklabels(out_features_vals)

            # Add text annotations for comp/comm ratio
            for i in range(len(out_features_vals)):
                for j in range(len(in_features_vals)):
                    if not np.isnan(comp_comm_ratio_data[i, j]):
                        text = ax2.text(j, i, f'{comp_comm_ratio_data[i, j]:.2f}',
                                       ha="center", va="center", color="white", fontsize=8)

            plt.tight_layout()
            plt.savefig('in_out_features_heatmap.png', dpi=150)
            print(f"Heatmap saved to in_out_features_heatmap.png")

        except ImportError:
            print("\nMatplotlib not available. Skipping heatmap generation.")

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Advanced Benchmark with Overlap Metrics")

    # Basic configuration
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--in_features", type=int, default=4096)

    # Single run parameters
    parser.add_argument("--out_features", type=int, default=12288)
    parser.add_argument("--num_chunks", type=int, default=4)

    # Grid search mode
    parser.add_argument("--grid_search", action="store_true", help="Run grid search over out_features and num_chunks")
    parser.add_argument("--grid_search_in_out", action="store_true", help="Run grid search over in_features and out_features")

    # Grid search parameters (for out_features x num_chunks)
    parser.add_argument("--out_features_min", type=int, default=4096)
    parser.add_argument("--out_features_max", type=int, default=16384)
    parser.add_argument("--out_features_step", type=int, default=2048)
    parser.add_argument("--num_chunks_min", type=int, default=2)
    parser.add_argument("--num_chunks_max", type=int, default=8)

    # Grid search parameters (for in_features x out_features)
    parser.add_argument("--in_features_min", type=int, default=8192)
    parser.add_argument("--in_features_max", type=int, default=40960)
    parser.add_argument("--in_features_step", type=int, default=8192)

    # Benchmark parameters
    parser.add_argument("--num_warmup", type=int, default=10)
    parser.add_argument("--num_iterations", type=int, default=50)

    args = parser.parse_args()

    # Get distributed info
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size < 2:
        raise ValueError("This benchmark requires at least 2 GPUs")

    # Run appropriate benchmark
    if args.grid_search_in_out:
        run_grid_search_in_out(args, rank, world_size)
    elif args.grid_search:
        run_grid_search(args, rank, world_size)
    else:
        run_single_benchmark(args, rank, world_size)


if __name__ == "__main__":
    main()


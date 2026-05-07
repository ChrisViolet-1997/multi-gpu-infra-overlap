#!/usr/bin/env python3
"""
Simple profiler that records CUDA events timing for overlap analysis.
"""

import torch
import torch.distributed as dist
import os
import json
from tp_overlap_poc import (
    BaselineRowParallelLinear,
    OverlapRowParallelLinear,
    setup_distributed,
    cleanup_distributed,
)


def detailed_timing_profile(mode: str, rank: int):
    """Profile with detailed CUDA event timing."""

    device = f"cuda:{rank}"

    # Config
    batch_size = 1
    seq_len = 2048
    in_features = 4096
    out_features = 12288
    num_chunks = 4

    # Create input
    torch.manual_seed(42 + rank)
    input_tensor = torch.randn(
        batch_size, seq_len, in_features,
        device=device, dtype=torch.float32
    )

    # Create layer
    if mode == "baseline":
        layer = BaselineRowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            process_group=None,
            device=device,
        )
    else:
        layer = OverlapRowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            num_chunks=num_chunks,
            process_group=None,
            device=device,
        )

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = layer(input_tensor)
        torch.cuda.synchronize()

    # Detailed timing
    timings = []
    with torch.no_grad():
        for i in range(10):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = layer(input_tensor)
            end.record()

            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
            timings.append(elapsed)

    avg_time = sum(timings) / len(timings)

    if rank == 0:
        print(f"\n{mode.upper()} Mode:")
        print(f"  Average time: {avg_time:.3f} ms")
        print(f"  Min time: {min(timings):.3f} ms")
        print(f"  Max time: {max(timings):.3f} ms")

        # Save to file
        result = {
            "mode": mode,
            "avg_time_ms": avg_time,
            "min_time_ms": min(timings),
            "max_time_ms": max(timings),
            "all_timings_ms": timings,
            "config": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "in_features": in_features,
                "out_features": out_features,
                "num_chunks": num_chunks,
            }
        }

        with open(f"{mode}_timing.json", 'w') as f:
            json.dump(result, f, indent=2)

    return avg_time


def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    setup_distributed(rank, world_size)

    if rank == 0:
        print("="*80)
        print("DETAILED TIMING ANALYSIS")
        print("="*80)

    baseline_time = detailed_timing_profile("baseline", rank)
    overlap_time = detailed_timing_profile("overlap", rank)

    if rank == 0:
        speedup = baseline_time / overlap_time
        print(f"\n{'='*80}")
        print(f"COMPARISON")
        print(f"{'='*80}")
        print(f"Baseline: {baseline_time:.3f} ms")
        print(f"Overlap:  {overlap_time:.3f} ms")
        print(f"Speedup:  {speedup:.2f}x")
        print(f"{'='*80}")

        if speedup < 1.0:
            print("\n⚠ Overlap is SLOWER than baseline")
            print("\nPossible reasons:")
            print("1. Chunking overhead > communication hiding benefit")
            print("2. GEMM kernels are too small after chunking")
            print("3. Stream synchronization overhead")
            print("4. Communication time is too small to hide")
        else:
            print(f"\n✓ Achieved {speedup:.2f}x speedup!")

    cleanup_distributed()


if __name__ == "__main__":
    main()

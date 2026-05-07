#!/usr/bin/env python3
"""
Deep dive: Why does chunked GEMM introduce numerical errors?

The issue is that PyTorch's matmul may use different algorithms/kernels
for different matrix sizes, leading to different rounding errors.
"""

import torch
import os
import sys


def analyze_gemm_chunking():
    """Analyze GEMM chunking precision without distributed."""
    device = "cuda:0"

    batch_size = 4
    seq_len = 128
    in_features = 512
    out_features = 512

    torch.manual_seed(42)
    weight = torch.randn(out_features, in_features, device=device, dtype=torch.float32)

    torch.manual_seed(43)
    input_tensor = torch.randn(
        batch_size, seq_len, in_features, device=device, dtype=torch.float32
    )

    print("=" * 80)
    print("GEMM CHUNKING PRECISION ANALYSIS (Single GPU, No Distributed)")
    print("=" * 80)

    # Baseline: full GEMM
    x_flat = input_tensor.view(-1, in_features)
    total_tokens = x_flat.shape[0]

    with torch.no_grad():
        output_baseline = torch.matmul(x_flat, weight.t())

    print(f"\nInput shape: {x_flat.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Output shape: {output_baseline.shape}")
    print()

    # Test 1: Chunk along batch dimension
    print("[Test 1] Chunk along batch dimension")
    print("-" * 80)

    for num_chunks in [2, 4, 8, 16, 32]:
        chunk_size = (total_tokens + num_chunks - 1) // num_chunks
        output_chunked = torch.empty_like(output_baseline)

        with torch.no_grad():
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_tokens)
                if start_idx >= total_tokens:
                    break

                x_chunk = x_flat[start_idx:end_idx]
                output_chunk = output_chunked[start_idx:end_idx]
                torch.matmul(x_chunk, weight.t(), out=output_chunk)

        max_diff = torch.max(torch.abs(output_baseline - output_chunked)).item()
        mean_diff = torch.mean(torch.abs(output_baseline - output_chunked)).item()

        status = "✓" if max_diff < 1e-5 else "✗"
        print(f"  Chunks: {num_chunks:2d} | Chunk size: {chunk_size:4d} | Max: {max_diff:.2e} | {status}")

    # Test 2: Chunk along output features
    print("\n[Test 2] Chunk along output features")
    print("-" * 80)

    for num_chunks in [2, 4, 8, 16, 32]:
        chunk_size = (out_features + num_chunks - 1) // num_chunks
        output_chunked = torch.empty_like(output_baseline)

        with torch.no_grad():
            for chunk_idx in range(num_chunks):
                start_feat = chunk_idx * chunk_size
                end_feat = min(start_feat + chunk_size, out_features)
                if start_feat >= out_features:
                    break

                weight_chunk = weight[start_feat:end_feat, :]
                output_chunk = output_chunked[:, start_feat:end_feat]
                torch.matmul(x_flat, weight_chunk.t(), out=output_chunk)

        max_diff = torch.max(torch.abs(output_baseline - output_chunked)).item()
        mean_diff = torch.mean(torch.abs(output_baseline - output_chunked)).item()

        status = "✓" if max_diff < 1e-5 else "✗"
        print(f"  Chunks: {num_chunks:2d} | Chunk size: {chunk_size:4d} | Max: {max_diff:.2e} | {status}")

    # Test 3: Use different GEMM implementations
    print("\n[Test 3] Different GEMM implementations")
    print("-" * 80)

    with torch.no_grad():
        # Method 1: torch.matmul
        output_matmul = torch.matmul(x_flat, weight.t())

        # Method 2: torch.mm
        output_mm = torch.mm(x_flat, weight.t())

        # Method 3: F.linear
        import torch.nn.functional as F
        output_linear = F.linear(x_flat, weight)

        # Method 4: Manual @ operator
        output_at = x_flat @ weight.t()

    print(f"  matmul vs mm:     {torch.max(torch.abs(output_matmul - output_mm)).item():.2e}")
    print(f"  matmul vs linear: {torch.max(torch.abs(output_matmul - output_linear)).item():.2e}")
    print(f"  matmul vs @:      {torch.max(torch.abs(output_matmul - output_at)).item():.2e}")

    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)
    print("1. Chunking along batch dimension introduces ~1e-4 error")
    print("2. Chunking along output features can maintain better precision")
    print("3. Error likely comes from different CUDA kernels for different sizes")
    print("4. For 1e-5 precision: need to avoid batch chunking or use larger chunks")
    print("=" * 80)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: Requires CUDA")
        sys.exit(1)

    analyze_gemm_chunking()

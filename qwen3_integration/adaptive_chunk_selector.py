#!/usr/bin/env python3
"""
Adaptive Chunk Selection Algorithm for TP Overlap Optimization

Based on insights from OPTIMIZATION_EXPERIMENTS.md:
1. Optimal overlap when Comp/Comm ratio ≈ 1.0
2. Chunk size should be power of 2 for GPU memory alignment
3. Avoid chunk_size < 128 (too much scheduling overhead)
4. More chunks provide finer-grained overlap but increase overhead
"""

import math
from typing import Tuple, Optional
import torch


class AdaptiveChunkSelector:
    """
    Adaptive chunk selection algorithm for TP overlap optimization.

    This class determines the optimal number of chunks for a given operator
    based on its computation and communication characteristics.
    """

    def __init__(
        self,
        min_chunk_size: int = 128,
        max_chunks: int = 32,
        target_comp_comm_ratio: float = 1.0,
    ):
        """
        Initialize the adaptive chunk selector.

        Args:
            min_chunk_size: Minimum tokens per chunk (avoid too small chunks)
            max_chunks: Maximum number of chunks (avoid excessive overhead)
            target_comp_comm_ratio: Target Comp/Comm ratio for optimal overlap
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunks = max_chunks
        self.target_comp_comm_ratio = target_comp_comm_ratio

    def select_chunks_by_ratio(
        self,
        comp_comm_ratio: float,
    ) -> int:
        """
        Select chunk count based on Comp/Comm ratio.

        Strategy:
        - Ratio < 0.5: Comp too fast, use 4 chunks
        - Ratio 0.5-0.8: Use 6 chunks
        - Ratio 0.8-1.2: Balanced, use 8 chunks (optimal)
        - Ratio 1.2-2.0: Use 12 chunks
        - Ratio > 2.0: Comp too slow, use 16+ chunks
        """
        if comp_comm_ratio < 0.5:
            return 4
        elif comp_comm_ratio < 0.8:
            return 6
        elif comp_comm_ratio < 1.2:
            return 8  # Optimal range
        elif comp_comm_ratio < 2.0:
            return 12
        elif comp_comm_ratio < 4.0:
            return 16
        else:
            return 24

    def select_chunks_by_dimensions(
        self,
        total_tokens: int,
        in_features: int,
        out_features: int,
    ) -> int:
        """
        Select chunk count based on operator dimensions.

        Larger output dimensions → more communication → need more chunks
        """
        # Estimate relative computation cost
        comp_cost = total_tokens * in_features * out_features

        # Estimate relative communication cost (proportional to output size)
        comm_cost = total_tokens * out_features

        # Approximate comp/comm ratio
        approx_ratio = comp_cost / (comm_cost * 1000)  # Scale factor for typical hardware

        return self.select_chunks_by_ratio(approx_ratio)

    def adjust_for_power_of_2(
        self,
        num_chunks: int,
        total_tokens: int,
    ) -> int:
        """
        Adjust chunk count to make chunk_size a power of 2.

        GPU memory alignment benefits from power-of-2 chunk sizes.
        """
        chunk_size = total_tokens // num_chunks

        # If chunk_size is already close to a power of 2, adjust num_chunks
        log2_size = math.log2(chunk_size)
        nearest_log2 = round(log2_size)
        nearest_pow2_size = 2 ** nearest_log2

        # Calculate adjusted num_chunks
        adjusted_chunks = total_tokens // nearest_pow2_size

        return max(2, adjusted_chunks)

    def select_chunks(
        self,
        total_tokens: int,
        in_features: int,
        out_features: int,
        comp_comm_ratio: Optional[float] = None,
    ) -> Tuple[int, int, str]:
        """
        Select optimal chunk count for an operator.

        Args:
            total_tokens: Total number of tokens (batch_size * seq_len)
            in_features: Input feature dimension
            out_features: Output feature dimension
            comp_comm_ratio: Measured Comp/Comm ratio (if available)

        Returns:
            (num_chunks, chunk_size, reason)
        """
        # Step 1: Base selection
        if comp_comm_ratio is not None:
            # Use measured ratio if available
            base_chunks = self.select_chunks_by_ratio(comp_comm_ratio)
            reason = f"Based on measured Comp/Comm={comp_comm_ratio:.2f}"
        else:
            # Estimate from dimensions
            base_chunks = self.select_chunks_by_dimensions(
                total_tokens, in_features, out_features
            )
            reason = "Based on operator dimensions"

        # Step 2: Adjust for power of 2
        adjusted_chunks = self.adjust_for_power_of_2(base_chunks, total_tokens)

        # Step 3: Enforce constraints
        chunk_size = total_tokens // adjusted_chunks

        # Ensure chunk_size >= min_chunk_size
        if chunk_size < self.min_chunk_size:
            adjusted_chunks = total_tokens // self.min_chunk_size
            chunk_size = total_tokens // adjusted_chunks
            reason += f" (adjusted for min_chunk_size={self.min_chunk_size})"

        # Ensure num_chunks <= max_chunks
        if adjusted_chunks > self.max_chunks:
            adjusted_chunks = self.max_chunks
            chunk_size = total_tokens // adjusted_chunks
            reason += f" (capped at max_chunks={self.max_chunks})"

        # Ensure at least 2 chunks
        adjusted_chunks = max(2, adjusted_chunks)
        chunk_size = total_tokens // adjusted_chunks

        return adjusted_chunks, chunk_size, reason


class Qwen3ChunkConfig:
    """
    Chunk configuration for all operators in a Qwen3 layer.

    This class stores the optimal chunk count for each operator type.
    """

    def __init__(
        self,
        q_proj_chunks: int = 8,
        k_proj_chunks: int = 8,
        v_proj_chunks: int = 8,
        o_proj_chunks: int = 8,
        gate_proj_chunks: int = 8,
        up_proj_chunks: int = 8,
        down_proj_chunks: int = 8,
    ):
        self.q_proj_chunks = q_proj_chunks
        self.k_proj_chunks = k_proj_chunks
        self.v_proj_chunks = v_proj_chunks
        self.o_proj_chunks = o_proj_chunks
        self.gate_proj_chunks = gate_proj_chunks
        self.up_proj_chunks = up_proj_chunks
        self.down_proj_chunks = down_proj_chunks

    @classmethod
    def from_adaptive_selection(
        cls,
        total_tokens: int,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
    ) -> "Qwen3ChunkConfig":
        """
        Create chunk configuration using adaptive selection.

        Args:
            total_tokens: batch_size * seq_len
            hidden_size: Model hidden dimension
            intermediate_size: MLP intermediate dimension
            num_attention_heads: Number of attention heads
            num_key_value_heads: Number of KV heads (for GQA)
            head_dim: Dimension per head

        Returns:
            Qwen3ChunkConfig with optimal chunk counts
        """
        selector = AdaptiveChunkSelector()

        q_out = num_attention_heads * head_dim
        kv_out = num_key_value_heads * head_dim

        # Attention operators
        q_chunks, _, _ = selector.select_chunks(total_tokens, hidden_size, q_out)
        k_chunks, _, _ = selector.select_chunks(total_tokens, hidden_size, kv_out)
        v_chunks, _, _ = selector.select_chunks(total_tokens, hidden_size, kv_out)
        o_chunks, _, _ = selector.select_chunks(total_tokens, q_out, hidden_size)

        # MLP operators
        gate_chunks, _, _ = selector.select_chunks(total_tokens, hidden_size, intermediate_size)
        up_chunks, _, _ = selector.select_chunks(total_tokens, hidden_size, intermediate_size)
        down_chunks, _, _ = selector.select_chunks(total_tokens, intermediate_size, hidden_size)

        return cls(
            q_proj_chunks=q_chunks,
            k_proj_chunks=k_chunks,
            v_proj_chunks=v_chunks,
            o_proj_chunks=o_chunks,
            gate_proj_chunks=gate_chunks,
            up_proj_chunks=up_chunks,
            down_proj_chunks=down_chunks,
        )

    def print_config(self):
        """Print the chunk configuration."""
        print("=" * 80)
        print("QWEN3 CHUNK CONFIGURATION")
        print("=" * 80)
        print("Attention Operators:")
        print(f"  - q_proj:    {self.q_proj_chunks} chunks")
        print(f"  - k_proj:    {self.k_proj_chunks} chunks")
        print(f"  - v_proj:    {self.v_proj_chunks} chunks")
        print(f"  - o_proj:    {self.o_proj_chunks} chunks")
        print("\nMLP Operators:")
        print(f"  - gate_proj: {self.gate_proj_chunks} chunks")
        print(f"  - up_proj:   {self.up_proj_chunks} chunks")
        print(f"  - down_proj: {self.down_proj_chunks} chunks")
        print("=" * 80)


def main():
    """Demo adaptive chunk selection."""
    print("ADAPTIVE CHUNK SELECTION DEMO")
    print("=" * 80)

    # Test with different sequence lengths
    for seq_len in [512, 1024, 2048, 4096]:
        total_tokens = seq_len  # batch_size = 1
        print(f"\nSequence Length: {seq_len} (Total Tokens: {total_tokens})")
        print("-" * 80)

        config = Qwen3ChunkConfig.from_adaptive_selection(
            total_tokens=total_tokens,
            hidden_size=4096,
            intermediate_size=12288,
        )

        config.print_config()


if __name__ == "__main__":
    main()

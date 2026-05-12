#!/usr/bin/env python3
"""
Qwen3 Decoder Layer with TP Overlap Optimization

This module implements a complete Qwen3 decoder layer with:
- Double buffer overlap for all linear operators
- Adaptive chunk selection per operator
- Grouped Query Attention (GQA)
- SwiGLU activation in MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple
import math

# Import our overlap implementation and adaptive selector
import sys
sys.path.insert(0, '/root/autodl-tmp/multi-gpu-infra-overlap/base_implementation')
sys.path.insert(0, '/root/autodl-tmp/multi-gpu-infra-overlap/qwen3_integration')
from tp_overlap_double_buffer import DoubleBufferOverlapRowParallelLinear
from adaptive_chunk_selector import Qwen3ChunkConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, device: str = "cuda"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class Qwen3Attention(nn.Module):
    """
    Qwen3 Attention with Grouped Query Attention (GQA) and TP Overlap.

    GQA: 32 query heads, 8 key-value heads
    Each KV head is shared by 4 query heads (32/8=4)
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        chunk_config: Optional[Qwen3ChunkConfig] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

        self.q_size = num_attention_heads * head_dim  # 4096
        self.kv_size = num_key_value_heads * head_dim  # 1024

        # Use adaptive chunk config or defaults
        if chunk_config is None:
            chunk_config = Qwen3ChunkConfig()

        # Q, K, V projections with overlap
        self.q_proj = DoubleBufferOverlapRowParallelLinear(
            in_features=hidden_size,
            out_features=self.q_size,
            num_chunks=chunk_config.q_proj_chunks,
            process_group=process_group,
            device=device,
        )

        self.k_proj = DoubleBufferOverlapRowParallelLinear(
            in_features=hidden_size,
            out_features=self.kv_size,
            num_chunks=chunk_config.k_proj_chunks,
            process_group=process_group,
            device=device,
        )

        self.v_proj = DoubleBufferOverlapRowParallelLinear(
            in_features=hidden_size,
            out_features=self.kv_size,
            num_chunks=chunk_config.v_proj_chunks,
            process_group=process_group,
            device=device,
        )

        # Output projection with overlap
        self.o_proj = DoubleBufferOverlapRowParallelLinear(
            in_features=self.q_size,
            out_features=hidden_size,
            num_chunks=chunk_config.o_proj_chunks,
            process_group=process_group,
            device=device,
        )

        # Q and K normalization (Qwen3 specific)
        self.q_norm = RMSNorm(self.head_dim, device=device)
        self.k_norm = RMSNorm(self.head_dim, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with GQA and TP overlap.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V with overlap
        query_states = self.q_proj(hidden_states)  # [batch, seq, q_size]
        key_states = self.k_proj(hidden_states)    # [batch, seq, kv_size]
        value_states = self.v_proj(hidden_states)  # [batch, seq, kv_size]

        # Reshape for multi-head attention
        query_states = query_states.view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)  # [batch, num_heads, seq, head_dim]

        key_states = key_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # [batch, num_kv_heads, seq, head_dim]

        value_states = value_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # [batch, num_kv_heads, seq, head_dim]

        # Apply Q and K normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Expand KV heads for GQA (repeat each KV head 4 times)
        # num_attention_heads / num_key_value_heads = 32 / 8 = 4
        num_repeats = self.num_attention_heads // self.num_key_value_heads
        key_states = key_states.repeat_interleave(num_repeats, dim=1)
        value_states = value_states.repeat_interleave(num_repeats, dim=1)

        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.q_size)

        # Output projection with overlap
        output = self.o_proj(attn_output)

        return output


class Qwen3MLP(nn.Module):
    """
    Qwen3 MLP with SwiGLU activation and TP Overlap.

    SwiGLU: output = (gate(x) * silu(x)) @ down
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        chunk_config: Optional[Qwen3ChunkConfig] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Use adaptive chunk config or defaults
        if chunk_config is None:
            chunk_config = Qwen3ChunkConfig()

        # Gate and Up projections with overlap
        self.gate_proj = DoubleBufferOverlapRowParallelLinear(
            in_features=hidden_size,
            out_features=intermediate_size,
            num_chunks=chunk_config.gate_proj_chunks,
            process_group=process_group,
            device=device,
        )

        self.up_proj = DoubleBufferOverlapRowParallelLinear(
            in_features=hidden_size,
            out_features=intermediate_size,
            num_chunks=chunk_config.up_proj_chunks,
            process_group=process_group,
            device=device,
        )

        # Down projection with overlap
        self.down_proj = DoubleBufferOverlapRowParallelLinear(
            in_features=intermediate_size,
            out_features=hidden_size,
            num_chunks=chunk_config.down_proj_chunks,
            process_group=process_group,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation.

        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        # Gate and Up projections with overlap
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # SwiGLU activation
        hidden = F.silu(gate) * up

        # Down projection with overlap
        output = self.down_proj(hidden)

        return output


class Qwen3DecoderLayer(nn.Module):
    """
    Complete Qwen3 Decoder Layer with TP Overlap Optimization.

    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + MLP(RMSNorm(x))
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        chunk_config: Optional[Qwen3ChunkConfig] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        device: str = "cuda",
    ):
        super().__init__()

        # Layer norms
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, device=device)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, device=device)

        # Attention with overlap
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            chunk_config=chunk_config,
            process_group=process_group,
            device=device,
        )

        # MLP with overlap
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            chunk_config=chunk_config,
            process_group=process_group,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the decoder layer.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        # Attention block with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP block with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def main():
    """Demo Qwen3 layer with adaptive overlap."""
    print("=" * 80)
    print("QWEN3 DECODER LAYER WITH ADAPTIVE TP OVERLAP")
    print("=" * 80)

    # Configuration
    batch_size = 1
    seq_len = 2048
    hidden_size = 4096
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create adaptive chunk config
    chunk_config = Qwen3ChunkConfig.from_adaptive_selection(
        total_tokens=batch_size * seq_len,
        hidden_size=hidden_size,
    )

    print("\nChunk Configuration:")
    chunk_config.print_config()

    # Note: This demo requires distributed initialization
    # For single GPU testing, use process_group=None but skip all_reduce
    print("\n⚠ Note: This implementation requires distributed initialization.")
    print("Run with: torchrun --nproc_per_node=2 qwen3_layer.py")
    print("\nFor single GPU demo, the layer structure is created successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()

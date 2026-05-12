#!/usr/bin/env python3
"""
CUDAGraph-optimized Qwen3 Decoder Layer.

This is a new implementation that doesn't modify existing code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional
import math
import sys
import os

# IMPORTANT: Add current directory first to avoid importing old version
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

sys.path.insert(0, '/root/autodl-tmp/multi-gpu-infra-overlap/base_implementation')
sys.path.insert(0, '/root/autodl-tmp/multi-gpu-infra-overlap/qwen3_integration')

from tp_overlap_cudagraph import CUDAGraphDoubleBufferOverlapRowParallelLinear
from qwen3_layer import RMSNorm  # Reuse existing RMSNorm
from adaptive_chunk_selector import Qwen3ChunkConfig


class CUDAGraphQwen3Attention(nn.Module):
    """CUDAGraph-optimized Qwen3 Attention."""

    def __init__(
        self,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        chunk_config: Optional[Qwen3ChunkConfig] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        device: str = "cuda",
        static_input_shape: Optional[tuple] = None,  # (batch, seq, hidden)
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

        self.q_size = num_attention_heads * head_dim
        self.kv_size = num_key_value_heads * head_dim

        if chunk_config is None:
            chunk_config = Qwen3ChunkConfig()

        # Q, K, V projections with CUDAGraph
        self.q_proj = CUDAGraphDoubleBufferOverlapRowParallelLinear(
            in_features=hidden_size,
            out_features=self.q_size,
            num_chunks=chunk_config.q_proj_chunks,
            process_group=process_group,
            device=device,
            static_input_shape=static_input_shape,
            enable_graph=False,
        )

        self.k_proj = CUDAGraphDoubleBufferOverlapRowParallelLinear(
            in_features=hidden_size,
            out_features=self.kv_size,
            num_chunks=chunk_config.k_proj_chunks,
            process_group=process_group,
            device=device,
            static_input_shape=static_input_shape,
            enable_graph=False,
        )

        self.v_proj = CUDAGraphDoubleBufferOverlapRowParallelLinear(
            in_features=hidden_size,
            out_features=self.kv_size,
            num_chunks=chunk_config.v_proj_chunks,
            process_group=process_group,
            device=device,
            static_input_shape=static_input_shape,
            enable_graph=False,
        )

        # Output projection
        # Note: o_proj input shape is different (batch, seq, q_size)
        o_proj_input_shape = None
        if static_input_shape is not None:
            batch, seq, _ = static_input_shape
            o_proj_input_shape = (batch, seq, self.q_size)

        self.o_proj = CUDAGraphDoubleBufferOverlapRowParallelLinear(
            in_features=self.q_size,
            out_features=hidden_size,
            num_chunks=chunk_config.o_proj_chunks,
            process_group=process_group,
            device=device,
            static_input_shape=o_proj_input_shape,
            enable_graph=False,
        )

        # Q and K normalization
        self.q_norm = RMSNorm(self.head_dim, device=device)
        self.k_norm = RMSNorm(self.head_dim, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Note: batch_size and seq_len are fixed at init time
        # All shapes are static, so view/transpose can be captured in graph

        # Project Q, K, V (CUDAGraph)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention (can be captured in graph)
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        query_states = query_states.view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)

        key_states = key_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        value_states = value_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Apply Q and K normalization (can be captured)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Expand KV heads for GQA (can be captured)
        num_repeats = self.num_attention_heads // self.num_key_value_heads
        key_states = key_states.repeat_interleave(num_repeats, dim=1)
        value_states = value_states.repeat_interleave(num_repeats, dim=1)

        # Scaled dot-product attention (can be captured)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        # Note: attention_mask must be None or pre-allocated static tensor
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values (can be captured)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back (can be captured)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.q_size)

        # Output projection (CUDAGraph)
        output = self.o_proj(attn_output)

        return output


class CUDAGraphQwen3MLP(nn.Module):
    """CUDAGraph-optimized Qwen3 MLP."""

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        chunk_config: Optional[Qwen3ChunkConfig] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        device: str = "cuda",
        static_input_shape: Optional[tuple] = None,  # (batch, seq, hidden)
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        if chunk_config is None:
            chunk_config = Qwen3ChunkConfig()

        # Gate and Up projections
        self.gate_proj = CUDAGraphDoubleBufferOverlapRowParallelLinear(
            in_features=hidden_size,
            out_features=intermediate_size,
            num_chunks=chunk_config.gate_proj_chunks,
            process_group=process_group,
            device=device,
            static_input_shape=static_input_shape,
            enable_graph=False,
        )

        self.up_proj = CUDAGraphDoubleBufferOverlapRowParallelLinear(
            in_features=hidden_size,
            out_features=intermediate_size,
            num_chunks=chunk_config.up_proj_chunks,
            process_group=process_group,
            device=device,
            static_input_shape=static_input_shape,
            enable_graph=False,
        )

        # Down projection
        # Note: down_proj input shape is different (batch, seq, intermediate)
        down_proj_input_shape = None
        if static_input_shape is not None:
            batch, seq, _ = static_input_shape
            down_proj_input_shape = (batch, seq, intermediate_size)

        self.down_proj = CUDAGraphDoubleBufferOverlapRowParallelLinear(
            in_features=intermediate_size,
            out_features=hidden_size,
            num_chunks=chunk_config.down_proj_chunks,
            process_group=process_group,
            device=device,
            static_input_shape=down_proj_input_shape,
            enable_graph=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # All operations can be captured in graph since shapes are static

        # Gate and Up projections (CUDAGraph)
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # SwiGLU activation (can be captured)
        hidden = F.silu(gate) * up

        # Down projection (CUDAGraph)
        output = self.down_proj(hidden)

        return output


class CUDAGraphQwen3DecoderLayer(nn.Module):
    """
    CUDAGraph-optimized Qwen3 Decoder Layer.

    Captures the ENTIRE forward pass in a single graph, including:
    - All 7 linear operators
    - View/transpose operations
    - Attention computation
    - Activations
    - Residual connections
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
        static_input_shape: Optional[tuple] = None,  # (batch, seq, hidden)
    ):
        super().__init__()

        self.static_input_shape = static_input_shape
        self.device = device

        # Layer norms
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, device=device)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, device=device)

        # Attention with CUDAGraph
        self.self_attn = CUDAGraphQwen3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            chunk_config=chunk_config,
            process_group=process_group,
            device=device,
            static_input_shape=static_input_shape,
        )

        # MLP with CUDAGraph
        self.mlp = CUDAGraphQwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            chunk_config=chunk_config,
            process_group=process_group,
            device=device,
            static_input_shape=static_input_shape,
        )

        # Layer-level graph
        self.layer_graph = None
        self.layer_graph_captured = False

        # Pre-allocate static buffers for layer-level graph
        if static_input_shape is not None:
            self._preallocate_layer_buffers(static_input_shape)

    def _preallocate_layer_buffers(self, input_shape: tuple):
        """Pre-allocate buffers for layer-level graph capture."""
        batch, seq, hidden = input_shape

        self.static_input = torch.empty(input_shape, device=self.device)
        self.static_output = torch.empty(input_shape, device=self.device)

    def _capture_layer_graph(self, hidden_states: torch.Tensor):
        """Capture the entire layer forward pass as a single graph."""
        if self.layer_graph_captured:
            return

        if self.static_input is None:
            print("Warning: Cannot capture layer graph without static buffers")
            return

        print(f"Capturing layer-level CUDAGraph for shape {hidden_states.shape}...")

        # Copy input to static buffer
        self.static_input.copy_(hidden_states)

        # Warmup
        for _ in range(3):
            self._forward_impl(self.static_input, self.static_output)
        torch.cuda.synchronize()

        # Capture
        self.layer_graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.layer_graph):
            self._forward_impl(self.static_input, self.static_output)

        torch.cuda.synchronize()

        self.layer_graph_captured = True
        print(f"✓ Layer graph captured successfully")

    def _forward_impl(self, hidden_states: torch.Tensor, output: torch.Tensor):
        """
        Actual forward implementation.

        This entire function will be captured as a single graph.
        """
        # Attention block with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=None)
        hidden_states = residual + hidden_states

        # MLP block with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Copy to output
        output.copy_(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with layer-level CUDAGraph.
        """
        # Check if we can use layer graph
        if self.static_input_shape is not None:
            if hidden_states.shape != self.static_input_shape:
                raise RuntimeError(
                    f"Input shape {hidden_states.shape} does not match "
                    f"static shape {self.static_input_shape}"
                )

            # Capture on first call
            if not self.layer_graph_captured:
                self._capture_layer_graph(hidden_states)

            # Replay graph
            if self.layer_graph is not None:
                # Copy input to static buffer
                self.static_input.copy_(hidden_states)

                # Replay
                self.layer_graph.replay()

                # Return output
                return self.static_output

        # Fallback to eager mode
        output = torch.empty_like(hidden_states)
        self._forward_impl(hidden_states, output)
        return output


def main():
    """Test CUDAGraph Qwen3 layer."""
    print("=" * 80)
    print("CUDAGRAPH QWEN3 DECODER LAYER TEST")
    print("=" * 80)

    # Configuration
    batch_size = 32
    seq_len = 1
    hidden_size = 4096
    device = "cuda"

    # Create chunk config
    chunk_config = Qwen3ChunkConfig(
        q_proj_chunks=4,
        k_proj_chunks=4,
        v_proj_chunks=4,
        o_proj_chunks=4,
        gate_proj_chunks=4,
        up_proj_chunks=4,
        down_proj_chunks=4,
    )

    print("\nChunk Configuration:")
    chunk_config.print_config()

    # Create layer
    layer = CUDAGraphQwen3DecoderLayer(
        hidden_size=hidden_size,
        intermediate_size=12288,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        chunk_config=chunk_config,
        device=device,
        static_input_shape=(batch_size, seq_len, hidden_size),
    )

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    print(f"\nInput shape: {x.shape}")
    print(f"\nFirst forward (will capture graphs for all 7 operators)...")

    # First forward
    with torch.no_grad():
        y = layer(x)

    print(f"Output shape: {y.shape}")

    print(f"\nSubsequent forwards (replaying graphs)...")
    with torch.no_grad():
        for i in range(3):
            y = layer(x)

    print("\n✓ All forwards completed successfully")
    print("=" * 80)


if __name__ == "__main__":
    main()

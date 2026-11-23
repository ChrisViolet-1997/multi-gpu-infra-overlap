#!/usr/bin/env python3
"""Compare chunked Qwen3 projection arithmetic with a local HF model layer."""

import argparse
import copy
from pathlib import Path
import sys

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "graph_implementation" / "core"))
sys.path.insert(0, str(repo_root / "base_implementation"))


PROJECTIONS = {
    "self_attn.q_proj": "column",
    "self_attn.k_proj": "column",
    "self_attn.v_proj": "column",
    "self_attn.o_proj": "row",
    "mlp.gate_proj": "column",
    "mlp.up_proj": "column",
    "mlp.down_proj": "row",
}


def replace_projection(
    layer: nn.Module,
    dotted_name: str,
    mode: str,
    chunks: int,
    backend: str,
    batch_size: int,
    seq_len: int,
) -> None:
    parent_name, child_name = dotted_name.rsplit(".", 1)
    parent = layer.get_submodule(parent_name)
    original = getattr(parent, child_name)
    if backend == "eager":
        from tp_overlap_double_buffer import DoubleBufferOverlapRowParallelLinear

        replacement = DoubleBufferOverlapRowParallelLinear(
            original.in_features,
            original.out_features,
            num_chunks=chunks,
            device=original.weight.device,
            parallel_mode=mode,
        )
    else:
        from tp_overlap_cudagraph import CUDAGraphDoubleBufferOverlapRowParallelLinear

        replacement = CUDAGraphDoubleBufferOverlapRowParallelLinear(
            original.in_features,
            original.out_features,
            num_chunks=chunks,
            device=original.weight.device,
            static_input_shape=(batch_size, seq_len, original.in_features),
            enable_graph=True,
            parallel_mode=mode,
        )
    replacement.load_from_linear(original)
    setattr(parent, child_name, replacement)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / "code/model/Qwen3-0.6b",
    )
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--backend", choices=("eager", "cuda_graph"), default="eager")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this local verification")
    if not (args.model / "config.json").is_file():
        raise FileNotFoundError(f"No config.json found in model directory: {args.model}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    ).eval().to(device)

    # Retain only one real checkpoint layer so this test stays light on VRAM.
    layer = copy.deepcopy(model.model.layers[0])
    model.model.layers = nn.ModuleList([layer])
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(
        0, vocab_size, (args.batch_size, args.seq_len), device=device
    )

    with torch.inference_mode():
        reference = model.model(input_ids=input_ids, use_cache=False).last_hidden_state

        for name, mode in PROJECTIONS.items():
            replace_projection(
                layer,
                name,
                mode,
                args.chunks,
                args.backend,
                args.batch_size,
                args.seq_len,
            )

        chunked = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
        repeated = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
        torch.cuda.synchronize()

    diff = torch.maximum((reference - chunked).abs(), (reference - repeated).abs())
    max_abs = diff.max().item()
    rms = diff.float().square().mean().sqrt().item()
    passed = torch.allclose(reference, chunked, atol=2e-5, rtol=2e-5)

    print(f"Model: {args.model}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(
        f"Dtype: float32, backend={args.backend}, batch={args.batch_size}, "
        f"seq_len={args.seq_len}, chunks={args.chunks}"
    )
    print(f"Compared layer output shape: {tuple(reference.shape)}")
    print(f"Max absolute error: {max_abs:.8e}")
    print(f"RMS error: {rms:.8e}")
    print("PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

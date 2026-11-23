#!/usr/bin/env python3
"""Legacy eager entry point backed by the shared TPLinear implementation."""

import os
import sys
from typing import Optional

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "graph_implementation", "core"))
from tp_linear import TPLinear


class DoubleBufferOverlapRowParallelLinear(TPLinear):
    """Eager row-parallel TP linear; chunks partition tokens, not output features."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_chunks: int = 4,
        process_group: Optional[dist.ProcessGroup] = None,
        device: str = "cuda",
        parallel_mode: str = "row",
        input_is_sharded: bool = False,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            num_chunks=num_chunks,
            parallel_mode=parallel_mode,
            backend="eager",
            process_group=process_group,
            device=device,
            dtype=torch.float32,
            overlap=True,
            input_is_sharded=input_is_sharded,
        )


def main():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    layer = DoubleBufferOverlapRowParallelLinear(4096, 4096, num_chunks=4, device=device)
    x = torch.randn(32, 1, 4096, device=device)
    with torch.inference_mode():
        for _ in range(5):
            output = layer(x)
        torch.cuda.synchronize()
    if local_rank == 0:
        print(f"Output shape: {tuple(output.shape)}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

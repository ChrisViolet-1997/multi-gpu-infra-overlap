#!/usr/bin/env python3
"""Legacy CUDA Graph entry point backed by the shared TPLinear implementation."""

from typing import Optional

import torch.distributed as dist

from tp_linear import TPLinear


class CUDAGraphDoubleBufferOverlapRowParallelLinear(TPLinear):
    """Compatibility wrapper selecting the shared eager or graph backend."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_chunks: int = 4,
        process_group: Optional[dist.ProcessGroup] = None,
        device: str = "cuda",
        static_input_shape: Optional[tuple] = None,
        enable_graph: bool = True,
        sequential: bool = False,
        parallel_mode: str = "row",
        input_is_sharded: bool = False,
    ):
        use_graph = enable_graph and static_input_shape is not None
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            num_chunks=num_chunks,
            parallel_mode=parallel_mode,
            backend="cuda_graph" if use_graph else "eager",
            process_group=process_group,
            device=device,
            static_input_shape=static_input_shape if use_graph else None,
            overlap=not sequential,
            input_is_sharded=input_is_sharded,
        )
        self.enable_graph = use_graph
        self.sequential = sequential


def main():
    import torch

    batch_size, seq_len, features = 32, 1, 4096
    layer = CUDAGraphDoubleBufferOverlapRowParallelLinear(
        features,
        features,
        num_chunks=4,
        static_input_shape=(batch_size, seq_len, features),
        enable_graph=True,
        device="cuda",
    )
    x = torch.randn(batch_size, seq_len, features, device="cuda")
    with torch.inference_mode():
        output = layer(x)
        torch.cuda.synchronize()
    print(f"Output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()

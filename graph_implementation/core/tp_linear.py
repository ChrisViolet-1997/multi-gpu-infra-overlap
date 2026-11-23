"""Shared TP linear semantics with eager and CUDA Graph execution backends."""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


class TPLinear(nn.Module):
    """Tensor-parallel linear layer with token-chunked, double-buffered execution.

    Inference-only. Column mode shards output features and does not reduce. Row mode shards input
    features and all-reduces each token chunk. ``chunks`` always partitions the
    flattened token (M) dimension, independent of the projection mode.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_chunks: int = 1,
        parallel_mode: str = "row",
        backend: str = "eager",
        process_group: Optional[dist.ProcessGroup] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        bias: bool = False,
        static_input_shape: Optional[Tuple[int, ...]] = None,
        overlap: bool = True,
        input_is_sharded: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_chunks = num_chunks
        self.parallel_mode = parallel_mode
        self.backend = backend
        self.process_group = process_group
        self.static_input_shape = tuple(static_input_shape) if static_input_shape else None
        self.overlap = overlap
        self.input_is_sharded = input_is_sharded

        if parallel_mode not in {"row", "column"}:
            raise ValueError("parallel_mode must be 'row' or 'column'")
        if input_is_sharded and parallel_mode != "row":
            raise ValueError("input_is_sharded is only valid for row-parallel layers")
        if backend not in {"eager", "cuda_graph"}:
            raise ValueError("backend must be 'eager' or 'cuda_graph'")
        if num_chunks < 1:
            raise ValueError("num_chunks must be at least 1")
        if backend == "cuda_graph" and self.static_input_shape is None:
            raise ValueError("cuda_graph backend requires static_input_shape")
        if process_group is not None and not dist.is_initialized():
            raise RuntimeError("Initialize torch.distributed before passing a process group")

        self.world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
        self.rank = dist.get_rank(process_group) if dist.is_initialized() else 0
        shard_size = in_features if parallel_mode == "row" else out_features
        if shard_size % self.world_size:
            raise ValueError("parallel feature dimension must be divisible by TP world size")

        if parallel_mode == "row":
            local_in = in_features // self.world_size
            local_out = out_features
        else:
            local_in = in_features
            local_out = out_features // self.world_size
        self.local_in_features = local_in
        self.local_out_features = local_out
        self.weight = nn.Parameter(torch.empty(local_out, local_in, device=device, dtype=dtype))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(local_out if parallel_mode == "column" else out_features,
                                             device=device, dtype=dtype)) if bias else None

        self.compute_stream = torch.cuda.Stream(device=device) if str(device).startswith("cuda") else None
        self.comm_stream = torch.cuda.Stream(device=device) if str(device).startswith("cuda") else None
        self._buffer_pair = None
        self._consumed_events = [None, None]
        self._graph_captured = False

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        parallel_mode: str,
        num_chunks: int = 1,
        backend: str = "eager",
        process_group: Optional[dist.ProcessGroup] = None,
        static_input_shape: Optional[Tuple[int, ...]] = None,
        overlap: bool = True,
        input_is_sharded: bool = False,
    ):
        module = cls(
            linear.in_features,
            linear.out_features,
            num_chunks=num_chunks,
            parallel_mode=parallel_mode,
            backend=backend,
            process_group=process_group,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            bias=linear.bias is not None,
            static_input_shape=static_input_shape,
            overlap=overlap,
            input_is_sharded=input_is_sharded,
        )
        return module.load_from_linear(linear)

    def load_from_linear(self, linear: nn.Linear):
        if linear.in_features != self.in_features or linear.out_features != self.out_features:
            raise ValueError("source linear dimensions do not match TPLinear")
        if (linear.bias is None) != (self.bias is None):
            raise ValueError("source linear bias does not match TPLinear")
        with torch.no_grad():
            if self.parallel_mode == "row":
                shards = torch.tensor_split(linear.weight, self.world_size, dim=1)
                self.weight.copy_(shards[self.rank])
                if linear.bias is not None:
                    self.bias.copy_(linear.bias)
            else:
                shards = torch.tensor_split(linear.weight, self.world_size, dim=0)
                self.weight.copy_(shards[self.rank])
                if linear.bias is not None:
                    bias_shards = torch.tensor_split(linear.bias, self.world_size, dim=0)
                    self.bias.copy_(bias_shards[self.rank])
        return self

    def _input_shard(self, x: torch.Tensor) -> torch.Tensor:
        if self.parallel_mode == "column" or self.input_is_sharded:
            return x
        start = self.rank * self.local_in_features
        return x[..., start:start + self.local_in_features]

    def _chunk_ranges(self, token_count: int):
        chunk_count = min(self.num_chunks, token_count)
        return [
            (index * token_count // chunk_count, (index + 1) * token_count // chunk_count)
            for index in range(chunk_count)
        ]

    def _apply_bias(self, output: torch.Tensor) -> None:
        if self.bias is not None:
            output.add_(self.bias)

    def _compute_into(self, x: torch.Tensor, output: torch.Tensor) -> None:
        torch.mm(x, self.weight.t(), out=output)
        if self.parallel_mode == "column":
            self._apply_bias(output)

    def _allocate_buffers(self, max_rows: int):
        shape = (max_rows, self.local_out_features)
        if self._buffer_pair is None or self._buffer_pair[0].shape[0] < max_rows:
            self._buffer_pair = [
                torch.empty(shape, device=self.weight.device, dtype=self.weight.dtype),
                torch.empty(shape, device=self.weight.device, dtype=self.weight.dtype),
            ]
            self._consumed_events = [None, None]

    def _forward_sequential(self, x_flat: torch.Tensor, output_flat: torch.Tensor) -> None:
        for start, end in self._chunk_ranges(x_flat.shape[0]):
            local = self._input_shard(x_flat[start:end]).contiguous()
            result = torch.mm(local, self.weight.t())
            if self.parallel_mode == "row" and self.world_size > 1:
                dist.all_reduce(result, op=dist.ReduceOp.SUM, group=self.process_group)
            self._apply_bias(result)
            output_flat[start:end].copy_(result)

    def _forward_eager_cuda(self, x_flat: torch.Tensor, output_flat: torch.Tensor) -> None:
        ranges = self._chunk_ranges(x_flat.shape[0])
        max_rows = max(end - start for start, end in ranges)
        self._allocate_buffers(max_rows)
        caller_stream = torch.cuda.current_stream(self.weight.device)
        input_ready = torch.cuda.Event()
        input_ready.record(caller_stream)
        self.compute_stream.wait_event(input_ready)

        for index, (start, end) in enumerate(ranges):
            slot = index % 2
            consumed = self._consumed_events[slot]
            if consumed is not None:
                self.compute_stream.wait_event(consumed)
            buffer = self._buffer_pair[slot][:end - start]
            with torch.cuda.stream(self.compute_stream):
                local = self._input_shard(x_flat[start:end]).contiguous()
                self._compute_into(local, buffer)
                computed = torch.cuda.Event()
                computed.record(self.compute_stream)

            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(computed)
                if self.parallel_mode == "row" and self.world_size > 1:
                    dist.all_reduce(buffer, op=dist.ReduceOp.SUM, group=self.process_group)
                if self.parallel_mode == "row":
                    self._apply_bias(buffer)
                communicated = torch.cuda.Event()
                communicated.record(self.comm_stream)

            if not self.overlap:
                self.compute_stream.wait_event(communicated)
            caller_stream.wait_event(communicated)
            output_flat[start:end].copy_(buffer)
            consumed = torch.cuda.Event()
            consumed.record(caller_stream)
            self._consumed_events[slot] = consumed


    def _capture_graphs(self, x_flat: torch.Tensor, ranges) -> None:
        max_rows = max(end - start for start, end in ranges)
        self._allocate_buffers(max_rows)
        self._static_input = torch.empty_like(x_flat)
        local_width = self.local_in_features if self.parallel_mode == "row" else self.in_features
        self._static_local_input = torch.empty(
            x_flat.shape[0], local_width, device=x_flat.device, dtype=x_flat.dtype
        )

        warmup_stream = torch.cuda.Stream(device=self.weight.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.weight.device))
        with torch.cuda.stream(warmup_stream):
            for index, (start, end) in enumerate(ranges):
                slot = index % 2
                local = self._static_local_input[start:end]
                buffer = self._buffer_pair[slot][:end - start]
                for _ in range(2):
                    self._compute_into(local, buffer)
        warmup_stream.synchronize()

        self.compute_graphs = []
        with torch.cuda.stream(warmup_stream):
            for index, (start, end) in enumerate(ranges):
                slot = index % 2
                local = self._static_local_input[start:end]
                buffer = self._buffer_pair[slot][:end - start]
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=warmup_stream):
                    self._compute_into(local, buffer)
                self.compute_graphs.append(graph)
        warmup_stream.synchronize()
        self._graph_captured = True

    def _forward_graph(self, x_flat: torch.Tensor, output_flat: torch.Tensor) -> None:
        expected_tokens = 1
        for size in self.static_input_shape[:-1]:
            expected_tokens *= size
        expected_width = self.local_in_features if self.input_is_sharded else self.in_features
        if tuple(x_flat.shape) != (expected_tokens, expected_width):
            raise RuntimeError("CUDA Graph backend received an input shape different from capture shape")
        ranges = self._chunk_ranges(x_flat.shape[0])
        if not self._graph_captured:
            self._capture_graphs(x_flat, ranges)

        caller_stream = torch.cuda.current_stream(self.weight.device)
        self._static_input.copy_(x_flat)
        self._static_local_input.copy_(self._input_shard(self._static_input))
        input_ready = torch.cuda.Event()
        input_ready.record(caller_stream)
        self.compute_stream.wait_event(input_ready)

        for index, (start, end) in enumerate(ranges):
            slot = index % 2
            consumed = self._consumed_events[slot]
            if consumed is not None:
                self.compute_stream.wait_event(consumed)
            with torch.cuda.stream(self.compute_stream):
                self.compute_graphs[index].replay()
                computed = torch.cuda.Event()
                computed.record(self.compute_stream)

            buffer = self._buffer_pair[slot][:end - start]
            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(computed)
                if self.parallel_mode == "row" and self.world_size > 1:
                    dist.all_reduce(buffer, op=dist.ReduceOp.SUM, group=self.process_group)
                if self.parallel_mode == "row":
                    self._apply_bias(buffer)
                communicated = torch.cuda.Event()
                communicated.record(self.comm_stream)

            if not self.overlap:
                self.compute_stream.wait_event(communicated)
            caller_stream.wait_event(communicated)
            output_flat[start:end].copy_(buffer)
            consumed = torch.cuda.Event()
            consumed.record(caller_stream)
            self._consumed_events[slot] = consumed


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_width = self.local_in_features if self.input_is_sharded else self.in_features
        if x.shape[-1] != expected_width:
            raise ValueError(f"expected input width {expected_width}, got {x.shape[-1]}")
        if self.static_input_shape is not None and tuple(x.shape) != self.static_input_shape:
            raise RuntimeError(
                f"input shape {tuple(x.shape)} does not match static shape {self.static_input_shape}"
            )
        x_flat = x.reshape(-1, expected_width)
        output_flat = torch.empty(
            x_flat.shape[0], self.local_out_features, device=x.device, dtype=self.weight.dtype
        )

        if self.backend == "cuda_graph":
            self._forward_graph(x_flat, output_flat)
        elif x.is_cuda and self.overlap and self.num_chunks > 1:
            self._forward_eager_cuda(x_flat, output_flat)
        else:
            self._forward_sequential(x_flat, output_flat)
        return output_flat.reshape(*x.shape[:-1], self.local_out_features)

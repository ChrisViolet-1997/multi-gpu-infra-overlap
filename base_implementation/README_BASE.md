# Base Entry Point

本目录的 `DoubleBufferOverlapRowParallelLinear` 是共享 TP linear 的 eager 兼容入口。分片、token chunking、双缓冲和 all-reduce 逻辑统一维护在 `graph_implementation/core/tp_linear.py`，避免 eager 与 CUDA Graph 两份实现各自演进。

单卡参考验证和当前限制见仓库根目录 [README.md](../README.md)。

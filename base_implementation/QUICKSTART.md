# Quick Start

在本机用 Qwen3-0.6B 验证共享 TP linear 的 eager 与 CUDA Graph 后端：

```bash
~/code/mini-sglang/.venv/bin/python graph_implementation/experiments/verify_qwen3_single_gpu.py \
  --model ~/code/model/Qwen3-0.6b --chunks 4 --backend eager

~/code/mini-sglang/.venv/bin/python graph_implementation/experiments/verify_qwen3_single_gpu.py \
  --model ~/code/model/Qwen3-0.6b --chunks 4 --backend cuda_graph
```

这验证单卡前向数值，不验证多 GPU 通信或 overlap 性能。eager 类入口 `DoubleBufferOverlapRowParallelLinear` 位于 `tp_overlap_double_buffer.py`；CUDA Graph 兼容入口 `CUDAGraphDoubleBufferOverlapRowParallelLinear` 位于 `graph_implementation/core/tp_overlap_cudagraph.py`。两者共用 `TPLinear`。

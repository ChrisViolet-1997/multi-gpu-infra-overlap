# Multi-GPU TP Overlap Demo

这是一个研究性 demo，探索 Tensor Parallel (TP) 线性层中的计算与通信重叠。仓库包含早期 NCCL/CUDA stream 实验，以及一个可在单卡上对照 Hugging Face Qwen3 参考结果的 chunked linear 验证路径。

## 当前可复现验证

统一实现 `TPLinear` 提供 column/row parallel 语义、token 维 chunking、双缓冲，以及 eager/CUDA Graph 两个执行后端。验证脚本加载本地 Qwen3-0.6B 权重，对 HF decoder layer 0 的七个投影替换为该实现，再比较替换前后的 layer 输出。

```bash
~/code/mini-sglang/.venv/bin/python graph_implementation/experiments/verify_qwen3_single_gpu.py \
  --model ~/code/model/Qwen3-0.6b \
  --chunks 4 --backend eager
```

默认使用 CUDA 和 FP32，输入为 batch=2、seq_len=16。用 `--backend cuda_graph` 可切换后端；Graph 后端要求静态输入形状。脚本会打印设备、输出形状、最大绝对误差和 RMS 误差；误差超出容差会以非零状态退出。可通过 `--batch-size`、`--seq-len`、`--chunks` 和 `--model` 调整验证参数。

最近一次本机验证环境：PyTorch 2.9.1+cu128、Transformers 4.57.3、NVIDIA GeForce RTX 4060 Ti。eager 与 CUDA Graph 后端均连续执行两次并通过；`chunks=4` 时最大绝对误差为 `3.05e-5`，RMS 误差为 `2.00e-6`。

### 验证边界

- 单卡运行只验证分块计算与 HF 参考层的数值一致性。
- `tp_size > 1` 的权重分片和 row-parallel all-reduce 路径尚未在多卡上验证。
- 这不是通信 overlap benchmark，也不能据此声称生产端到端加速。
- 仓库中早期性能报告对应旧实现和特定负载，结果没有通过当前参考路径重新复现。

## 代码结构

| 路径 | 内容 |
|---|---|
| `graph_implementation/core/tp_linear.py` | 共享 TP 分片、token chunking、double buffer 和 eager/CUDA Graph 后端 |
| `graph_implementation/experiments/verify_qwen3_single_gpu.py` | 本地 HF 模型单卡数值对照 |
| `graph_implementation/core/tp_overlap_cudagraph.py` | 共享 `TPLinear` 的 CUDA Graph 兼容入口 |
| `graph_implementation/core/qwen3_layer_cudagraph.py` | 早期自定义 Qwen3 layer 原型，尚未对齐 HF 全层行为 |
| `graph_implementation/experiments/` | 正确性、grid search、整层 benchmark 和历史实验记录 |
| `base_implementation/` | 初始 double-buffer overlap PoC |
| `parameter_grid_search/` | 初始参数搜索脚本与报告 |
| `qwen3_integration/` | 初始 Qwen3 集成和 chunk 选择逻辑 |
| `archive/`, `profiles/`, `scripts/` | 历史分析脚本、profiling 输出和辅助工具 |

## 历史实验说明

`parameter_grid_search/OPTIMIZATION_EXPERIMENTS.md`、`qwen3_integration/QWEN3_PER_OPERATOR_OPTIMIZATION.md` 和 `graph_implementation/experiments/EXPERIMENT_SUMMARY.md` 保存了早期实验记录。报告中的速度提升和 token 阈值不应视为经过当前代码验证的结论。

当前主要入口是上面的单卡数值验证。执行多 GPU benchmark 前，需要先以真实 TP 布局重新集成和验证 column-parallel 与 row-parallel 算子，并在目标 GPU 互联上测量。

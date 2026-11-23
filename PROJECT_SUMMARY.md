# 项目状态总结

## 项目目标

本仓库记录了以 chunking、CUDA streams、NCCL 和 double buffering 探索 TP 计算-通信 overlap 的过程。它目前是研究 demo，不是可直接用于生产的 TP 实现。

## 当前验证路径

`graph_implementation/core/tp_linear.py` 是统一实现，两个旧入口都委托给它：

- Column-parallel：每个 rank 持有输出维权重分片，不做 all-reduce。
- Row-parallel：每个 rank 持有输入维权重分片，局部计算后对每个 token chunk all-reduce。
- Chunk 沿 token/M 维切分；双缓冲协调计算与通信。
- `backend="eager"` 和 `backend="cuda_graph"` 共用分片、chunk 与通信逻辑。

`base_implementation/tp_overlap_double_buffer.py` 与 `graph_implementation/core/tp_overlap_cudagraph.py` 保留旧类名作为兼容包装。`graph_implementation/experiments/verify_qwen3_single_gpu.py` 将共享层用于本地 Qwen3-0.6B 第一层，并与 Hugging Face 原层比较输出。单卡下 `tp_size=1`，因此验证的是计算一致性，不包含跨 GPU 通信。

运行方式：

```bash
~/code/mini-sglang/.venv/bin/python graph_implementation/experiments/verify_qwen3_single_gpu.py \
  --model ~/code/model/Qwen3-0.6b --chunks 4 --backend eager
```

CUDA Graph 后端验证：

```bash
~/code/mini-sglang/.venv/bin/python graph_implementation/experiments/verify_qwen3_single_gpu.py \
  --model ~/code/model/Qwen3-0.6b --chunks 4 --backend cuda_graph
```

最近一次验证使用 PyTorch 2.9.1+cu128、Transformers 4.57.3、RTX 4060 Ti、FP32、batch=2、seq_len=16。eager 和 CUDA Graph 均连续执行两次并通过；4 chunks 的输出形状为 `(2, 16, 1024)`，最大绝对误差 `3.05e-5`，RMS 误差 `2.00e-6`。

## 尚未验证

- `tp_size > 1` 下真实多 GPU 的 shard 分发和 NCCL all-reduce。
- CUDA stream 上计算与通信的实际并行度及端到端收益。
- 早期自定义 Qwen3 layer 是否与完整 HF Qwen3 在所有输入条件下等价。
- 早期报告中的 chunk 最优值、速度提升和 total-token 阈值能否在修正后的 TP 布局中复现。

## 目录索引

| 路径 | 说明 |
|---|---|
| `graph_implementation/core/` | 共享 TP linear、兼容入口和自定义 Qwen3 layer 原型 |
| `graph_implementation/experiments/` | 单卡正确性验证及历史多 GPU 实验脚本/报告 |
| `base_implementation/` | 初始 overlap PoC、环境和正确性工具 |
| `parameter_grid_search/` | 早期参数搜索实现与实验记录 |
| `qwen3_integration/` | 早期 Qwen3 layer 和 chunk selector |
| `archive/` | 历史调试、分析和比较脚本 |
| `profiles/` | 历史 profiler 输出 |
| `scripts/` | profiling 与分析辅助脚本 |

## 维护说明

早期报告作为实验历史保留，但其中的结果不是当前实现的验证结果。`README.md` 描述可复现入口和验证边界；本文件记录项目当前状态与剩余验证项。

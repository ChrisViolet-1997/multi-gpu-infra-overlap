# Graph Implementation

本目录包含两类代码：早期 CUDA Graph/NCCL overlap 原型，以及当前用于单卡数值对照的 Qwen3 chunked linear。

## 单卡参考验证

```bash
~/code/mini-sglang/.venv/bin/python graph_implementation/experiments/verify_qwen3_single_gpu.py \
  --model ~/code/model/Qwen3-0.6b --chunks 4 --backend eager
```

脚本加载本地 HF Qwen3-0.6B，只保留并验证第一层。Column-parallel 投影使用输出维权重分片，row-parallel 投影使用输入维权重分片并 all-reduce。chunk 一律沿 token/M 维切分。默认在 CUDA 上以 FP32 执行，比较替换前后的 HF 层输出。

传入 `--backend cuda_graph` 可运行静态形状 CUDA Graph 路径。单卡 `tp_size=1` 只能验证数值一致性。多 rank row-parallel 实现会调用 all-reduce，但尚未经过真实多 GPU 测试；本脚本也不测通信 overlap 或性能。

## 文件

| 路径 | 用途 |
|---|---|
| `core/tp_linear.py` | 共享 TP 语义、token chunking、双缓冲和 eager/CUDA Graph 后端 |
| `experiments/verify_qwen3_single_gpu.py` | 与本地 HF Qwen3 layer 做单卡数值对照 |
| `core/tp_overlap_cudagraph.py` | 共享 `TPLinear` 的 CUDA Graph 兼容入口 |
| `core/qwen3_layer_cudagraph.py` | 自定义 Qwen3 decoder layer 原型，未与 HF 全层对齐验证 |
| `experiments/grid_search_per_operator.py` | 旧版 per-operator grid search |
| `experiments/benchmark_full_layer.py` | 旧版整层 benchmark |
| `experiments/verify_correctness.py` | 旧版多 GPU correctness 脚本 |
| `experiments/EXPERIMENT_SUMMARY.md` | 历史实验记录，结果不代表当前实现 |

旧原型的速度数据、chunk 最优值与 token 阈值依赖其当时的算子分配和计时路径，尚未在当前实现中复现。多 GPU benchmark 前应先验证标准 Qwen3 TP 的算子布局、权重分片、通信和输出正确性。

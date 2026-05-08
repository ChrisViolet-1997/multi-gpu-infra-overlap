# Multi-GPU Infrastructure: Tensor Parallel Computation-Communication Overlap

这个项目实现了 Tensor Parallel (TP) 场景下的计算-通信 overlap 优化，通过 chunking 和 double buffering 技术降低端到端延迟。

## 项目结构

```
multi-gpu-infra-overlap/
├── base_implementation/          # 基础实现
│   ├── tp_overlap_poc.py        # 基础 PoC
│   ├── tp_overlap_double_buffer.py  # Double buffer 实现
│   ├── test_correctness.py      # 正确性验证
│   └── README_BASE.md           # 基础实现说明
│
├── parameter_grid_search/        # 任务1: 参数 Grid Search
│   ├── advanced_benchmark.py    # 高级 benchmark 工具
│   ├── run_advanced_benchmark.py  # 执行脚本
│   ├── OPTIMIZATION_EXPERIMENTS.md  # 实验报告
│   └── README.md                # 任务1说明
│
└── qwen3_integration/            # 任务2: Qwen3 集成
    ├── qwen3_layer.py           # Qwen3 decoder layer
    ├── adaptive_chunk_selector.py  # 自适应 chunk 选择
    ├── grid_search_qwen3_chunks.py  # Grid search 工具
    ├── compare_qwen3_configs.py  # 配置对比
    ├── optimal_qwen3_chunks.py  # 最优配置
    ├── QWEN3_PER_OPERATOR_OPTIMIZATION.md  # 实验报告
    └── README.md                # 任务2说明
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+ with CUDA
- 至少 2 个 GPU (支持 NCCL)
- NCCL 2.0+

### 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 运行示例

#### 1. 基础 Double Buffer Overlap

```bash
cd base_implementation
torchrun --nproc_per_node=2 tp_overlap_double_buffer.py
```

#### 2. 参数 Grid Search (任务1)

```bash
cd parameter_grid_search
torchrun --nproc_per_node=2 run_advanced_benchmark.py \
  --in_features 6144 \
  --out_features 12288 \
  --num_chunks 8 \
  --batch_size 1 \
  --seq_len 2048
```

#### 3. Qwen3 集成 (任务2)

```bash
cd qwen3_integration

# Grid search 找最优配置
BATCH_SIZE=512 SEQ_LEN=1 torchrun --nproc_per_node=2 grid_search_qwen3_chunks.py

# 对比不同配置
BATCH_SIZE=512 SEQ_LEN=1 torchrun --nproc_per_node=2 compare_qwen3_configs.py
```

## 核心技术

### 1. Double Buffering

使用两个输出缓冲区交替使用，消除数据竞争：
- Chunk i 写入 buffer A，同时 chunk i-1 的通信从 buffer B 读取
- 实现真正的计算-通信 overlap

### 2. 分离 CUDA Streams

- 计算和通信使用不同的 CUDA stream
- 通过 Event 精确控制依赖关系

### 3. Per-Operator Chunk Tuning

- 不同算子需要不同的 chunk 数量
- 基于输出维度和 Comp/Comm 比例自适应选择

## 主要成果

### 任务1: 参数优化 (Prefill Phase)

**测试场景**: batch_size=1, seq_len=2048

**最优配置**:
- Input Features: 6144
- Output Features: 12288
- Num Chunks: 8

**性能提升**:
- Latency: 35.596 ms
- Speedup: **1.44x**
- Overlap: 30.6%

**关键发现**:
- Comp/Comm Ratio ≈ 1.0 时 overlap 效果最好
- Chunk 数量需要平衡 overlap 粒度和调度开销
- 8 chunks 是最优选择（chunk_size=256 tokens）

### 任务2: Qwen3 集成

#### Prefill Phase (seq_len=2048)

**最优配置**: Per-operator tuning
```python
q_proj: 8, k_proj: 4, v_proj: 4, o_proj: 8,
gate_proj: 16, up_proj: 16, down_proj: 4
```

**性能**: +2.8% vs Fixed 4 chunks

#### Decode Phase - Batch Size 影响

| Batch Size | Total Tokens | 最优策略 | 加速 |
|------------|--------------|----------|------|
| 1-64       | 1-64         | No Overlap | Baseline |
| 512        | 512          | Per-operator | **+16.6%** |

**关键发现**:
- **Total tokens < 64**: 不要使用 overlap（性能下降 47-149%）
- **Total tokens ≥ 512**: Per-operator tuning 非常有效（+16.6%）
- **输出维度决定最优 chunk 数**: 大输出需要更多 chunks

## 实际应用建议

### 根据场景选择策略

```python
def select_strategy(batch_size: int, seq_len: int):
    total_tokens = batch_size * seq_len

    if total_tokens <= 64:
        # Decode with small batch: No Overlap
        return "chunk=1 for all operators"
    elif total_tokens <= 512:
        # Decode with medium batch: Minimal chunking
        return "chunk=2-4"
    elif total_tokens >= 1024:
        # Prefill or large batch decode: Per-operator tuning
        return "Per-operator optimized chunks"
```

### 性能预期

| 场景 | 配置 | 预期加速 |
|------|------|----------|
| Prefill (seq_len=2048) | Per-operator | +2.8% |
| Decode (batch≤64) | No Overlap | Baseline (避免下降) |
| Decode (batch=512) | Per-operator | +16.6% |

## 优化原则

1. **Comp/Comm 平衡**: 目标 Comp/Comm Ratio ≈ 1.0
2. **Total tokens 是关键**: tokens < 64 时不要用 overlap
3. **输出维度决定 chunks**: 大输出需要更多 chunks
4. **Chunk size 为 2^n**: GPU 内存对齐优化
5. **避免过小 chunk**: chunk_size < 128 调度开销过大

## 性能指标

### Overlap 指标

- **Speedup**: (Comp + Comm) / Total_Latency
- **Overlap Percentage**: (Comp + Comm - Total) / (Comp + Comm) × 100%
- **Overlap Ratio**: (Comp + Comm - Total) / min(Comp, Comm)

### 最佳实践

- Prefill: Speedup 1.3-1.5x
- Decode (large batch): Speedup 1.15-1.20x
- Comp/Comm Ratio: 0.8-1.2 (最佳范围)

## 参考文献

- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
- NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/
- Megatron-LM Tensor Parallel: https://github.com/NVIDIA/Megatron-LM

## 实验环境

- **硬件**: 2x GPU with NVLink
- **框架**: PyTorch with NCCL backend
- **模型**: Qwen3-8B
- **实验日期**: 2026-05-08

## 许可证

本项目仅用于研究和学习目的。

## 联系方式

如有问题或建议，请提交 Issue。

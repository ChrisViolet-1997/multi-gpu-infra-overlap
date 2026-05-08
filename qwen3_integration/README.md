# Qwen3 Integration (Task 2)

这个目录包含第二个任务的内容：将 overlap 优化应用到 Qwen3-8B 模型，为不同算子设计自适应的 chunk 数量策略。

## 任务目标

1. 分析 Qwen3-8B 模型的各个算子特性
2. 为每个算子找到最优的 chunk 数量
3. 实现完整的 Qwen3 decoder layer
4. 验证在不同 batch size 下的性能

## 文件说明

### 核心实现

- **`qwen3_layer.py`**: 完整的 Qwen3 Decoder Layer 实现
  - `RMSNorm`: RMS Layer Normalization
  - `Qwen3Attention`: 带 GQA 的 Attention 模块
  - `Qwen3MLP`: 带 SwiGLU 的 MLP 模块
  - `Qwen3DecoderLayer`: 完整的 decoder layer

- **`adaptive_chunk_selector.py`**: 自适应 chunk 选择算法
  - `AdaptiveChunkSelector`: 基于算子维度的自适应选择
  - `Qwen3ChunkConfig`: Qwen3 的 chunk 配置类

### Grid Search 工具

- **`grid_search_qwen3_chunks.py`**: 为每个算子 grid search 最优 chunk
  - 测试 chunk=[1, 2, 4, 8, 16]
  - 每个配置重复 5 次取平均（减少抖动）
  - 自动保存最优配置到 `optimal_qwen3_chunks.py`

- **`compare_qwen3_configs.py`**: 对比不同配置的性能
  - Fixed 1/2/4/8/16 chunks
  - Optimal per-operator chunks
  - 完整的性能分析

### 配置文件

- **`optimal_qwen3_chunks.py`**: Grid search 生成的最优配置
  - 根据 batch size 和 seq_len 自动生成
  - 包含每个算子的最优 chunk 数量

### 实验报告

- **`QWEN3_PER_OPERATOR_OPTIMIZATION.md`**: 完整实验报告
  - Prefill 阶段优化结果
  - Decode 阶段 batch size 影响分析
  - 最终推荐配置

## 关键发现

### 1. Prefill Phase (batch_size=1, seq_len=2048)

**最优配置**:
```python
optimal_chunks = {
    'q_proj': 8,
    'k_proj': 4,
    'v_proj': 4,
    'o_proj': 8,
    'gate_proj': 16,
    'up_proj': 16,
    'down_proj': 4,
}
```

**性能**:
- Optimal per-operator: 104.676 ms (BEST)
- Fixed 4 chunks: 107.614 ms
- **Speedup: +2.8%**

### 2. Decode Phase - Batch Size 影响

#### Batch Size = 1 (Total Tokens = 1)

**最优配置**: Fixed 16 chunks 或 Per-operator tuning
- Fixed 16 chunks: 3.618 ms (BEST)
- Optimal per-operator: 3.623 ms (-0.1%)
- No Overlap: 3.709 ms (-2.5%)

**结论**: Overlap 有小幅帮助 (~2.5%)

#### Batch Size = 4-64 (Total Tokens = 4-64)

**最优配置**: No Overlap (chunk=1)
- Optimal (all chunk=1): 3.798 ms (BEST)
- Fixed 2 chunks: 5.584 ms (-47%)
- Fixed 4 chunks: 9.450 ms (-149%)

**结论**: Chunking 是灾难，不要使用 overlap！

#### Batch Size = 512 (Total Tokens = 512)

**最优配置**: Per-operator tuning
```python
optimal_chunks = {
    'q_proj': 4,
    'k_proj': 1,
    'v_proj': 1,
    'o_proj': 4,
    'gate_proj': 8,
    'up_proj': 8,
    'down_proj': 1,
}
```

**性能**:
- Optimal per-operator: 27.876 ms (BEST)
- No Overlap: 32.509 ms
- **Speedup: +16.6%**

**结论**: Per-operator tuning 非常有效！

### 3. 核心规律

**Total Tokens 是关键因素**:

| Total Tokens | 最优策略 | 预期加速 |
|--------------|----------|----------|
| ≤ 64         | No Overlap (chunk=1) | Baseline (避免性能下降) |
| 256-512      | Moderate chunking (2-4) | ~5-10% |
| ≥ 512        | Per-operator tuning | ~16% |
| ≥ 1024 (Prefill) | Per-operator tuning | ~3% |

**算子特性**:
- 小输出算子 (k_proj, v_proj): 少 chunks 或 no overlap
- 中等输出算子 (q_proj, o_proj): 4-8 chunks
- 大输出算子 (gate_proj, up_proj): 8-16 chunks

## 使用方法

### 1. Grid Search 找最优配置

```bash
# Prefill phase
BATCH_SIZE=1 SEQ_LEN=2048 torchrun --nproc_per_node=2 grid_search_qwen3_chunks.py

# Decode phase with large batch
BATCH_SIZE=512 SEQ_LEN=1 torchrun --nproc_per_node=2 grid_search_qwen3_chunks.py
```

### 2. 对比不同配置

```bash
# Prefill
BATCH_SIZE=1 SEQ_LEN=2048 torchrun --nproc_per_node=2 compare_qwen3_configs.py

# Decode with large batch
BATCH_SIZE=512 SEQ_LEN=1 torchrun --nproc_per_node=2 compare_qwen3_configs.py
```

### 3. 使用最优配置

```python
from optimal_qwen3_chunks import optimal_chunks
from qwen3_layer import Qwen3DecoderLayer
from adaptive_chunk_selector import Qwen3ChunkConfig

# 创建配置
config = Qwen3ChunkConfig(
    q_proj_chunks=optimal_chunks['q_proj'],
    k_proj_chunks=optimal_chunks['k_proj'],
    v_proj_chunks=optimal_chunks['v_proj'],
    o_proj_chunks=optimal_chunks['o_proj'],
    gate_proj_chunks=optimal_chunks['gate_proj'],
    up_proj_chunks=optimal_chunks['up_proj'],
    down_proj_chunks=optimal_chunks['down_proj'],
)

# 创建 layer
layer = Qwen3DecoderLayer(
    hidden_size=4096,
    intermediate_size=12288,
    chunk_config=config,
    device="cuda",
)
```

## 实际应用建议

### 根据场景选择策略

```python
def select_chunk_config(batch_size: int, seq_len: int):
    total_tokens = batch_size * seq_len

    if total_tokens <= 64:
        # Small: No Overlap
        return Qwen3ChunkConfig(
            q_proj_chunks=1, k_proj_chunks=1, v_proj_chunks=1,
            o_proj_chunks=1, gate_proj_chunks=1, up_proj_chunks=1,
            down_proj_chunks=1,
        )
    elif total_tokens <= 512:
        # Medium: Minimal chunking
        return Qwen3ChunkConfig(
            q_proj_chunks=2, k_proj_chunks=2, v_proj_chunks=2,
            o_proj_chunks=2, gate_proj_chunks=4, up_proj_chunks=4,
            down_proj_chunks=2,
        )
    elif total_tokens <= 1024:
        # Large: Moderate chunking
        return Qwen3ChunkConfig(
            q_proj_chunks=4, k_proj_chunks=4, v_proj_chunks=4,
            o_proj_chunks=4, gate_proj_chunks=8, up_proj_chunks=8,
            down_proj_chunks=4,
        )
    else:
        # Prefill: Per-operator tuning
        return Qwen3ChunkConfig(
            q_proj_chunks=8, k_proj_chunks=4, v_proj_chunks=4,
            o_proj_chunks=8, gate_proj_chunks=16, up_proj_chunks=16,
            down_proj_chunks=4,
        )
```

### 性能预期

| 场景 | Batch Size | Seq Len | 推荐配置 | 预期加速 |
|------|------------|---------|----------|----------|
| Decode (小batch) | 1-64 | 1 | No Overlap | Baseline |
| Decode (大batch) | 512+ | 1 | Per-operator | +16.6% |
| Prefill | 1 | 2048+ | Per-operator | +2.8% |

## 总结

1. **Total tokens 是决定性因素**: tokens < 64 时不要用 overlap
2. **Per-operator tuning 在大 batch 时非常有效**: batch=512 时加速 16.6%
3. **输出维度决定最优 chunk 数**: 大输出需要更多 chunks
4. **实际应用需要根据场景动态选择策略**

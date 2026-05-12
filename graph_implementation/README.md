# CUDAGraph Implementation

本目录包含基于 CUDAGraph 的 Tensor Parallel 计算-通信 overlap 优化实现。

## 目录结构

```
graph_implementation/
├── core/                           # 核心实现代码
│   ├── tp_overlap_cudagraph.py    # CUDAGraph 版本的 TP overlap linear layer
│   └── qwen3_layer_cudagraph.py   # CUDAGraph 版本的 Qwen3 decoder layer
│
└── experiments/                    # 实验和测试代码
    ├── benchmark_cudagraph.py                  # 单 operator benchmark
    ├── final_benchmark.py                      # 完整 layer benchmark
    ├── test_nccl_cudagraph.py                 # NCCL 在 graph 中的测试
    ├── test_cudagraph_distributed.py          # 分布式测试
    ├── test_full_layer_graph.py               # 完整 layer graph 测试
    ├── profile_decode.py                       # Decode 阶段 profiling
    ├── profile_lightweight.py                  # 轻量级 profiling
    ├── show_bs32_chunk4_execution.py          # BS=32 Chunk=4 分析
    ├── test_bs64.py                           # BS=64 测试
    ├── true_per_operator_grid_search.py       # Per-operator grid search
    ├── rigorous_comparison.py                 # BS=128 严格对比
    ├── grid_search_bs256.py                   # BS=256 grid search
    ├── rigorous_comparison_bs256.py           # BS=256 严格对比
    ├── run_profile.sh                         # Profiling 脚本
    └── compare_chunks_profile.sh              # Chunk 对比脚本
```

## 核心代码说明

### 1. tp_overlap_cudagraph.py

CUDAGraph 优化的 TP overlap linear layer 实现。

**关键特性**:
- 整个 forward pass 录制为单个 CUDAGraph
- 消除 kernel launch overhead
- 支持 per-operator chunk 配置
- 预分配静态 buffer

**使用示例**:
```python
from tp_overlap_cudagraph import CUDAGraphDoubleBufferOverlapRowParallelLinear

layer = CUDAGraphDoubleBufferOverlapRowParallelLinear(
    in_features=4096,
    out_features=4096,
    num_chunks=2,
    static_input_shape=(256, 1, 4096),  # 固定 shape
    enable_graph=True,
    device="cuda",
)

# 第��次调用会自动捕获 graph
output = layer(input)  # Captures graph

# 后续调用直接 replay graph (极快)
output = layer(input)  # Replays graph
```

### 2. qwen3_layer_cudagraph.py

完整的 Qwen3 decoder layer CUDAGraph 实现。

**关键特性**:
- 整个 layer (7 个 operators + attention + MLP) 录制为一个 graph
- 包含 view/transpose/attention 等所有操作
- Layer-level graph capture

**使用示例**:
```python
from qwen3_layer_cudagraph import CUDAGraphQwen3DecoderLayer
from adaptive_chunk_selector import Qwen3ChunkConfig

# 配置 per-operator chunks
chunk_config = Qwen3ChunkConfig(
    q_proj_chunks=2, k_proj_chunks=2, v_proj_chunks=2,
    o_proj_chunks=1, gate_proj_chunks=2, up_proj_chunks=2,
    down_proj_chunks=1,
)

layer = CUDAGraphQwen3DecoderLayer(
    hidden_size=4096,
    intermediate_size=12288,
    chunk_config=chunk_config,
    static_input_shape=(256, 1, 4096),
    device="cuda",
)

output = layer(input)
```

## 性能结果

### 单 Operator (DoubleBufferOverlapRowParallelLinear)

| 配置 | 原始 | CUDAGraph | 加速比 |
|------|------|-----------|--------|
| BS=32, Chunk=1 | 0.515 ms | 0.152 ms | **3.38x** |
| BS=32, Chunk=4 | 1.567 ms | 0.197 ms | **7.96x** |
| BS=512, Chunk=4 | 3.560 ms | 1.634 ms | **2.18x** |

### 完整 Qwen3DecoderLayer

| 配置 | 原始 | CUDAGraph | 加速比 |
|------|------|-----------|--------|
| BS=32, Chunk=1 | 6.029 ms | 1.695 ms | **3.56x** |
| BS=32, Chunk=4 | 11.171 ms | 1.997 ms | **5.59x** |
| BS=128, Chunk=1 | ~8.3 ms | 4.217 ms | **~2.0x** |
| BS=256, All=1 | ~16.6 ms | 8.316 ms | **~2.0x** |
| BS=256, [2,2,2,1,2,2,1] | ~16.3 ms | 8.161 ms | **~2.0x** |

### Per-Operator Tuning 收益

| Batch Size | Baseline (all=1) | Optimal Config | 提升 |
|------------|------------------|----------------|------|
| BS=32 | 1.695 ms | 1.997 ms (chunk=4) | **-17.8%** ❌ |
| BS=64 | 2.701 ms | 2.974 ms (chunk=4) | **-10.1%** ❌ |
| BS=128 | 4.217 ms | 4.242 ms (q=2) | **-0.6%** ❌ |
| BS=256 | 8.316 ms | 8.161 ms [2,2,2,1,2,2,1] | **+1.87%** ✅ |

## 最优配置策略

基于严格的 grid search 和统计测试：

```python
def select_optimal_chunks_cudagraph(batch_size):
    """
    CUDAGraph 模式下的最优 chunk 配置。

    Returns: [q, k, v, o, gate, up, down]
    """
    if batch_size < 256:
        # 小 batch: 统一用 chunk=1
        return [1, 1, 1, 1, 1, 1, 1]
    else:
        # 大 batch: Per-operator tuning
        # q/k/v/gate/up 用 chunk=2
        # o/down 用 chunk=1
        return [2, 2, 2, 1, 2, 2, 1]
```

**关键发现**:
1. **BS < 256**: CUDAGraph 已经消除了大部分 overhead，chunking 反而增加调度开销
2. **BS >= 256**: 计算量足够大，chunk=2 的 overlap 收益开始超过调度开销
3. **Chunk=4/8/16 都太大**: 即使在大 batch 下也不如 chunk=2

## 技术要点

### 1. 整层 Graph Capture

关键突破是把 **Linear + View + Transpose + Attention** 整个 forward pass 录进一个大 graph：

```python
with torch.cuda.graph(layer_graph):
    # 7 个 linear operators
    query = q_proj(x)
    key = k_proj(x)
    value = v_proj(x)

    # View/Transpose (静态 shape，可以录进 graph)
    query = query.view(...).transpose(...)

    # Attention 计算
    attn_output = matmul(query, key.T) @ value

    # 其他 operators
    output = o_proj(attn_output)
    # ...
```

### 2. 静态 Shape 要求

- 必须在初始化时指定 `static_input_shape`
- 不同 batch size 需要不同的 graph
- Decode 场景 (seq_len=1) 完全满足要求

### 3. 禁用子模块 Graph

避免嵌套 graph capture：

```python
# ❌ 错误: 子模块各自 capture graph
q_proj = Linear(..., enable_graph=True)  # 会在 layer capture 时失败

# ✅ 正确: 禁用子模块 graph，整层 capture
q_proj = Linear(..., enable_graph=False)
layer_graph.capture(...)  # 整层作为一个 graph
```

## 使用建议

### 生产环境

```python
# 为常用 batch size 预先创建 graph cache
graph_cache = {
    32: create_layer(batch_size=32, chunks=[1,1,1,1,1,1,1]),
    64: create_layer(batch_size=64, chunks=[1,1,1,1,1,1,1]),
    128: create_layer(batch_size=128, chunks=[1,1,1,1,1,1,1]),
    256: create_layer(batch_size=256, chunks=[2,2,2,1,2,2,1]),
    512: create_layer(batch_size=512, chunks=[2,2,2,1,2,2,1]),
}

# 运行时选择对应的 graph
layer = graph_cache[batch_size]
output = layer(input)
```

### 开发/调试

- 使用 `enable_graph=False` 禁用 graph，方便调试
- Graph capture 失败时会自动 fallback 到 eager mode
- 第一次运行会有 graph capture 开销，后续运行极快

## 限制

1. **不能动态调整 chunk**: Graph 一旦 capture 就固定了
2. **需要预先知道 batch size**: 不同 batch size 需要不同 graph
3. **内存开销**: 每个 graph 会占用额外显存
4. **首次运行慢**: Graph capture 需要时间

## 实验文件说明

详见 `experiments/` 目录下的各个脚本，包含：
- 性能 benchmark
- Grid search 实验
- 统计显著性测试
- Profiling 工具

## 参考

- 原始实现: `base_implementation/tp_overlap_double_buffer.py`
- Qwen3 集成: `qwen3_integration/qwen3_layer.py`
- 项目总结: `PROJECT_SUMMARY.md`

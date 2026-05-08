# Parameter Grid Search (Task 1)

这个目录包含第一个任务的内容：通过 grid search 寻找最优的 input features 和 chunk 数量配置。

## 任务目标

在 Tensor Parallel (TP) 场景下，通过 chunking 和 double buffering 技术实现计算与通信的重叠，找到：
1. 最优的 input features 大小
2. 最优的 chunk 数量

## 文件说明

### 核心工具

- **`advanced_benchmark.py`**: 高级 benchmark 工具
  - `measure_overlap_metrics()`: 测量 overlap 指标
  - `measure_separated_compute_comm()`: 分离测量计算和通信时间
  - `grid_search_parameters()`: Grid search 功能
  - `grid_search_in_out_features()`: Input/Output features grid search

- **`run_advanced_benchmark.py`**: Benchmark 执行脚本
  - 命令行接口
  - 支持单配置和 grid search 模式

### 实验结果

- **`OPTIMIZATION_EXPERIMENTS.md`**: 完整实验报告
  - 实验一：寻找最优 Input Features
  - 实验二：验证更多 Chunks 的效果
  - 实验三：寻找最优 Chunk 数量
  - 最终推荐配置和优化原则

### 可视化

- **`in_out_features_heatmap.png`**: Input/Output features 热力图
- **`overlap_heatmap.png`**: Overlap 效果热力图

## 关键发现

### 最优配置 (Prefill: batch_size=1, seq_len=2048)

```python
input_features = 6144
output_features = 12288
num_chunks = 8
```

**性能指标**:
- Total Latency: 35.596 ms
- Speedup: 1.44x
- Overlap Percentage: 30.6%
- Comp/Comm Ratio: 0.87 (接近理想的 1:1)

### 优化原则

1. **Comp/Comm Ratio ≈ 1.0 时 overlap 效果最好**
   - Ratio < 1: 计算太快，无法完全覆盖通信
   - Ratio ≈ 1: 理想状态，最大化 overlap
   - Ratio > 1: 计算太慢，成为瓶颈

2. **Chunk 数量选择**
   - 不是越多越好
   - 需要在 overlap 粒度和调度开销之间平衡
   - 优先选择 2 的幂次（GPU 内存对齐）
   - 避免 chunk_size < 128（调度开销过大）

3. **参数调优流程**
   - 固定 chunk 数量，扫描 input_features
   - 固定 input_features，扫描 chunk 数量
   - 验证最优配置

## 使用方法

### 单配置测试

```bash
torchrun --nproc_per_node=2 run_advanced_benchmark.py \
  --in_features 6144 \
  --out_features 12288 \
  --num_chunks 8 \
  --batch_size 1 \
  --seq_len 2048
```

### Grid Search

```bash
# 测试不同 input features
for in_feat in 4096 6144 8192; do
  torchrun --nproc_per_node=2 run_advanced_benchmark.py \
    --in_features $in_feat \
    --out_features 12288 \
    --num_chunks 4
done

# 测试不同 chunk 数量
for chunks in 4 8 12 16; do
  torchrun --nproc_per_node=2 run_advanced_benchmark.py \
    --in_features 6144 \
    --out_features 12288 \
    --num_chunks $chunks
done
```

## 实验结果总结

### Input Features 对比 (num_chunks=4)

| Input Features | Comp/Comm Ratio | Speedup | Overlap % |
|----------------|-----------------|---------|-----------|
| 4096           | 0.58            | 1.21x   | 17.6%     |
| **6144**       | **0.88**        | **1.28x** | **22.1%** |
| 8192           | 1.17            | 1.24x   | 19.5%     |

### Chunk 数量对比 (input_features=6144)

| Num Chunks | Total Latency | Speedup | Overlap % |
|------------|---------------|---------|-----------|
| 4          | 39.647 ms     | 1.28x   | 22.1%     |
| **8**      | **35.596 ms** | **1.44x** | **30.6%** |
| 12         | 39.384 ms     | 1.45x   | 30.9%     |
| 16         | 36.405 ms     | 1.38x   | 27.6%     |

**结论**: input_features=6144, num_chunks=8 是最优配置

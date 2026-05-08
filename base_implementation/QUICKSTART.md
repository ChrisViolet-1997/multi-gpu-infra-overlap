# 快速开始指南

## 5分钟上手

### 1. 检查环境

```bash
python check_env.py
```

应该看到：
- ✓ PyTorch installed
- ✓ CUDA available
- ✓ Distributed available
- ✓ Found X GPUs

### 2. 运行Benchmark

```bash
./run_benchmark.sh
```

预期输出：
```
================================================================================
DOUBLE BUFFER OVERLAP BENCHMARK
================================================================================
Configuration:
  - World Size: 2 GPUs
  - Batch Size: 1
  - Sequence Length: 2048
  ...

[1/3] Benchmarking Baseline (No Overlap)...
[2/3] Benchmarking Original Overlap (with wait_event)...
[3/3] Benchmarking Double Buffer Overlap...

================================================================================
RESULTS
================================================================================
Baseline Latency:              40.689 ms
Original Overlap Latency:      41.972 ms
Double Buffer Overlap Latency: 34.283 ms

Original Overlap Speedup:      0.97x
Double Buffer Speedup:         1.19x
================================================================================

✓ Double Buffer is 22.4% faster than Original Overlap!
✓ SUCCESS: Double Buffer achieved 1.19x speedup!
```

### 3. 验证正确性

```bash
python test_correctness.py
```

应该看到：
```
Testing correctness of overlap implementations...
✓ All tests passed!
```

## 常见问题

### Q: 为什么加速比不高？

A: 可能的原因：
1. **问题规模太小**：通信时间 > 计算时间
   - 解决：增大batch_size或seq_len
   ```bash
   BATCH_SIZE=4 SEQ_LEN=4096 ./run_benchmark.sh
   ```

2. **Chunk数量不合适**：
   - 太少：overlap不充分
   - 太多：overhead太大
   - 解决：尝试不同值
   ```bash
   NUM_CHUNKS=8 ./run_benchmark.sh
   ```

3. **硬件限制**：PCIe带宽低于NVLink
   - 检查：`nvidia-smi topo -m`

### Q: 如何Profile分析？

A: 使用nvprof：

```bash
# 进入scripts目录
cd scripts

# 运行完整的profile（包括baseline、原始overlap、double buffer）
./profile_with_nvprof.sh

# 分析overlap
python analyze_double_buffer.py
```

或者只profile double buffer：

```bash
/usr/local/cuda/bin/nvprof --profile-child-processes --print-gpu-trace \
    torchrun --nproc_per_node=2 scripts/profile_double_buffer.py \
    2>&1 | tee profiles/my_profile.txt

# 查看GEMM和NCCL kernels
grep -E "volta_sgemm|ncclDevKernel" profiles/my_profile.txt | head -40
```

### Q: 如何在自己的模型中使用？

A: 替换标准的Linear层：

```python
from tp_overlap_double_buffer import DoubleBufferOverlapRowParallelLinear

# 原来的代码
# self.linear = RowParallelLinear(in_features, out_features)

# 替换为
self.linear = DoubleBufferOverlapRowParallelLinear(
    in_features=in_features,
    out_features=out_features,
    num_chunks=4,  # 根据实际情况调整
    process_group=your_tp_group,
    device="cuda"
)
```

## 测试不同配置

### 小规模测试
```bash
BATCH_SIZE=1 SEQ_LEN=1024 ./run_benchmark.sh
```

### 中等规模
```bash
BATCH_SIZE=2 SEQ_LEN=2048 ./run_benchmark.sh
```

### 大规模测试
```bash
BATCH_SIZE=8 SEQ_LEN=8192 OUT_FEATURES=24576 ./run_benchmark.sh
```

### 调整Chunk数量
```bash
# 测试不同chunk数量
for chunks in 2 4 8 16; do
    echo "Testing NUM_CHUNKS=$chunks"
    NUM_CHUNKS=$chunks ./run_benchmark.sh
done
```

## 下一步

- 阅读 [README.md](README.md) 了解实现原理
- 查看 [tp_overlap_double_buffer.py](tp_overlap_double_buffer.py) 源码
- 使用nvprof分析自己的配置

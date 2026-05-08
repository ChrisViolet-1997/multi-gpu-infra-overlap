# Tensor Parallel Computation-Communication Overlap with Double Buffering

高性能分布式训练中的计算-通信overlap优化实现，使用double buffering技术实现真正的并行执行。

## 📋 目录

- [问题背景](#问题背景)
- [解决方案](#解决方案)
- [性能对比](#性能对比)
- [快速开始](#快速开始)
- [实现原理](#实现原理)
- [Profile分析](#profile分析)
- [文件结构](#文件结构)

## 问题背景

在Tensor Parallel训练中，Row-Parallel层的标准实现是串行的：

```
[GEMM计算] → [AllReduce通信] → [GEMM计算] → [AllReduce通信] → ...
```

这导致GPU在通信时空闲，浪费计算资源。

### 初始Overlap尝试的问题

最初的overlap实现虽然使用了多stream和chunking，但由于**数据竞争保护**导致完全串行执行：

```python
# 原始代码中的wait导致串行
if chunk_idx > 0:
    self.compute_stream.wait_event(self.comm_events[chunk_idx - 1])
```

**nvprof验证结果**：
- Baseline: 40.7ms
- 原始Overlap: 42.0ms (更慢！)
- 原因：chunking开销 + 无真正overlap

## 解决方案

### Double Buffering

使用**两个output buffer**交替使用，消除数据竞争：

```
Buffer 0: Chunk 0, Chunk 2, Chunk 4, ...
Buffer 1: Chunk 1, Chunk 3, Chunk 5, ...
```

**关键优势**：
- Chunk i 写入 Buffer A
- Chunk i-1 通信读取 Buffer B
- 无数据竞争，可并行执行

### Timeline对比

**原始Overlap (串行)**：
```
Time: ----[GEMM]----[NCCL]----[GEMM]----[NCCL]----
      无overlap，完全串行
```

**Double Buffer (并行)**：
```
Compute Stream: ----[GEMM_0]----[GEMM_1]----[GEMM_2]----
Comm Stream:    --------[NCCL_0]----[NCCL_1]----[NCCL_2]
                        ^^^^^^^^ Overlap!
```

## 性能对比

### Benchmark结果

配置：`batch_size=1, seq_len=2048, in_features=4096, out_features=12288, num_chunks=4`

| 实现方式 | 延迟 (ms) | 加速比 | 说明 |
|---------|----------|--------|------|
| Baseline | 40.7 | 1.00x | 串行执行 |
| 原始Overlap | 42.0 | 0.97x | 无真正overlap，反而更慢 |
| **Double Buffer** | **34.3** | **1.19x** | 真正的overlap |

### nvprof验证

**原始Overlap**：完全串行
```
1081.22ms  3.88ms GEMM
1085.10ms  6.64ms NCCL  ← 等待GEMM完成
1091.77ms  3.87ms GEMM  ← 等待NCCL完成
```

**Double Buffer**：真正overlap
```
978.62ms  4.68ms GEMM
982.39ms  6.45ms NCCL  ← 与GEMM overlap 0.91ms
983.33ms  5.07ms GEMM  ← 与NCCL overlap 5.07ms!
```

发现 **6个overlap区间**，总overlap时间 **13.28ms**

## 快速开始

### 环境要求

- PyTorch with CUDA
- 至少2个GPU
- NCCL

### 运行Benchmark

```bash
# 基础测试
torchrun --nproc_per_node=2 tp_overlap_double_buffer.py

# 测试不同配置
BATCH_SIZE=4 SEQ_LEN=4096 torchrun --nproc_per_node=2 tp_overlap_double_buffer.py

# 调整chunk数量
NUM_CHUNKS=8 torchrun --nproc_per_node=2 tp_overlap_double_buffer.py
```

### 正确性验证

```bash
python test_correctness.py
```

## 实现原理

### 核心代码

```python
class DoubleBufferOverlapRowParallelLinear(nn.Module):
    def forward(self, x):
        # 分配两个buffer
        buffer_0 = torch.empty(...)
        buffer_1 = torch.empty(...)
        buffers = [buffer_0, buffer_1]

        for chunk_idx in range(num_chunks):
            # 交替使用buffer
            buffer_idx = chunk_idx % 2
            current_buffer = buffers[buffer_idx]

            # Compute: 写入当前buffer
            with torch.cuda.stream(self.compute_stream):
                # 移除了wait! 不同chunk用不同buffer
                torch.matmul(x_chunk, self.weight.t(), out=buffer_chunk)
                self.compute_events[chunk_idx].record()

            # Comm: 读取当前buffer
            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(self.compute_events[chunk_idx])
                dist.all_reduce(buffer_chunk, ...)
                output_chunk.copy_(buffer_chunk)
```

### 为什么需要Double Buffer？

**问题**：直接移除wait会导致数据竞争
- Chunk i-1 的AllReduce正在**读取** output buffer
- Chunk i 的GEMM会**写入**同一个buffer
- Race condition！

**解决**：Double buffer
- 不同chunk使用不同buffer
- 读写操作在不同内存区域
- 无数据竞争，安全并行

## Profile分析

### 使用nvprof分析

```bash
# 进入scripts目录
cd scripts

# 运行profile脚本（会profile三个版本）
./profile_with_nvprof.sh

# 分析overlap
python analyze_double_buffer.py
```

或者手动profile：

```bash
# Profile double buffer
/usr/local/cuda/bin/nvprof --profile-child-processes --print-gpu-trace \
    torchrun --nproc_per_node=2 scripts/profile_double_buffer.py \
    2>&1 | tee profiles/double_buffer_nvprof.txt

# 提取GEMM和NCCL kernels
grep -E "volta_sgemm|ncclDevKernel" profiles/double_buffer_nvprof.txt | head -40
```

### 关键指标

从nvprof输出可以看到：

1. **GEMM时间**: ~4-5ms per chunk
2. **NCCL时间**: ~6-7ms per chunk
3. **Overlap时间**: ~13ms total
4. **加速比**: 1.19x

### Timeline分析

使用analyze_double_buffer.py自动检测overlap：

```bash
cd scripts
python analyze_double_buffer.py
```

输出示例：
```
Found 6 overlapping kernel pairs:

1. GEMM [978.62-983.30ms]
   overlaps with
   NCCL [982.39-988.84ms]
   Overlap duration: 0.91ms

✓ SUCCESS: Found 6 overlaps!
  Total overlap time: 13.28ms
```

## 文件结构

```
.
├── README.md                          # 本文档
├── tp_overlap_poc.py                  # 原始overlap实现（有wait，无真正overlap）
├── tp_overlap_double_buffer.py        # Double buffer优化实现 ⭐
├── test_correctness.py                # 正确性测试
├── check_env.py                       # 环境检查
├── run_benchmark.sh                   # 快速benchmark脚本
│
├── scripts/                           # 辅助脚本
│   ├── profile_double_buffer.py       # Profile脚本
│   ├── analyze_double_buffer.py       # Timeline分析
│   ├── simple_profiler.py             # 简单性能测试
│   └── ...
│
├── profiles/                          # Profile输出
│   ├── baseline_nvprof.txt
│   ├── overlap_nvprof.txt
│   └── ...
│
└── archive/                           # 历史实验代码
    └── ...
```

## 进一步优化

### 1. 调整Chunk数量

```bash
# 测试不同chunk数量
for chunks in 2 4 8 16; do
    NUM_CHUNKS=$chunks torchrun --nproc_per_node=2 tp_overlap_double_buffer.py
done
```

**建议**：
- 小问题：num_chunks=2-4
- 大问题：num_chunks=8-16

### 2. 增大问题规模

Overlap收益随问题规模增加：

```bash
# 更大的batch和sequence length
BATCH_SIZE=8 SEQ_LEN=8192 torchrun --nproc_per_node=2 tp_overlap_double_buffer.py
```

### 3. 多级Pipeline

可以扩展到3-buffer或更复杂的pipeline结构。

## 关键收获

1. **诊断方法**：nvprof + timeline分析是发现overlap问题的关键
2. **数据竞争**：必须考虑读写冲突，不能简单移除同步
3. **Double Buffer**：经典的空间换时间策略，消除数据竞争
4. **性能验证**：实测加速19%，证明overlap有效

## 参考资料

- [Megatron-LM Tensor Parallel](https://github.com/NVIDIA/Megatron-LM)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [CUDA Streams and Events](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)

## License

MIT

## 贡献

欢迎提Issue和PR！

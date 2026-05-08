# Base Implementation

这个目录包含 Tensor Parallel 计算-通信 Overlap 的基础实现。

## 文件说明

### 核心实现

- **`tp_overlap_poc.py`**: 基础 PoC 实现
  - `BaselineRowParallelLinear`: 无 overlap 的基线实现
  - `OverlapRowParallelLinear`: 带 overlap 的初始实现
  - 基础 benchmark 工具

- **`tp_overlap_double_buffer.py`**: Double Buffer 优化实现
  - `DoubleBufferOverlapRowParallelLinear`: 使用双缓冲消除数据竞争
  - 实现真正的计算-通信 overlap
  - 解决了原始实现中的 wait_event 阻塞问题

### 测试工具

- **`test_correctness.py`**: 正确性验证
  - 验证 overlap 实现的数值正确性
  - 对比 baseline 和 overlap 版本的输出

- **`check_env.py`**: 环境检查
  - 检查 CUDA、PyTorch、NCCL 等依赖

### 文档

- **`README.md`**: 项目总体说明
- **`QUICKSTART.md`**: 快速开始指南

## 使用方法

### 运行基础 benchmark

```bash
torchrun --nproc_per_node=2 tp_overlap_double_buffer.py
```

### 验证正确性

```bash
torchrun --nproc_per_node=2 test_correctness.py
```

## 核心概念

### Double Buffering

使用两个输出缓冲区交替使用：
- Chunk i 写入 buffer A，同时 chunk i-1 的通信从 buffer B 读取
- 消除数据竞争，实现真正的 overlap

### 关键优化

1. **分离 CUDA Streams**: 计算和通信使用不同的 stream
2. **双缓冲**: 消除 wait_event 阻塞
3. **Event 同步**: 精确控制依赖关系

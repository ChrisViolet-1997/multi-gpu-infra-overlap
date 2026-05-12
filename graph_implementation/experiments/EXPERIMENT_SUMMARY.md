# TP Overlap CUDAGraph 实验总结

## 实验环境

- GPU: 2x GPU (TP=2)
- 模型: Qwen3-8B (hidden=4096, intermediate=12288, heads=32, kv_heads=8)
- 框架: PyTorch + NCCL
- 核心实现: `graph_implementation/core/tp_overlap_cudagraph.py`

## Bug 修复

### Bug 1: Overlap 模式 event 同步位置错误

**文件**: `tp_overlap_cudagraph.py` (graph forward path, line ~296)

**问题**: `compute_done.record()` 放在 `chunk_buffers[i].copy_(output_chunk_view)` 之前，导致 comm_stream 的 `wait_event` 只等了 graph replay 完成，没等 copy 完成。comm_stream 可能在 copy 未完成时就开始 all_reduce，造成数据竞争。

**现象**: 同一 layer、同一 input 两次 forward 结果不一致（max_diff ~3.8），非确定性行为。

**修复**: 将 `event.record()` 移到 `copy_` 之后。

### Bug 2: Eager 路径 non-contiguous tensor 传给 all_reduce

**文件**: `tp_overlap_cudagraph.py` (_forward_impl, eager overlap path)

**问题**: `output_flat[:, chunk_start:chunk_end]` 是非连续 view，直接传给 `dist.all_reduce` 会报 `ValueError: Tensors must be contiguous`。

**修复**: 先 `.contiguous()` 拷贝到独立 buffer，all_reduce 后再 copy 回原位置。同时用 event 保证同步正确性。

## 实验结果

### Task 1: 正确性验证

```
torchrun --nproc_per_node=2 graph_implementation/experiments/verify_correctness.py
```

所有 7 个 operator × 3 种 chunk (2, 4, 8) 全部通过，max_diff < 3.4e-5。

### Task 2: Per-Operator Grid Search

```bash
# Prefill
torchrun --nproc_per_node=2 grid_search_per_operator.py --batch_size 16 --seq_len 1024 --output configs/prefill_bs16_seq1024.json
# Decode
torchrun --nproc_per_node=2 grid_search_per_operator.py --batch_size 16 --seq_len 1 --output configs/decode_bs16_seq1.json
```

**Prefill (BS=16, SeqLen=1024, 16384 tokens)**:

| Operator  | in→out      | Best Chunk | Speedup |
|-----------|-------------|:----------:|--------:|
| q_proj    | 4096→4096   | 16         | 15.4%   |
| k_proj    | 4096→1024   | 8          | 16.0%   |
| v_proj    | 4096→1024   | 8          | 16.7%   |
| o_proj    | 4096→4096   | 16         | 15.4%   |
| gate_proj | 4096→12288  | 16         | 17.8%   |
| up_proj   | 4096→12288  | 16         | 17.8%   |
| down_proj | 12288→4096  | 16         | 30.9%   |

**Decode (BS=16, SeqLen=1, 16 tokens)**:

| Operator  | in→out      | Best Chunk | Speedup |
|-----------|-------------|:----------:|--------:|
| q_proj    | 4096→4096   | 1          | —       |
| k_proj    | 4096→1024   | 1          | —       |
| v_proj    | 4096→1024   | 1          | —       |
| o_proj    | 4096→4096   | 1          | —       |
| gate_proj | 4096→12288  | 1          | —       |
| up_proj   | 4096→12288  | 1          | —       |
| down_proj | 12288→4096  | 1          | —       |

Decode 阶段所有 operator chunk>1 均为负收益（-55% ~ -860%），最优全部为 chunk=1。

### Task 3: Full Layer Benchmark

```bash
# Prefill
torchrun --nproc_per_node=2 benchmark_full_layer.py --batch_size 16 --seq_len 1024 --config configs/prefill_bs16_seq1024.json
# Decode
torchrun --nproc_per_node=2 benchmark_full_layer.py --batch_size 16 --seq_len 1 --config configs/decode_bs16_seq1.json
```

**Prefill (BS=16, SeqLen=1024, 16384 tokens)**:

| Config              | Latency (ms) | Speedup |
|---------------------|:------------:|--------:|
| baseline (chunk=1)  | 740.2        | —       |
| static chunk=2      | 710.5        | +4.0%   |
| static chunk=4      | 654.7        | +11.6%  |
| static chunk=8      | 640.4        | +13.5%  |
| optimal (per-op)    | 629.0        | +15.0%  |

**Decode (BS=16, SeqLen=1, 16 tokens)**:

| Config              | Latency (ms) | Speedup  |
|---------------------|:------------:|---------:|
| baseline (chunk=1)  | 3.33         | —        |
| static chunk=2      | 5.37         | -61.2%   |
| static chunk=4      | 9.24         | -177.3%  |
| static chunk=8      | 16.61        | -398.8%  |
| optimal (per-op)    | 3.28         | +1.7%    |

Decode 场景下 optimal config（全部 chunk=1）与 baseline 一致，无额外开销。

## 关键结论

1. **Prefill 阶段**: Overlap 有效，最优 per-operator 配置带来 15% 端到端加速
2. **Decode 阶段**: Overlap 完全无效，chunk>1 严重劣化。原因是 token 数太少，compute 时间极短，chunk 分割的固定开销（event/copy/launch）远超 overlap 收益
3. **实际部署策略**: 需要根据 token 数动态选择 chunk config — prefill 用 optimal config，decode 用 chunk=1
4. **down_proj 收益最大** (30.9%): in_features=12288 使得单次 compute 时间长，overlap 空间充足
5. **k_proj/v_proj 在 chunk=8 最优**: output 较小 (1024)，chunk=16 时每个 chunk 只有 64 列，overhead 反超收益

## 文件结构

```
graph_implementation/
├── core/
│   ├── tp_overlap_cudagraph.py      # 核心实现 (含 bug fix)
│   └── qwen3_layer_cudagraph.py     # Full layer 封装
└── experiments/
    ├── verify_correctness.py        # Task 1: 正确性验证
    ├── grid_search_per_operator.py  # Task 2: 最优 chunk 搜索 (支持 --batch_size --seq_len --output)
    ├── benchmark_full_layer.py      # Task 3: 端到端 benchmark (支持 --batch_size --seq_len --config)
    ├── configs/
    │   ├── prefill_bs16_seq1024.json  # Prefill 最优配置
    │   └── decode_bs16_seq1.json      # Decode 最优配置
    └── EXPERIMENT_SUMMARY.md        # 本文档
```

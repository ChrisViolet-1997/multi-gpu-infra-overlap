# Base Implementation

`tp_overlap_double_buffer.py` 保留早期 Python 入口和类名，实际计算委托给 `graph_implementation/core/tp_linear.py` 的共享 `TPLinear`，并选择 eager 后端。

```python
from tp_overlap_double_buffer import DoubleBufferOverlapRowParallelLinear

layer = DoubleBufferOverlapRowParallelLinear(
    in_features=4096,
    out_features=4096,
    num_chunks=4,
    device="cuda",
)
```

默认是 row-parallel；可显式设 `parallel_mode="column"`。所有 chunk 沿展平后的 token/M 维划分。Row-parallel 会在 TP world size 大于 1 时对每个 token chunk 执行 all-reduce；分布式进程组已初始化时，未显式传 group 会使用默认 group。

当前可复现的参考正确性验证见仓库根目录 [README.md](../README.md)。早期本目录文档中的性能数字与 buffer 示意图对应旧实现，不代表共享实现的实测结果。

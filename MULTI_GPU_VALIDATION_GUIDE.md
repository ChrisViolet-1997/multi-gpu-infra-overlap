# 双卡验证工作指南

本文供后续 Codex 在双 GPU 机器上继续验证本项目时使用。目标是先证明 TP 分片和数值正确，再测 overlap；不得把单卡结果或旧报告当作多卡性能证据。

## 开始前

1. 先运行 `git status --short`，检查并保留现场已有改动。阅读根目录 `README.md`、本文件、`graph_implementation/core/tp_linear.py` 和单卡验证脚本。
2. 从当前机器探测 Python、PyTorch、CUDA、GPU 数量、设备名、Transformers 和 NCCL。不要假设虚拟环境路径、CUDA 版本或 GPU 型号与开发机相同。
3. 从用户给定路径或环境变量取得模型目录。验证 `config.json`、权重索引/权重文件可读，并从 config 读取模型维度、dtype、层数和 attention/head 配置。不要将开发机的 `~/code/model/...` 或旧的 `/root/autodl-tmp/...` 路径写进代码。
4. NCCL 初始化前，每个进程按 `LOCAL_RANK` 选择 CUDA device。先确认 `torch.cuda.device_count() >= 2`，并记录实际 TP process group 的 world size。

建议用显式环境配置运行命令，路径由目标机器维护者填写：

```bash
export CODEX_VALIDATION_PYTHON=/path/to/venv/bin/python
export QWEN3_MODEL_DIR=/path/to/Qwen3-model

"$CODEX_VALIDATION_PYTHON" -c 'import torch, transformers; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), transformers.__version__)'
nvidia-smi
```

所有新脚本都应提供 `--model`、`--dtype` 等参数，或读取清晰命名的环境变量；不要依赖某个用户的 shell 配置。导入仓库模块时使用相对脚本位置构造路径，不使用机器专属绝对路径。

## 当前实现边界

`TPLinear` 的预期布局：

- Column-parallel：按输出维切权重，每 rank 产出本地输出 shard，不做 all-reduce。
- Row-parallel：按输入维切权重，每 rank 计算部分输出，对每个 token chunk 做 all-reduce。
- `chunks` 切展平后的 token/M 维。双缓冲保护仍在通信中的 chunk；eager 和 CUDA Graph 后端共用 TP 语义。
- Row-parallel 的 `input_is_sharded=True` 表示调用方已经提供本 rank 的输入分片，不应再次切输入。

现有 `graph_implementation/experiments/verify_qwen3_single_gpu.py` 是 `tp_size=1` 的对照脚本。不要直接对它运行 `torchrun --nproc_per_node=2`：Qwen3 attention 需要每 rank 的本地 head/KV head 形状，o/down projection 接受本地输入 shard，完整层还必须匹配 RoPE、causal mask、norm、残差及输出布局。现有自定义 Qwen3 layer 尚不能替代 HF 层作为已验证的多卡参考。

## 验证顺序

### 1. 单算子 TP 数值验证

先新增或完善分布式验证脚本，例如 `graph_implementation/experiments/verify_tp_linear.py`：

- 以 `torchrun --standalone --nproc_per_node=2` 启动；所有 rank 使用相同 shape、dtype、seed、输入和参考权重。
- 从同一份完整 `nn.Linear` 权重分别构造 column/row shards。不要把 rank 0 的 local shard 原样 broadcast 到其他 rank；各 rank 的切片必须对应完整权重的不同区间。
- Column-parallel：验证每 rank 的局部输出等于完整参考输出相同列范围；必要时 gather 后再与完整 `F.linear` 对照。
- Row-parallel：用完整 replicated input 自动切 shard；另测 `input_is_sharded=True` 的本地输入路径。all-reduce 后与完整 `F.linear` 对照，bias 只加一次。
- 覆盖 `num_chunks=1,2,4`，以及 token 数不能被 chunk 数整除的输入；覆盖带 bias/无 bias、batch/seq 两种输入形状。
- eager 和 CUDA Graph 都测。Graph 使用固定 shape；至少调用两次，第一次包含 capture，后续调用验证 replay 和双缓冲复用。
- 所有 rank 都参与断言统计。将每 rank 的最大误差用 collective 汇总为全局最大误差；任一 rank 失败则整体失败，并确保失败路径仍销毁 process group。

优先测试 `tp_size=2`、小矩阵和短 token 数，先把权重切片、collective 顺序、dtype 和 shape 错误查清楚，再扩大到 Qwen3 实际维度。

### 2. Qwen3 decoder layer 对照

单算子通过后，再做多卡 HF layer 对照。每 rank 可加载本地完整 checkpoint，再按真实 Qwen3 TP 布局替换/分片投影权重；若显存不够，再实现 rank-0 checkpoint 加载与 shard broadcast。所有 rank 输入相同 token ids，hidden state 在 residual stream 上保持预期的 replicated 布局。

对照时从同一 checkpoint、同一输入和同一 dtype 计算 HF reference。参考层必须使用与目标版本 Transformers 一致的 RoPE、causal mask、Q/K norm、GQA、MLP、残差和最终张量布局。报告每个 rank 的输出 shape、全局 max absolute error、RMS error、dtype 和容差。无法与 HF 对齐时，先停在单算子验证，不要把随机权重自洽或单 rank 结果叫作模型正确性。

### 3. overlap 性能验证

只有 correctness 通过后才测性能：

- 对比相同 TP 分片、相同输入、相同 dtype、相同通信算法下的 `chunks=1` 与 `chunks>1`。关闭 overlap 的对照也必须调用相同次数/大小的 all-reduce。
- 先 warmup；CUDA Graph capture 时间单独记录且不计入 steady-state replay latency。分别报告 eager、Graph capture、Graph replay。
- latency 用 CUDA events 或可靠的设备同步测量，并以所有 rank 中最慢的耗时作为 step latency；重复多轮，报告 median 和离散度。
- 使用 Nsight Systems 或等效 timeline 检查 GEMM 与 NCCL 是否真实并发。仅看到两个 stream、异步 API 或较高的 `(compute + comm) / total` 计算值，不足以证明有效 overlap。
- 报告 GPU 型号、GPU 间互联、驱动/CUDA/PyTorch/NCCL 版本、TP size、shape、token 数、chunk 数、warmup/迭代次数及通信 bytes。
- 测试代表性的 prefill 与 decode 负载。不要仅用极大 batch 制造足够计算量，也不要从单一总 token 阈值推导通用策略。

## 运行检查与交付

运行前后记录 `nvidia-smi`；避免多个分布式 benchmark 同时争用同一对 GPU。每次只跑一个 `torchrun` job，并确认所有 rank 正常退出。完成后至少运行 Python 编译检查、目标 correctness 脚本和 `git diff --check`。

最终报告分开写清：

- 数值正确性是否通过，以及比较对象和误差。
- eager/Graph 的 correctness 与性能结果分别是什么。
- timeline 是否证明确实有 GEMM/NCCL 重叠。
- 未测试的形状、dtype、TP degree 和互联。

不要删除用户已有的实验数据、checkpoint、profiling 输出或未提交改动来清理测试现场。

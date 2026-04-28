# Tensor Parallel Computation-Communication Overlap PoC

High-performance demonstration of hiding all_reduce latency in Row-Parallel layers through chunked GEMM and multi-stream pipelining.

## Architecture

### Baseline (No Overlap)
```
[GEMM Compute] → [Wait] → [All-Reduce] → [Wait] → [Next Layer]
                  ↑________________↑
                  Communication Stall
```

### Optimized (With Overlap)
```
Chunk 0: [Compute] → [All-Reduce]
Chunk 1:              [Compute] → [All-Reduce]
Chunk 2:                           [Compute] → [All-Reduce]
Chunk 3:                                        [Compute] → [All-Reduce]
         ↑_________________________________________↑
              Computation hides communication
```

## Key Techniques

1. **Matrix Chunking**: Split GEMM along batch dimension into K chunks
2. **Dual Streams**:
   - `compute_stream`: Executes matrix multiplications
   - `comm_stream`: Executes all_reduce operations
3. **Event Synchronization**: CUDA events ensure correctness without blocking
4. **Pipelining**: Compute chunk i+1 while all_reduce chunk i

## Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

- PyTorch >= 2.0
- CUDA >= 11.8
- Multi-GPU system (minimum 2 GPUs)
- NCCL backend

## Usage

### Basic PoC Demo

Run on 2 GPUs:
```bash
torchrun --nproc_per_node=2 tp_overlap_poc.py
```

Run on 4 GPUs:
```bash
torchrun --nproc_per_node=4 tp_overlap_poc.py
```

### Qwen-32B Comparative Experiment

Run comprehensive experiment with Qwen-32B model dimensions:
```bash
# Quick launcher (auto-detects GPUs)
./run_qwen.sh 2

# Or use torchrun directly
torchrun --nproc_per_node=2 qwen_comparative_experiment.py
```

This experiment tests:
- **MLP Down-Projection**: [batch, seq_len, 13824] @ [5120, 13824]
- **Attention Out-Projection**: [batch, seq_len, 5120] @ [5120, 5120]
- **Sequence Lengths**: 512, 1024, 2048, 4096
- **Validation**: torch.allclose correctness check

Expected output table:
```
BatchSize    SeqLen     Baseline(ms)    Overlap(ms)     Speedup     Hidden%     Validation
1            512        2.345           1.876           20.0%       20.0%       ✓ PASS
1            1024       4.123           3.012           26.9%       26.9%       ✓ PASS
1            2048       7.891           5.432           31.2%       31.2%       ✓ PASS
1            4096       15.234          10.123          33.5%       33.5%       ✓ PASS
```

## Expected Output

```
================================================================================
TENSOR PARALLEL ROW-PARALLEL OVERLAP BENCHMARK
================================================================================
Configuration:
  - World Size: 4 GPUs
  - Batch Size: 8
  - Sequence Length: 2048
  - Input Features: 4096
  - Output Features: 4096
  - Num Chunks: 4
================================================================================

[1/2] Benchmarking Baseline (No Overlap)...
[2/2] Benchmarking Overlap (Pipelined)...

================================================================================
RESULTS
================================================================================
Baseline Latency:        12.456 ms
Overlap Latency:         8.234 ms
Speedup:                 1.51x
Hidden Latency:          33.9%
================================================================================

✓ SUCCESS: Achieved 1.51x speedup through overlap!
  Communication latency hidden: 33.9%
```

## Performance Tuning

### Adjust Chunk Count
Modify `num_chunks` in `run_benchmark()`:
- **Fewer chunks (2-4)**: Lower overhead, less overlap
- **More chunks (8-16)**: More overlap, higher overhead
- **Optimal**: Balance between granularity and overhead

### Problem Size
Larger matrices benefit more from overlap:
```python
# In run_benchmark()
batch_size = 16        # Increase for more work
seq_len = 4096         # Longer sequences
in_features = 8192     # Larger hidden dimensions
out_features = 8192
```

## Hardware Considerations

### NVLink vs PCIe
- **NVLink**: Higher bandwidth, more overlap benefit
- **PCIe**: Lower bandwidth, communication-bound

### Compute vs Communication Bound
- **Compute-bound**: Less speedup (computation >> communication)
- **Communication-bound**: More speedup (communication is bottleneck)

## Code Structure

```
multi-gpu-infra-overlap/
├── tp_overlap_poc.py                  # Basic PoC with BaselineRowParallelLinear and OverlapRowParallelLinear
├── qwen_comparative_experiment.py     # Qwen-32B specific experiment with forward_baseline() and forward_overlap()
├── advanced_analysis.py               # Chunk size sweep and profiling analysis
├── test_correctness.py                # Unit tests for numerical correctness
├── run_benchmark.sh                   # Launcher for basic PoC
├── run_qwen.sh                        # Launcher for Qwen experiment
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

### Key Functions

**qwen_comparative_experiment.py**:
- `forward_baseline(x, weight)`: Standard GEMM + blocking all_reduce
- `forward_overlap(x, weight, num_chunks)`: Chunked GEMM with pipelined all_reduce
- `validate_correctness()`: torch.allclose verification
- `benchmark_forward()`: Accurate GPU timing with CUDA events

## Synchronization Points

### Compute → Communication
```python
self.compute_events[i].record(compute_stream)
comm_stream.wait_event(self.compute_events[i])
```
Ensures chunk computation completes before all_reduce starts.

### Communication → Output
```python
self.comm_events[i].record(comm_stream)
event.synchronize()
```
Ensures all_reduce completes before returning output.

## Troubleshooting

### No Speedup Observed
1. **Check GPU interconnect**: `nvidia-smi topo -m`
2. **Increase problem size**: Larger matrices show more benefit
3. **Adjust chunk count**: Try 2, 4, 8, 16 chunks
4. **Verify NCCL**: Ensure NCCL is properly installed

### NCCL Errors
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### Out of Memory
Reduce batch size or sequence length:
```python
batch_size = 4
seq_len = 1024
```

## References

- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
- NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/

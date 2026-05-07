#!/bin/bash
# Profile using nvprof to analyze overlap

set -e

echo "=========================================="
echo "Profiling with nvprof"
echo "=========================================="

NPROC=${NPROC:-2}

# Profile baseline
echo ""
echo "[1/3] Profiling baseline..."
/usr/local/cuda/bin/nvprof \
    --profile-child-processes \
    --print-gpu-trace \
    torchrun --nproc_per_node=$NPROC profile_overlap.py --mode baseline \
    2>&1 | tee ../profiles/baseline_nvprof.txt

# Profile original overlap
echo ""
echo "[2/3] Profiling original overlap..."
/usr/local/cuda/bin/nvprof \
    --profile-child-processes \
    --print-gpu-trace \
    torchrun --nproc_per_node=$NPROC profile_overlap.py --mode overlap \
    2>&1 | tee ../profiles/overlap_nvprof.txt

# Profile double buffer
echo ""
echo "[3/3] Profiling double buffer..."
/usr/local/cuda/bin/nvprof \
    --profile-child-processes \
    --print-gpu-trace \
    torchrun --nproc_per_node=$NPROC profile_double_buffer.py \
    2>&1 | tee ../profiles/double_buffer_nvprof.txt

echo ""
echo "=========================================="
echo "Profiling complete!"
echo "=========================================="
echo "Generated files in profiles/:"
echo "  - baseline_nvprof.txt"
echo "  - overlap_nvprof.txt"
echo "  - double_buffer_nvprof.txt"
echo ""
echo "To analyze overlap, run:"
echo "  python analyze_double_buffer.py"
echo ""

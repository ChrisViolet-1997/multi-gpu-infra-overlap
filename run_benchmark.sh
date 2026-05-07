#!/bin/bash
# Quick benchmark script

set -e

echo "=========================================="
echo "Tensor Parallel Overlap Benchmark"
echo "=========================================="

# Default config
NPROC=${NPROC:-2}
BATCH_SIZE=${BATCH_SIZE:-1}
SEQ_LEN=${SEQ_LEN:-2048}
IN_FEATURES=${IN_FEATURES:-4096}
OUT_FEATURES=${OUT_FEATURES:-12288}
NUM_CHUNKS=${NUM_CHUNKS:-4}

echo "Configuration:"
echo "  GPUs: $NPROC"
echo "  Batch Size: $BATCH_SIZE"
echo "  Seq Length: $SEQ_LEN"
echo "  In Features: $IN_FEATURES"
echo "  Out Features: $OUT_FEATURES"
echo "  Num Chunks: $NUM_CHUNKS"
echo "=========================================="
echo ""

# Run benchmark
BATCH_SIZE=$BATCH_SIZE \
SEQ_LEN=$SEQ_LEN \
IN_FEATURES=$IN_FEATURES \
OUT_FEATURES=$OUT_FEATURES \
NUM_CHUNKS=$NUM_CHUNKS \
torchrun --nproc_per_node=$NPROC tp_overlap_double_buffer.py

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="

#!/bin/bash
# Quick launcher script for TP overlap benchmark

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}TP Overlap Benchmark Launcher${NC}"
echo -e "${GREEN}========================================${NC}"

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. CUDA is required.${NC}"
    exit 1
fi

# Count available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo -e "${YELLOW}Detected ${NUM_GPUS} GPU(s)${NC}"

if [ "$NUM_GPUS" -lt 2 ]; then
    echo -e "${RED}Error: At least 2 GPUs required for this benchmark${NC}"
    exit 1
fi

# Parse command line arguments
NPROC=${1:-2}

if [ "$NPROC" -gt "$NUM_GPUS" ]; then
    echo -e "${YELLOW}Warning: Requested $NPROC GPUs but only $NUM_GPUS available${NC}"
    echo -e "${YELLOW}Using $NUM_GPUS GPUs instead${NC}"
    NPROC=$NUM_GPUS
fi

echo -e "${GREEN}Running benchmark on ${NPROC} GPU(s)...${NC}"
echo ""

# Set NCCL environment variables for better debugging
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Run the benchmark
torchrun --nproc_per_node=$NPROC tp_overlap_poc.py

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Benchmark Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

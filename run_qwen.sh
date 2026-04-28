#!/bin/bash
# Qwen-32B TP Overlap Experiment Launcher
set -e

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NPROC=${1:-2}

if [ "$NPROC" -gt "$NUM_GPUS" ]; then
    NPROC=$NUM_GPUS
fi

echo "Running Qwen-32B experiment on ${NPROC} GPUs..."
torchrun --nproc_per_node=$NPROC qwen_comparative_experiment.py

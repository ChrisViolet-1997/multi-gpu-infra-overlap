#!/bin/bash
# Test chunking overlap benefits in prefill stage

cd "$(dirname "$0")/experiments"

echo "========================================"
echo "Prefill Stage Overlap Test"
echo "========================================"
echo ""
echo "Testing configurations:"
echo "  - BS=4, SeqLen=512"
echo "  - BS=8, SeqLen=512"
echo "  - BS=16, SeqLen=512"
echo ""
echo "Comparing:"
echo "  - Baseline: [1,1,1,1,1,1,1]"
echo "  - Chunked:  [2,2,2,1,2,2,1]"
echo ""
echo "Estimated time: ~5-10 minutes"
echo ""

torchrun --nproc_per_node=2 prefill_comparison.py 2>&1 | tee prefill_test_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================"
echo "Test completed!"
echo "========================================"

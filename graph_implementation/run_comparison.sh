#!/bin/bash
# Unified test entry point for CUDAGraph RowParallelLinear comparison

cd "$(dirname "$0")/experiments"

echo "========================================"
echo "CUDAGraph RowParallelLinear Analysis"
echo "========================================"
echo ""
echo "This script will:"
echo "  STEP 1: Grid Search (find optimal config)"
echo "    - Search space: [1,2,4] for q/k/v/gate/up, [1] for o/down"
echo "    - Total: 3^5 = 243 configurations"
echo "    - Each config: 5 runs, drop max/min, average 3"
echo "    - Batch size: 256"
echo "    - Estimated time: ~30-40 minutes"
echo ""
echo "  STEP 2: Rigorous Comparison"
echo "    - Verify accuracy"
echo "    - Compare baseline vs optimal"
echo "    - Test batch sizes: 128, 256, 512"
echo "    - 10 runs per config"
echo "    - Estimated time: ~5-10 minutes"
echo ""
echo "Total estimated time: ~35-50 minutes"
echo ""

echo "========================================"
echo "STEP 1: Grid Search"
echo "========================================"
torchrun --nproc_per_node=2 grid_search_optimal.py 2>&1 | tee grid_search_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================"
echo "STEP 2: Rigorous Comparison"
echo "========================================"
echo ""
echo "Grid search completed. Now running rigorous comparison..."
echo ""

torchrun --nproc_per_node=2 rigorous_comparison.py 2>&1 | tee comparison_results_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================"
echo "All tests completed!"
echo "========================================"
echo "Results saved to:"
echo "  - experiments/grid_search_*.log"
echo "  - experiments/comparison_results_*.log"
echo "========================================"

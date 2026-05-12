#!/bin/bash
# Monitor grid search progress

OUTPUT_FILE="/tmp/claude-0/-root-autodl-tmp-multi-gpu-infra-overlap/tasks/bd40d4c.output"

echo "Monitoring Grid Search Progress..."
echo "Press Ctrl+C to stop monitoring (grid search will continue running)"
echo ""

while true; do
    clear
    echo "========================================"
    echo "Grid Search Progress Monitor"
    echo "========================================"
    echo ""

    # Show progress
    echo "Progress Updates:"
    grep "Progress:" "$OUTPUT_FILE" | tail -5
    echo ""

    # Show best configurations found
    echo "Best Configurations Found:"
    grep "New best" "$OUTPUT_FILE" | tail -10
    echo ""

    # Show current status
    echo "Current Status:"
    tail -3 "$OUTPUT_FILE"
    echo ""

    echo "Last updated: $(date)"
    echo "Refreshing in 30 seconds..."

    sleep 30
done

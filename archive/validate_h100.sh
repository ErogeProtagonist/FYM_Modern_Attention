#!/bin/bash
# =============================================================================
# H100 MLA Validation Script
# Quick test to verify MLA training works on H100 cluster
# =============================================================================

echo "=============================================="
echo "H100 MLA Validation Run"
echo "=============================================="

# Verify GPUs
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Navigate to code
cd ~/yadda

# Quick 100-step MLA test with torch.compile
echo "Starting MLA validation (100 steps)..."
torchrun --standalone --nproc_per_node=8 train_edge.py \
    --model_type mla \
    --shards_dir /lambda/nfs/riddy/edu_fineweb10B \
    --max_steps 100 \
    --val_interval 50 \
    --save_interval 100 \
    --batch_size 16 \
    --compile

echo ""
echo "=============================================="
echo "Validation complete! Check tok/sec above."
echo "=============================================="
echo ""
echo "Expected throughput on 8x H100:"
echo "  - Hybrid: ~800k-1.2M tok/sec"
echo "  - MLA: ~600k-900k tok/sec (due to up-projections)"
echo ""
echo "Training time estimate for 10B tokens:"
echo "  - At 1M tok/sec: ~10,000 sec = ~2.8 hours"
echo "  - At 600k tok/sec: ~16,667 sec = ~4.6 hours"

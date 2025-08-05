#!/bin/bash
# Quick launcher for OneCycleLR scheduler training on Vertex AI

echo "🚀 Launching CBM training with OneCycleLR scheduler..."

# Default parameters
EPOCHS=8
GPU_TYPE="T4"
BATCH_SIZE=32
REGION="europe-west4"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --gpu-type)
            GPU_TYPE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Build and submit job
python3 scripts/launch_vertex.py \
    --script "scripts/train_cbm_vertex_lr_schedule.py" \
    --gpu-type "$GPU_TYPE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --resume-from "gs://art-dna-ml-models/cbm/vertex-runs/best_checkpoint.pth" \
    $DRY_RUN

echo "✅ Job submission completed!"
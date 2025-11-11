#!/usr/bin/env bash
set -euo pipefail

# Distributed launcher configuration (torchrun via python -m torch.distributed.run)
# Number of GPUs to use on this node (set to the available GPU count)
NPROC_PER_NODE=8
# TCP port used by torch.distributed for rendezvous; change if the default collides
MASTER_PORT=29500
# Python interpreter that provides torch.distributed
PYTHON=python

# Absolute path to the training entry point
SCRIPT="$(dirname "$0")/train_phase_classifier.py"

# ---------------------------------------------------------------------------
# Dataset shard configuration
# ---------------------------------------------------------------------------
# Directory that contains training shards (expects X_*.npy / y_*.npy files)
TRAIN_SHARDS="/workspace/siyan/cache/phase_patches/abdomenatlas3/train"
# Optional validation shards directory; leave empty to disable validation
VAL_SHARDS="/workspace/siyan/cache/phase_patches/abdomenatlas3/test"

# ---------------------------------------------------------------------------
# DINOv3 backbone configuration
# ---------------------------------------------------------------------------
# Local path to the cloned DINOv3 repository that provides model definitions
DINOV3_REPO="/workspace/siyan/vlm_med/dinov3"
# Teacher checkpoint to initialise the backbone (typically the DINOv3 teacher .pth)
DINOV3_CKPT="/workspace/siyan/vlm_med/dinov3/checkpoints/dinov3_teacher.pth"
# Identifier of the backbone architecture (informational only)
DINOV3_ARCH="dinov3_vitb16"
# Whether to keep the backbone frozen (1) or allow finetuning (0)
FREEZE_BACKBONE=1

# ---------------------------------------------------------------------------
# Classification head hyperparameters
# ---------------------------------------------------------------------------
# Number of transformer blocks stacked in the multi-head attention head
HEAD_DEPTH=6
# Attention heads per transformer block
HEAD_HEADS=8
# MLP expansion ratio inside each transformer block
HEAD_MLP_RATIO=4.0
# Dropout probability applied within the attention head
HEAD_DROPOUT=0.1
# Total number of phase classes to predict
NUM_CLASSES=3

# ---------------------------------------------------------------------------
# Optimiser and training schedule
# ---------------------------------------------------------------------------
# Number of epochs to train
EPOCHS=60
# Interval (in epochs) to persist additional checkpoints (epoch_XXXX.pth)
CKPT_INTERVAL=100
# Per-GPU batch size (effective batch = BATCH_SIZE * NPROC_PER_NODE)
BATCH_SIZE=128
# Base learning rate for AdamW
LEARNING_RATE=3e-4
# Weight decay applied by AdamW
WEIGHT_DECAY=1e-4
# AdamW beta coefficients (beta1 beta2)
BETAS="0.9 0.999"
# Gradient accumulation steps to simulate larger effective batch sizes
GRAD_ACCUM=1

# ---------------------------------------------------------------------------
# DataLoader parameters
# ---------------------------------------------------------------------------
# Number of worker processes per GPU for data loading
NUM_WORKERS=8
# Prefetch factor per worker to pipeline batches
PREFETCH_FACTOR=4
# Enable persistent workers across epochs (1 to enable)
PERSISTENT_WORKERS=0
# Pin host memory for faster host->GPU transfers (1 keeps default pinned memory behaviour)
PIN_MEMORY=1

# ---------------------------------------------------------------------------
# Loss weighting and bookkeeping
# ---------------------------------------------------------------------------
# Optional comma-separated class weights for the imbalanced phase distribution (empty string disables weighting)
CLASS_WEIGHTS="1.0,0.7,12.0"
# Directory where checkpoints, logs, and configuration snapshots will be stored
WORKDIR="/workspace/siyan/experiments/phase_classifier"
# Optional checkpoint path to resume training from (leave empty to start fresh)
RESUME=""
# Base random seed; DDP ranks add their rank index to this seed
SEED=2025

# ---------------------------------------------------------------------------
# Assemble command
# ---------------------------------------------------------------------------
LAUNCHER=("$PYTHON" -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT")

CMD=("${LAUNCHER[@]}" "$SCRIPT"
  --train-shards "$TRAIN_SHARDS"
  --dinov3-repo "$DINOV3_REPO"
  --dinov3-ckpt "$DINOV3_CKPT"
  --dinov3-arch "$DINOV3_ARCH"
  --num-classes "$NUM_CLASSES"
  --head-depth "$HEAD_DEPTH"
  --head-heads "$HEAD_HEADS"
  --head-mlp-ratio "$HEAD_MLP_RATIO"
  --head-dropout "$HEAD_DROPOUT"
  --epochs "$EPOCHS"
  --ckpt-interval "$CKPT_INTERVAL"
  --batch "$BATCH_SIZE"
  --lr "$LEARNING_RATE"
  --wd "$WEIGHT_DECAY"
  --grad-accum "$GRAD_ACCUM"
  --num-workers "$NUM_WORKERS"
  --prefetch-factor "$PREFETCH_FACTOR"
  --workdir "$WORKDIR"
  --seed "$SEED"
)

if [[ -n "$VAL_SHARDS" ]]; then
  CMD+=(--val-shards "$VAL_SHARDS")
fi

# Split BETAS into individual arguments
if [[ -n "$BETAS" ]]; then
  read -r BETA1 BETA2 <<< "$BETAS"
  CMD+=(--betas "$BETA1" "$BETA2")
fi

if (( FREEZE_BACKBONE )); then
  CMD+=(--freeze-backbone)
fi

if (( PERSISTENT_WORKERS )); then
  CMD+=(--persistent-workers)
fi

if (( PIN_MEMORY )); then
  CMD+=(--pin-memory)
fi

if [[ -n "$CLASS_WEIGHTS" ]]; then
  CMD+=(--class-weights "$CLASS_WEIGHTS")
fi

if [[ -n "$RESUME" ]]; then
  CMD+=(--resume "$RESUME")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "✅ Phase classifier training complete"

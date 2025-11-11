#!/usr/bin/env bash
set -euo pipefail

# Path to Python interpreter
PY=python

# Script path
SCRIPT="$(dirname "$0")/prepare_phase_patches.py"

# Dataset and metadata paths
# Root folder containing per-case subdirectories with CT volumes
DATASET_ROOT="/workspace/vlm_medical/open_data/AbdomenAtlas3/AbdomenAtlasTest"
# Excel workbook that maps case IDs to phase labels and spacing metadata
METADATA_XLSX="/workspace/vlm_medical/open_data/AbdomenAtlas3/BDMAP_venous_arterial_delay.xlsx"

# Output directory
# Destination root where train/test shards and metadata will be written
OUTDIR="/workspace/siyan/cache/phase_patches/abdomenatlas3"

# Ensure output directory exists
mkdir -p "$OUTDIR"

# Sampling configuration
# Number of axial slices to stack per sample (must be odd)
SLICE_STACK=3
# Comma-separated crop sizes (shortest side in pixels) to sample per patch
CROP_SIZES="128,192,256"
# How many patch centers to draw per slice before multi-scale expansion
PATCHES_PER_SLICE=9
# Relative oversampling factor per phase class (higher => more patches)
CLASS_PATCH_MULTIPLIERS="Arterial:1.0,Venous:1.0,Delay:4.0"
# Foreground-guided sampling is disabled because the classification task does not use segmentation masks

# Final square patch resolution written to disk
TARGET_SIZE=256
# HU window center for CT intensity normalization
HU_CENTER=60
# HU window width for CT intensity normalization
HU_WIDTH=400
# Maximum number of slices to sample from each volume
MAX_SLICES_PER_VOL=999999
# Number of patches per shard file before flushing to disk
SHARD_SIZE=2048
# Parallel worker processes used for sampling
NUM_WORKERS=8
# Random seed base for reproducible sampling
SEED=2025
# Portion of cases reserved for the held-out test split
TEST_RATIO=0.2
# Number of patches to sample during dry-run storage estimation (0 disables dry-run)
DRY_RUN_PATCHES=0
# Set to 1 to resume from existing shards instead of overwriting them
RESUME=0

# Optional newline-delimited list of case IDs to process (empty => all)
CASE_LIST=""
# Optional Excel sheet override when metadata workbook has multiple sheets
EXCEL_SHEET=""

CMD=("$PY" "$SCRIPT"
  --dataset-root "$DATASET_ROOT"
  --metadata-xlsx "$METADATA_XLSX"
  --outdir "$OUTDIR"
  --slice-stack "$SLICE_STACK"
  --crop-sizes "$CROP_SIZES"
  --patches-per-slice "$PATCHES_PER_SLICE"
  --class-patch-multipliers "$CLASS_PATCH_MULTIPLIERS"
  --target-size "$TARGET_SIZE"
  --hu-center "$HU_CENTER"
  --hu-width "$HU_WIDTH"
  --max-slices-per-vol "$MAX_SLICES_PER_VOL"
  --shard-size "$SHARD_SIZE"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --test-ratio "$TEST_RATIO"
)

if [[ -n "$CASE_LIST" ]]; then
  CMD+=(--case-list "$CASE_LIST")
fi

if [[ -n "$EXCEL_SHEET" ]]; then
  CMD+=(--excel-sheet "$EXCEL_SHEET")
fi

if (( DRY_RUN_PATCHES > 0 )); then
  CMD+=(--dry-run-patches "$DRY_RUN_PATCHES")
fi

if (( RESUME )); then
  CMD+=(--resume)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "✅ Phase patch preparation complete"

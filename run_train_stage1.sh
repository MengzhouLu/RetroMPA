#!/bin/bash
# Stage 1 Pretraining Launch Script (DDP)
# Run with: bash run_train_stage1.sh [--arg ...]

NUM_GPUS=$(python - <<'PY'
import config
print(config.num_gpus)
PY
)

torchrun --nproc_per_node="$NUM_GPUS" \
    scripts/train_stage1.py \
    "$@"

#!/bin/bash
# Stage 2 Fine-tuning Launch Script (DDP)
# Run with: bash run_train_stage2.sh [--arg ...]

NUM_GPUS=$(python - <<'PY'
import config
print(config.num_gpus)
PY
)

torchrun --nproc_per_node="$NUM_GPUS" \
    scripts/train_stage2.py \
    "$@"

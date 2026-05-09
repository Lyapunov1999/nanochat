#!/bin/bash

set -euo pipefail

# Batch compare multiple optimizers with the base-only pretraining flow.
#
# Example:
#   bash runs/pretrain_compare.sh
#   OPTIMIZERS="nanochat adamw myopt1" COMPARE_NAME=apr-exp bash runs/pretrain_compare.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

COMPARE_NAME="${COMPARE_NAME:-optimizer-compare}"
COMPARE_ROOT="${COMPARE_ROOT:-$NANOCHAT_BASE_DIR/optimizer_compare/$COMPARE_NAME}"
OPTIMIZERS="${OPTIMIZERS:-nanochat adamw sgd}"

mkdir -p "$COMPARE_ROOT"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

first_run=1
for optimizer in $OPTIMIZERS; do
    experiment_name="${COMPARE_NAME}-${optimizer}"
    report_dir="$COMPARE_ROOT/$optimizer/report"
    run_skip_setup="${SKIP_SETUP:-}"
    if [ $first_run -eq 0 ] && [ -z "$run_skip_setup" ]; then
        run_skip_setup=1
    fi

    log "Running optimizer=$optimizer"
    OPTIMIZER="$optimizer" \
    EXPERIMENT_NAME="$experiment_name" \
    NANOCHAT_REPORT_DIR="$report_dir" \
    SKIP_SETUP="$run_skip_setup" \
    FORCE_TOKENIZER_TRAIN="$first_run" \
    bash runs/pretrain_base.sh

    first_run=0
done

python -m scripts.pretrain_compare_report --root "$COMPARE_ROOT"
log "Comparison summary written to $COMPARE_ROOT/compare.md"

#!/bin/bash

set -euo pipefail

# Single-card base-model experiment entrypoint.
# Runs: report reset -> tokenizer/check -> base_train -> base_eval -> report generate
#
# Examples:
#   bash runs/pretrain_base.sh
#   OPTIMIZER=adamw MODEL_TAG=d24-adamw bash runs/pretrain_base.sh
#   OPTIMIZER=myopt1 BASE_TRAIN_EXTRA_ARGS="--optimizer-lr=0.001" bash runs/pretrain_base.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

OPTIMIZER="${OPTIMIZER:-nanochat}"
DEPTH="${DEPTH:-24}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-8}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
WANDB_RUN="${WANDB_RUN:-dummy}"
ENABLE_FP8="${ENABLE_FP8:-1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pretrain-${OPTIMIZER}}"
MODEL_TAG="${MODEL_TAG:-d${DEPTH}-${OPTIMIZER}}"
INITIAL_TOKENIZER_SHARDS="${INITIAL_TOKENIZER_SHARDS:-8}"
DATASET_SHARDS="${DATASET_SHARDS:-170}"
FORCE_TOKENIZER_TRAIN="${FORCE_TOKENIZER_TRAIN:-0}"
RUN_BASE_EVAL="${RUN_BASE_EVAL:-1}"

mkdir -p "$NANOCHAT_BASE_DIR"

if [ -z "${NANOCHAT_REPORT_DIR:-}" ]; then
    export NANOCHAT_REPORT_DIR="$NANOCHAT_BASE_DIR/reports/$EXPERIMENT_NAME"
fi
mkdir -p "$NANOCHAT_REPORT_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if [ -z "${SKIP_SETUP:-}" ]; then
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
fi
source .venv/bin/activate

log "Base dir: $NANOCHAT_BASE_DIR"
log "Report dir: $NANOCHAT_REPORT_DIR"
log "Experiment: $EXPERIMENT_NAME"
log "Optimizer: $OPTIMIZER"
log "Model tag: $MODEL_TAG"
log "Single-card mode: nproc_per_node=$NPROC_PER_NODE"

python -m nanochat.report reset

TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
if [ ! -d "$TOKENIZER_DIR" ] || [ "$FORCE_TOKENIZER_TRAIN" = "1" ]; then
    log "Training tokenizer"
    python -m nanochat.dataset -n "$INITIAL_TOKENIZER_SHARDS"
    python -m scripts.tok_train
else
    log "Reusing existing tokenizer at $TOKENIZER_DIR"
fi

if [ -d "$TOKENIZER_DIR" ]; then
    log "Evaluating tokenizer"
    python -m scripts.tok_eval
fi

log "Ensuring pretraining dataset shards are available"
python -m nanochat.dataset -n "$DATASET_SHARDS"

BASE_TRAIN_ARGS=(
    --depth="$DEPTH"
    --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO"
    --device-batch-size="$DEVICE_BATCH_SIZE"
    --optimizer="$OPTIMIZER"
    --run="$WANDB_RUN"
    --model-tag="$MODEL_TAG"
)

if [ "$ENABLE_FP8" = "1" ]; then
    BASE_TRAIN_ARGS+=(--fp8)
fi

log "Starting base pretraining"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    "${BASE_TRAIN_ARGS[@]}" \
    ${BASE_TRAIN_EXTRA_ARGS:-}

if [ "$RUN_BASE_EVAL" = "1" ]; then
    log "Running base evaluation"
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
        --model-tag="$MODEL_TAG" \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        ${BASE_EVAL_EXTRA_ARGS:-}
fi

python -m nanochat.report generate
log "Report generated at $NANOCHAT_REPORT_DIR/report.md"

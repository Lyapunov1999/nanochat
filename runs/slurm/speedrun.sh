#!/bin/bash
#SBATCH --job-name=nanochat-speedrun
#SBATCH --account=rrg-schmidtm
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=runs/slurm/logs/%x-%j.out
#SBATCH --error=runs/slurm/logs/%x-%j.err
## If your Compute Canada cluster requires them, uncomment and fill these in:
## #SBATCH --partition=YOUR_PARTITION
## #SBATCH --qos=YOUR_QOS

set -euo pipefail

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p runs/slurm/logs

# -----------------------------------------------------------------------------
# Cluster / environment configuration

# Customize module loads for your specific Compute Canada cluster if needed.
# These are intentionally placeholders because module names vary by site/image.
if command -v module >/dev/null 2>&1; then
    log "Environment modules detected; load site-specific Python/CUDA modules here if required."
    # module load python/3.12 cuda/12.8
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export WANDB_RUN="${WANDB_RUN:-dummy}"
export TRAIN_DEVICE_BATCH_SIZE="${TRAIN_DEVICE_BATCH_SIZE:-16}"
export EVAL_DEVICE_BATCH_SIZE="${EVAL_DEVICE_BATCH_SIZE:-16}"

mkdir -p "$NANOCHAT_BASE_DIR"

log "Job id: ${SLURM_JOB_ID:-unknown}"
log "Working directory: $(pwd)"
log "Base directory: $NANOCHAT_BASE_DIR"
log "WANDB_RUN: $WANDB_RUN"
log "Single-H100 mode: this will be much slower than the upstream 8xH100 speedrun."

# -----------------------------------------------------------------------------
# Python environment setup

if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d ".venv" ]; then
    log "Creating virtual environment"
    uv venv
fi

log "Syncing Python dependencies"
uv sync --extra gpu

# shellcheck disable=SC1091
source .venv/bin/activate

log "Python: $(python --version 2>&1)"
log "uv: $(uv --version 2>&1)"

# -----------------------------------------------------------------------------
# Report initialization

log "Resetting run report"
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer and dataset

log "Downloading initial dataset shards"
python -m nanochat.dataset -n 8

log "Starting background dataset download for pretraining"
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!

log "Training tokenizer"
python -m scripts.tok_train

log "Evaluating tokenizer"
python -m scripts.tok_eval

log "Waiting for dataset download to complete"
wait "$DATASET_DOWNLOAD_PID"

# -----------------------------------------------------------------------------
# Base model pretraining and evaluation

log "Starting base model training on a single H100"
python -m scripts.base_train \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size="$TRAIN_DEVICE_BATCH_SIZE" \
    --fp8 \
    --run="$WANDB_RUN"

log "Running base evaluation"
python -m scripts.base_eval \
    --device-batch-size="$EVAL_DEVICE_BATCH_SIZE"

log "Generating final report"
python -m nanochat.report generate

log "Speedrun job complete"

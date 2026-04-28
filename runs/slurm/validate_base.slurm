#!/bin/bash
#SBATCH --job-name=nanochat-validate
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=10:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs

# Fill these in for your cluster, e.g.:
# module purge
# module load cuda/12.4
# module load python/3.11

if [ ! -d ".venv" ]; then
    echo ".venv not found. Run runs/slurm/prep.slurm first." >&2
    exit 1
fi

source .venv/bin/activate

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

RUN_NAME="${RUN_NAME:-d24-validate}"
MODEL_TAG="${MODEL_TAG:-d24_validate}"
DEPTH="${DEPTH:-24}"
TARGET_RATIO="${TARGET_RATIO:-8}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
EVAL_DEVICE_BATCH_SIZE="${EVAL_DEVICE_BATCH_SIZE:-16}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

echo "Run name: $RUN_NAME"
echo "Model tag: $MODEL_TAG"
echo "Depth: $DEPTH"
echo "Target ratio: $TARGET_RATIO"
echo "Device batch size: $DEVICE_BATCH_SIZE"
echo "Max seq len: $MAX_SEQ_LEN"
echo "Eval batch size: $EVAL_DEVICE_BATCH_SIZE"
echo "GPUs: $NPROC_PER_NODE"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --depth="$DEPTH" \
    --run="$RUN_NAME" \
    --model-tag="$MODEL_TAG" \
    --target-param-data-ratio="$TARGET_RATIO" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --max-seq-len="$MAX_SEQ_LEN" \
    --fp8 \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=999999 \
    $EXTRA_TRAIN_ARGS

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$EVAL_DEVICE_BATCH_SIZE" \
    $EXTRA_EVAL_ARGS

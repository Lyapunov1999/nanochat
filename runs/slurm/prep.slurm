#!/bin/bash
#SBATCH --job-name=nanochat-prep
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs

# Fill these in for your cluster, e.g.:
# module purge
# module load cuda/12.4
# module load python/3.11

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

python -m nanochat.report reset
python -m nanochat.dataset -n "${DATASET_SHARDS:-170}"
python -m scripts.tok_train
python -m scripts.tok_eval

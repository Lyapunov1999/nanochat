#!/bin/bash

set -euo pipefail

# Fill these in for your cluster before using these examples.
ACCOUNT="${ACCOUNT:-<ACCOUNT>}"
PARTITION="${PARTITION:-<PARTITION>}"
QOS="${QOS:-<QOS>}"

cat <<EOF
# 1) Prep once on shared storage
sbatch \\
  --account=${ACCOUNT} \\
  --partition=${PARTITION} \\
  --qos=${QOS} \\
  runs/slurm/prep.slurm

# 2) Two parallel 2-GPU screening runs
sbatch \\
  --account=${ACCOUNT} \\
  --partition=${PARTITION} \\
  --qos=${QOS} \\
  --export=ALL,RUN_NAME=d24-screen-a,MODEL_TAG=d24_screen_a,DEPTH=24,TARGET_RATIO=8,DEVICE_BATCH_SIZE=8,MAX_SEQ_LEN=2048 \\
  runs/slurm/screen_base.slurm

sbatch \\
  --account=${ACCOUNT} \\
  --partition=${PARTITION} \\
  --qos=${QOS} \\
  --export=ALL,RUN_NAME=d24-screen-b,MODEL_TAG=d24_screen_b,DEPTH=24,TARGET_RATIO=8.5,DEVICE_BATCH_SIZE=8,MAX_SEQ_LEN=2048 \\
  runs/slurm/screen_base.slurm

# 3) Promote the best candidate to a 4-GPU validation run
sbatch \\
  --account=${ACCOUNT} \\
  --partition=${PARTITION} \\
  --qos=${QOS} \\
  --export=ALL,RUN_NAME=d24-validate,MODEL_TAG=d24_validate,DEPTH=24,TARGET_RATIO=8,DEVICE_BATCH_SIZE=8,MAX_SEQ_LEN=2048,EVAL_DEVICE_BATCH_SIZE=16 \\
  runs/slurm/validate_base.slurm

# OOM fallback order
#   DEVICE_BATCH_SIZE: 16 -> 8 -> 4 -> 2 -> 1
#   MAX_SEQ_LEN: 2048 -> 1024 -> 512
#   DEPTH: 24 -> 22 -> 20
EOF

#!/usr/bin/env bash
#SBATCH --job-name=dexa-still
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-dexa-still-%j.out
#
# Distill the STILL perceivers against a frozen base model on one GPU.
#
#   sbatch scripts/slurm/train_still.sbatch                                   # default config
#   sbatch --export=ALL,CONFIG=configs/still-train.yaml scripts/slurm/train_still.sbatch
#   CONFIG=configs/still-train.yaml sbatch scripts/slurm/train_still.sbatch
#
set -euo pipefail

# --- site setup (uncomment / edit for your cluster) ------------------------
# module load cuda/12.1
# module load anaconda3
# conda activate dexa
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
if [[ -d "${REPO_ROOT}/.venv" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

# --- run -------------------------------------------------------------------
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# export HF_TOKEN=...    # set for gated base models

CONFIG="${CONFIG:-configs/still-train.yaml}"
echo "==> node $(hostname)  config=${CONFIG}"
nvidia-smi || true

srun python -m dexa.compaction.still.train --config "${CONFIG}"

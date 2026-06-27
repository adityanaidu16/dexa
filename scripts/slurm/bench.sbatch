#!/usr/bin/env bash
#SBATCH --job-name=dexa-bench
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-dexa-bench-%j.out
#
# Submit a config-driven Dexa benchmark on one GPU.
#
#   sbatch scripts/slurm/bench.sbatch                                  # default config
#   sbatch --export=ALL,CONFIG=configs/llama32-3b.yaml scripts/slurm/bench.sbatch
#   CONFIG=configs/llama31-8b.yaml sbatch scripts/slurm/bench.sbatch
#
set -euo pipefail

# --- site setup (uncomment / edit for your cluster) ------------------------
# module load cuda/12.1
# module load anaconda3
# conda activate dexa
# Or, with the repo virtualenv created by scripts/setup_env.sh:
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
if [[ -d "${REPO_ROOT}/.venv" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

# --- run -------------------------------------------------------------------
# HF auth for gated models; xet workaround for stalled downloads.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# export HF_TOKEN=...    # set in your environment / --export for gated meta-llama

CONFIG="${CONFIG:-configs/llama32-1b.yaml}"
echo "==> node $(hostname)  config=${CONFIG}"
nvidia-smi || true

srun python -m dexa.cli run --config "${CONFIG}"

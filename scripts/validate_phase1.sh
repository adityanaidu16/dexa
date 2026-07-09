#!/usr/bin/env bash
# Provider-agnostic Phase-1 validation on any CUDA box (RunPod, an SSH VM, SLURM).
# Runs the two benchmarks that need a *trained* model: exact incremental recompute
# and selective (HKVD) recompute. On Modal use scripts/modal_validate.py instead.
#
# On a fresh RunPod PyTorch pod:
#   cd /workspace && git clone https://github.com/adityanaidu16/dexa.git && cd dexa
#   CONFIG="" bash scripts/runpod_bootstrap.sh      # set up the persistent venv only
#   bash scripts/validate_phase1.sh                 # then this
#
# On a box where the .venv already exists (setup_env.sh / runpod_bootstrap.sh):
#   bash scripts/validate_phase1.sh [MODEL]
#
# Env: MODEL (default ungated Llama-3.1-8B), REPS, STEPS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${1:-${MODEL:-unsloth/Llama-3.1-8B-Instruct}}"
REPS="${REPS:-10}"
STEPS="${STEPS:-20}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# Prefer the repo venv (created by setup_env.sh / runpod_bootstrap.sh); else system python.
if [ -x .venv/bin/python ]; then
  PY=".venv/bin/python"
else
  PY="${PYTHON:-python}"
  echo "==> no .venv found; using ${PY}. Run scripts/setup_env.sh first if imports fail." >&2
fi

echo "==> torch / GPU check"
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(), \
  torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

echo -e "\n====================================================================="
echo "Layer B — exact incremental recompute (tokens saved on an agent loop)"
echo "====================================================================="
$PY benchmarks/incremental_recompute_bench.py --steps "$STEPS" --timed --device cuda --model "$MODEL"

echo -e "\n====================================================================="
echo "Layer C — selective (HKVD) recompute vs recency/random (magnitude test)"
echo "====================================================================="
$PY benchmarks/selective_recompute_bench.py --device cuda --reps "$REPS" --model "$MODEL"

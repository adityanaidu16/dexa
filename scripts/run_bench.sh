#!/usr/bin/env bash
# Thin wrapper around `dexa run --config <cfg>`.
#
#   ./scripts/run_bench.sh                              # default config
#   ./scripts/run_bench.sh configs/llama32-3b.yaml      # pick a config
#   ./scripts/run_bench.sh configs/llama31-8b.yaml --   # extra args pass through
#
# Assumes the .venv is on PATH (run `source .venv/bin/activate` first, or use the
# SLURM scripts which activate it for you).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CFG="${1:-configs/llama32-1b.yaml}"
shift || true   # remaining args (if any) pass through to the CLI

# Work around HF's xet transfer protocol stalling in some cluster envs.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

echo "==> dexa run --config ${CFG}"
exec python -m dexa.cli run --config "${CFG}" "$@"

#!/usr/bin/env bash
# Idempotent environment bootstrap for a CUDA GPU box.
#
#   ./scripts/setup_env.sh                 # CPU/CUDA torch (cu121) + bench extras
#   CUDA=cu124 ./scripts/setup_env.sh      # pick a CUDA wheel matching your driver
#   ./scripts/setup_env.sh --vllm          # also install the [gpu] (vLLM) extra
#
# Safe to re-run: it skips work that is already done.
set -euo pipefail

# --- locate repo root (this script lives in scripts/) ----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- args / config ---------------------------------------------------------
CUDA="${CUDA:-cu121}"          # CUDA wheel tag, e.g. cu121, cu124. Override via env.
PYTHON_BIN="${PYTHON_BIN:-python3}"
WANT_VLLM=0
for arg in "$@"; do
  case "${arg}" in
    --vllm) WANT_VLLM=1 ;;
    *) echo "unknown arg: ${arg}" >&2; exit 2 ;;
  esac
done

VENV="${REPO_ROOT}/.venv"
PIP="${VENV}/bin/pip"

# --- 1. virtualenv ---------------------------------------------------------
if [[ ! -d "${VENV}" ]]; then
  echo "==> creating virtualenv at ${VENV}"
  "${PYTHON_BIN}" -m venv "${VENV}"
else
  echo "==> reusing existing virtualenv at ${VENV}"
fi

"${PIP}" install --upgrade pip wheel >/dev/null

# --- 2. CUDA torch ---------------------------------------------------------
# Install torch from the matching CUDA index BEFORE the editable install so the
# CUDA wheel (not the default CPU one) is what gets resolved.
echo "==> installing CUDA torch (${CUDA})"
"${PIP}" install torch --index-url "https://download.pytorch.org/whl/${CUDA}"

# --- 3. project + extras ---------------------------------------------------
echo "==> installing dexa with [torch,bench] extras (editable)"
"${PIP}" install -e '.[torch,bench]'

if [[ "${WANT_VLLM}" -eq 1 ]]; then
  echo "==> installing vLLM ([gpu] extra)"
  "${PIP}" install -e '.[gpu]'
fi

# --- next steps ------------------------------------------------------------
cat <<EOF

==> done. Next steps:
  source ${VENV}/bin/activate
  export HF_TOKEN=...                 # for gated meta-llama models
  export HF_HUB_DISABLE_XET=1         # if model downloads stall
  ./scripts/run_bench.sh configs/llama32-1b.yaml

See docs/CLUSTER.md for the full runbook.
EOF

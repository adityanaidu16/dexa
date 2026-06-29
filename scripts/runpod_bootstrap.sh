#!/usr/bin/env bash
# One-shot bootstrap for a fresh RunPod pod (a "PyTorch" template with CUDA torch
# already installed). Clones Dexa, installs the rest on top of the base torch,
# puts the HF cache on the persistent /workspace volume, and runs a benchmark.
#
# Quickstart on a RunPod pod's web terminal:
#   cd /workspace
#   wget -qO- https://raw.githubusercontent.com/adityanaidu16/dexa/main/scripts/runpod_bootstrap.sh | bash
# or, if you already cloned the repo:
#   bash scripts/runpod_bootstrap.sh
#
# Env knobs:
#   CONFIG    config to run (default configs/llama32-1b.yaml; "" to skip running)
#   REPO_URL  git remote (default the public repo)
#   HF_TOKEN  required for gated meta-llama models (else use an ungated mirror)
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/adityanaidu16/dexa.git}"
WORKDIR="${WORKDIR:-/workspace}"
CONFIG="${CONFIG:-configs/llama32-1b.yaml}"

# Persist the (large) HF model cache on the RunPod volume so it survives restarts
# and isn't re-downloaded; also dodge the xet protocol that stalls in some envs.
export HF_HOME="${HF_HOME:-$WORKDIR/hf}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "$HF_HOME" "$WORKDIR"
cd "$WORKDIR"

# --- clone / update -------------------------------------------------------
if [ ! -d dexa/.git ]; then
  echo "==> cloning $REPO_URL"
  git clone "$REPO_URL" dexa
fi
cd dexa
git pull --ff-only || true

# --- python: a venv ON the persistent /workspace volume -------------------
# Installing into the container's system python is wiped on every pod reset.
# A venv with --system-site-packages lives on /workspace (persists) while still
# reusing the pod image's CUDA torch, so a reset needs no reinstall.
SYS_PY="${PYTHON:-python}"
if [ ! -x .venv/bin/python ]; then
  echo "==> creating persistent venv (.venv, --system-site-packages)"
  $SYS_PY -m venv .venv --system-site-packages
fi
PY=".venv/bin/python"
echo "==> torch check (must already be CUDA-enabled on a RunPod PyTorch pod)"
$PY - <<'PYEOF'
import sys
try:
    import torch
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
          "device", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"))
    if not torch.cuda.is_available():
        print("WARNING: torch sees no CUDA GPU — pick a RunPod *PyTorch* template on a GPU pod.", file=sys.stderr)
except ImportError:
    print("ERROR: torch not found. Use a RunPod PyTorch template (torch preinstalled).", file=sys.stderr)
    sys.exit(1)
PYEOF

# Install Dexa + the HF/bench stack into the venv WITHOUT reinstalling torch.
# Skip if already importable (persisted venv from a previous boot).
if $PY -c "import dexa, transformers" 2>/dev/null; then
  echo "==> deps already present in persistent venv (skipping install)"
else
  echo "==> pip install -e '.[hf,bench]' into .venv (reusing base torch)"
  $PY -m pip install -q --upgrade pip
  $PY -m pip install -q -e '.[hf,bench]'
fi

# --- HF auth for gated models (non-fatal: most runs use ungated mirrors) ---
if [ -n "${HF_TOKEN:-}" ] && [ "${HF_TOKEN}" != "hf_xxx" ]; then
  echo "==> validating HF_TOKEN"
  if $PY -c "from huggingface_hub import login; login('${HF_TOKEN}')" 2>/dev/null; then
    echo "   hf login ok"
  else
    echo "   WARNING: HF_TOKEN invalid — ignoring it. Use an ungated model" >&2
    echo "   (e.g. configs point at meta-llama; pass --model unsloth/Llama-3.2-1B-Instruct)." >&2
    unset HF_TOKEN
  fi
elif [ "${HF_TOKEN:-}" = "hf_xxx" ]; then
  echo "==> HF_TOKEN is the placeholder 'hf_xxx' — ignoring. Use an ungated mirror via --model." >&2
  unset HF_TOKEN
fi

echo "==> environment ready. HF_HOME=$HF_HOME"
$PY -m pytest -q tests/test_config_run.py 2>/dev/null && echo "   (config runner self-check ok)" || true

# --- run ------------------------------------------------------------------
# If no valid HF token, default to the ungated mirror so gated meta-llama configs
# still run out of the box. Override with MODEL=... (or "" to use the config's).
MODEL_ARG=()
if [ -n "${MODEL:-}" ]; then
  MODEL_ARG=(--model "$MODEL")
elif [ -z "${HF_TOKEN:-}" ]; then
  MODEL_ARG=(--model "unsloth/Llama-3.2-1B-Instruct")
  echo "==> no HF token: defaulting to ungated --model unsloth/Llama-3.2-1B-Instruct"
fi

if [ -n "$CONFIG" ]; then
  echo "==> dexa run --config $CONFIG ${MODEL_ARG[*]:-}"
  $PY -m dexa.cli run --config "$CONFIG" "${MODEL_ARG[@]}"
  echo "==> done. results in $(grep -E '^out_dir:' "$CONFIG" | awk '{print $2}')"
else
  echo "==> setup only (CONFIG empty). Run e.g.: python -m dexa.cli run --config configs/llama32-1b.yaml"
fi

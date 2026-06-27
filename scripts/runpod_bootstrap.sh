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

# --- python deps on top of the pod's preinstalled CUDA torch --------------
PY="${PYTHON:-python}"
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

# Install Dexa + the HF/bench stack WITHOUT reinstalling torch (reuse the pod's).
echo "==> pip install -e '.[hf,bench]' (reusing base torch)"
$PY -m pip install -q --upgrade pip
$PY -m pip install -q -e '.[hf,bench]'

# --- HF auth for gated models --------------------------------------------
if [ -n "${HF_TOKEN:-}" ]; then
  echo "==> hf login via HF_TOKEN"
  $PY -m huggingface_hub.commands.huggingface_cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null \
    || $PY -c "from huggingface_hub import login; login('${HF_TOKEN}')"
fi

echo "==> environment ready. HF_HOME=$HF_HOME"
$PY -m pytest -q tests/test_config_run.py 2>/dev/null && echo "   (config runner self-check ok)" || true

# --- run ------------------------------------------------------------------
if [ -n "$CONFIG" ]; then
  echo "==> dexa run --config $CONFIG"
  $PY -m dexa.cli run --config "$CONFIG"
  echo "==> done. results in $(grep -E '^out_dir:' "$CONFIG" | awk '{print $2}')"
else
  echo "==> setup only (CONFIG empty). Run e.g.: python -m dexa.cli run --config configs/llama32-1b.yaml"
fi

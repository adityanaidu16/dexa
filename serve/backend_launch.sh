#!/usr/bin/env bash
# BYOC: run the Dexa computer-use backend on your own GPU box (A100/H100, 80GB).
# Same flags as the Modal deploy, so cost/quality is identical. Serves an OpenAI-compatible
# API on :8000 as model "dexa-cua-vlm" — point the gateway's DEXA_BACKEND_URL at this host.
#
#   pip install "vllm==0.24.0" qwen-vl-utils
#   DEXA_MAX_PIXELS=1050000 ./serve/backend_launch.sh
set -euo pipefail

MODEL="${DEXA_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
MAX_PIXELS="${DEXA_MAX_PIXELS:-1050000}"
MAX_MODEL_LEN="${DEXA_MAX_MODEL_LEN:-16384}"
MAX_IMAGES="${DEXA_MAX_IMAGES:-3}"

exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --served-model-name dexa-cua-vlm \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES}}" \
  --mm-processor-kwargs "{\"max_pixels\": ${MAX_PIXELS}}"

#!/usr/bin/env bash
# Run the Dexa gateway locally.
#   ./platform/run_local.sh            # mock backend (no GPU) — great for the demo/dashboard
#   DEXA_BACKEND_URL=https://<modal-url> ./platform/run_local.sh   # real Qwen2.5-VL backend
set -euo pipefail
cd "$(dirname "$0")/.."
export DEXA_MOCK="${DEXA_BACKEND_URL:+0}"; export DEXA_MOCK="${DEXA_MOCK:-1}"
echo "gateway  : http://localhost:8080/v1   (backend: ${DEXA_BACKEND_URL:-mock})"
echo "dashboard: http://localhost:8080/dashboard"
exec uvicorn dexa_platform.gateway.app:app --host 0.0.0.0 --port 8080

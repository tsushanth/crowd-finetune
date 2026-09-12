#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PORT="${PORT:-8001}"
MODEL="${MODEL:-outputs/reasoning-grpo-merged}"

python -c "import vllm" 2>/dev/null || pip install -q vllm

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --served-model-name reasoning-model \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90
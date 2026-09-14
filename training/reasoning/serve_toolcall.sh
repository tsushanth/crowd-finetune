#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PORT="${PORT:-8001}"
MODEL="${MODEL:-outputs/toolcall-7b-lora-merged}"

export VLLM_USE_FLASHINFER_SAMPLER=0

python3 -c "import vllm" 2>/dev/null || pip install -q vllm

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --served-model-name toolcall-7b-lora \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16
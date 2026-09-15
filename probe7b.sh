#!/usr/bin/env bash
# Helper for toolcall-7b box (instance 51027615). Usage: bash probe7b.sh [command-file|"cmd"]
set -euo pipefail
KEY="$HOME/.ssh/runpod_cuda"
PORT="27614"
HOST="ssh7.vast.ai"
if [ $# -gt 0 ] && [ -f "$1" ]; then
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p "$PORT" root@"$HOST" 'bash -s' < "$1"
else
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p "$PORT" root@"$HOST" "${@:-true}"
fi

#!/usr/bin/env bash
# Deploy + train toolcall 7B LoRA on Vast box 50776326
# Usage: bash deploy_toolcall_7b.sh
set -euo pipefail

LOCALREPO="/Users/sushanthtiruvaipati/Documents/Default Project/crowd-finetune"
INSTANCE="50776326"
HOST="ssh7.vast.ai"
PORT="16326"
VAST="/tmp/pv/bin/vastai"

echo "== start instance $INSTANCE =="
$VAST start instance "$INSTANCE" 2>&1 || true

echo "== polling for ready (up to 20 min) =="
for i in $(seq 1 80); do
  status=$($VAST show instances --raw 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print([i['actual_status'] for i in d if i['id']==${INSTANCE}][0])" 2>/dev/null || echo "unknown")
  if [ "$status" = "running" ]; then
    echo "  running!"
    break
  fi
  echo "  ($i/80) status=$status, retry 15s..."
  sleep 15
done

if [ "${status:-}" != "running" ]; then
  echo "Instance did not start. Check: $VAST show instances"
  exit 1
fi

SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=10 -o StrictHostKeyChecking=accept-new -p $PORT root@$HOST"

echo "== waiting for sshd =="
for i in $(seq 1 60); do
  nc -z -G 5 "$HOST" "$PORT" 2>/dev/null && break
  echo "  ($i/60) sshd not ready, retry 10s..."
  sleep 10
done
$SSH 'true' || { echo "ssh never came up"; exit 1; }

echo "== sync repo =="
$SSH 'cd /root/crowd-finetune && git fetch -q origin && git reset --hard -q origin/main && git log --oneline -1'

echo "== sync data files =="
rsync -az -e "ssh -p $PORT" \
  "$LOCALREPO/data/toolcall_sft.jsonl" \
  "$LOCALREPO/.env" \
  root@$HOST:/root/crowd-finetune/

echo "== install deps =="
$SSH 'cd /root/crowd-finetune && python -m pip install -q -r training/reasoning/requirements.txt peft transformers datasets accelerate bitsandbytes'

echo "== train 7B LoRA on toolcall data (499 rows, 3 epochs) =="
$SSH 'cd /root/crowd-finetune && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python -m training.train_lora \
  --data data/toolcall_sft.jsonl \
  --base Qwen/Qwen2.5-7B-Instruct \
  --out outputs/toolcall-7b-lora \
  --epochs 3 \
  --max-len 512 \
  --batch 1 \
  --grad-accum 16 \
  --lr 1e-4'

echo "== eval on toolcall bench =="
$SSH 'cd /root/crowd-finetune && python -m training.eval_toolcall \
  --model outputs/toolcall-7b-lora \
  --out data/toolcall_7b_eval.json'

echo "== pull results =="
rsync -az -e "ssh -p $PORT" \
  root@$HOST:/root/crowd-finetune/data/toolcall_7b_eval.json \
  "$LOCALREPO/data/"

echo "== serve checkpoint =="
$SSH 'cd /root/crowd-finetune && nohup bash training/reasoning/serve.sh \
  MODEL=outputs/toolcall-7b-lora \
  PORT=8001 \
  > /root/vllm_serve.log 2>&1 &'

echo ""
echo "=== DEPLOYED ==="
echo "Checkpoint: outputs/toolcall-7b-lora"
echo "Eval:       data/toolcall_7b_eval.json"
echo "Monitor:    $SSH 'tail -f /root/vllm_serve.log'"
echo "Test:       curl http://$HOST:8001/v1/models"

#!/usr/bin/env bash
set -euo pipefail

LOCALREPO="/Users/sushanthtiruvaipati/Documents/Default Project/crowd-finetune"
INSTANCE="51027615"
HOST="ssh7.vast.ai"
PORT="27614"
VAST="/tmp/pv/bin/vastai"
REPO_PATH="/workspace/coderepair"

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

echo "== ensure repo exists =="
$SSH "if [ ! -d $REPO_PATH/.git ]; then git clone -q https://github.com/tsushanth/crowd-finetune.git $REPO_PATH; else cd $REPO_PATH && git fetch -q origin && git reset --hard -q origin/main; fi"
$SSH "cd $REPO_PATH && git log --oneline -1"

echo "== probe vllm =="
$SSH "
if [ -x /workspace/vllmvenv/bin/vllm ]; then
  echo 'vllmvenv present'
elif /workspace/vllmvenv/bin/python -c 'import vllm' 2>/dev/null; then
  echo 'vllmvenv present (python)'
else
  echo 'creating vllmvenv'
  python3 -m venv /workspace/vllmvenv && /workspace/vllmvenv/bin/pip install -q -U pip &&
  /workspace/vllmvenv/bin/pip install -q 'vllm==0.29.0' 'flashinfer-python==0.6.18'
  echo 'vllmvenv created'
fi"
$SSH "/workspace/vllmvenv/bin/python -c 'import vllm' 2>&1 | tail -1 || true"

echo "== sync data files =="
rsync -az -e "ssh -p $PORT" \
  "$LOCALREPO/data/toolcall_sft.jsonl" \
  "$LOCALREPO/data/toolcall_corpus.jsonl" \
  "$LOCALREPO/data/bench.jsonl" \
  "$LOCALREPO/.env" \
  root@$HOST:$REPO_PATH/

echo "== install deps =="
$SSH "cd $REPO_PATH && python3 -m pip install -q peft transformers accelerate"

echo "== train 7B LoRA on toolcall data (499 rows, 3 epochs) =="
$SSH "cd $REPO_PATH && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python3 -m training.train_lora \
  --data data/toolcall_sft.jsonl \
  --base Qwen/Qwen2.5-7B-Instruct \
  --out outputs/toolcall-7b-lora \
  --epochs 3 \
  --max-len 512 \
  --batch 1 \
  --grad-accum 8 \
  --lr 1e-4"

echo "== merge LoRA -> dense checkpoint =="
$SSH "cd $REPO_PATH && python3 -m training.reasoning.merge \
  --base Qwen/Qwen2.5-7B-Instruct \
  --adapter outputs/toolcall-7b-lora \
  --output $REPO_PATH/outputs/toolcall-7b-lora-merged"

echo "== kill any existing vllm =="
$SSH "pkill -f 'vllm serve' 2>/dev/null || true; sleep 2"

echo "== start vLLM serve (merged 7B) =="
$SSH "cd $REPO_PATH && setsid nohup env VLLM_USE_FLASHINFER_SAMPLER=0 \
  /workspace/vllmvenv/bin/vllm serve outputs/toolcall-7b-lora-merged \
  --port 8001 --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --dtype bfloat16 > /workspace/vllm_serve.log 2>&1 < /dev/null &"

echo "== wait for vLLM health (up to 120s) =="
for i in $(seq 1 24); do
  sleep 5
  if $SSH "curl -s -m 3 http://127.0.0.1:8001/health" 2>/dev/null | grep -q ok; then
    echo "  vLLM healthy!"
    break
  fi
  echo "  ($i/24) waiting..."
done

echo "== bench throughput =="
$SSH "cd $REPO_PATH && /workspace/vllmvenv/bin/python training/bench_serve.py \
  --model toolcall-7b-lora --bench data/toolcall_corpus.jsonl \
  --gpu-rate 1.0 --concurrency 8 --repeats 4 --max-new 64" 2>&1 | tee /tmp/vllm_bench_7b.log

echo "== pull bench results =="
rsync -az -e "ssh -p $PORT" root@$HOST:$REPO_PATH/data/evals/ "$LOCALREPO/data/evals_7b/" 2>/dev/null || true

echo ""
echo "=== DEPLOYED ==="
echo "Adapter:    outputs/toolcall-7b-lora"
echo "Merged:     outputs/toolcall-7b-lora-merged"
echo "Bench log:  /tmp/vllm_bench_7b.log"
echo "Monitor:    $SSH 'tail -f /workspace/vllm_serve.log'"
echo "Test:       curl http://$HOST:8001/v1/models"

#!/usr/bin/env bash
# Resume-from-exited toolcall-7b ops box 51027615 (port 27614) and finish the
# 7B LoRA -> merge -> serve -> bench pipeline, then pull bench + eval to local.
set -euo pipefail

P="51027615"
KEY="$HOME/.ssh/runpod_cuda"
HOST="ssh7.vast.ai"
PORT="27614"
VAST="/tmp/pv/bin/vastai"

echo "== start $P =="
$VAST start instance "$P" 2>&1 || true

echo "== poll for running (up to 20 min) =="
for i in $(seq 1 80); do
  st=$($VAST show instances --raw 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print([i['actual_status'] for i in d if i['id']==${P}][0])" 2>/dev/null || echo unknown)
  echo "  ($i/80) status=$st"
  [ "$st" = "running" ] && { echo "  READY"; break; }
  sleep 15
done
[ "$st" = "running" ] || { echo "did not start"; exit 1; }

SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=25 -p $PORT root@$HOST"
for i in $(seq 1 40); do
  sleep 15
  $SSH 'true' 2>/dev/null && { echo "  sshd up!"; break; }
  echo "  ($i/40) sshd not ready..."; sleep 10
done

R="/workspace/coderepair"
echo "== repo + adapter state on restart =="
$SSH "cd $R && git log --oneline -1; ls -d outputs/toolcall-7b-lora 2>/dev/null && find outputs/toolcall-7b-lora -name 'adapter_model.safetensors' -exec du -h {} \; 2>/dev/null || echo '  no adapter yet'"

echo "== resume/finish 7B LoRA (499 rows x3ep, 8bit) — nohup =="
$SSH "cd $R && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m training.train_lora \
  --data data/toolcall_sft.jsonl \
  --base Qwen/Qwen2.5-7B-Instruct \
  --out outputs/toolcall-7b-lora \
  --epochs 3 --max-len 512 --batch 1 --grad-accum 8 --lr 1e-4 \
  > /workspace/train_7b_resume.log 2>&1 < /dev/null & echo train_pid=\$!; sleep 6; tail -4 /workspace/train_7b_resume.log"

echo "== poll for adapter appearing (up to 4h) =="
for i in $(seq 1 420); do
  sleep 30
  n=$($SSH "cd $R && find outputs/toolcall-7b-lora -name 'adapter_model.safetensors' 2>/dev/null | wc -l" 2>/dev/null || echo 0)
  last=$($SSH "tail -2 /workspace/train_7b_resume.log 2>/dev/null | tr '\n' ' '" 2>/dev/null || echo '')
  echo "  ($i/420) adapters=$n | $last"
  [ "$n" -ge 1 ] && { echo "  ADAPTER READY"; break; }
done

echo "== merge LoRA -> dense =="
$SSH "cd $R && python3 -m training.reasoning.merge \
  --base Qwen/Qwen2.5-7B-Instruct \
  --adapter outputs/toolcall-7b-lora \
  --output $R/outputs/toolcall-7b-lora-merged" 2>&1 | tail -3
$SSH "ls -d $R/outputs/toolcall-7b-lora-merged && echo merged-ok"

echo "== serve merged via vLLM (flashinfer sampler OFF) on :8001 =="
$SSH "pkill -f 'vllm serve' 2>/dev/null || true; sleep 2"
$SSH "cd $R && setsid nohup env VLLM_USE_FLASHINFER_SAMPLER=0 \
  /workspace/vllmvenv/bin/vllm serve outputs/toolcall-7b-lora-merged \
  --port 8001 --served-model-name toolcall-7b-lora \
  --max-model-len 4096 --gpu-memory-utilization 0.90 --dtype bfloat16 \
  > /workspace/vllm_7b.log 2>&1 < /dev/null &"

echo "== wait vLLM health (up to 4 min) =="
for i in $(seq 1 24); do
  sleep 10
  $SSH "curl -s -m 3 http://127.0.0.1:8001/health" 2>/dev/null | grep -q ok && { echo "  vLLM up!"; break; }
  echo "  ($i/24) waiting..."
done

echo "== bench_serve 7B (throughput + cost/1K) =="
$SSH "cd $R && /workspace/vllmvenv/bin/python training/bench_serve.py \
  --model toolcall-7b-lora --bench data/toolcall_corpus.jsonl \
  --gpu-rate 1.0 --concurrency 8 --repeats 4 --max-new 64" 2>&1 | tee /tmp/bench_7b_final.log

echo "== eval bench-56 + full-555 (DB_PATH=toolcall.db) =="
$SSH "cd $R && env DB_PATH=$R/data/toolcall.db python3 -m training.eval_toolcall \
  --bench data/bench.jsonl --samples 56 \
  --adapter outputs/toolcall-7b-lora --competitor --gpu-rate 1.0 \
  --local-tokens-per-sec $(grep -oE 'tokens/sec[a-z ]*[0-9.]+' /tmp/bench_7b_final.log | head -1 || echo 380)"
$SSH "cd $R && env DB_PATH=$R/data/toolcall.db python3 -m training.eval_toolcall \
  --bench data/toolcall_corpus.jsonl --samples 555 \
  --adapter outputs/toolcall-7b-lora --competitor --gpu-rate 1.0 \
  --local-tokens-per-sec $(grep -oE '[0-9.]+' /tmp/bench_7b_final.log | head -1 || echo 380)"

echo "== pull eval db + merged =="
rsync -az -e "ssh -i $KEY -p $PORT" root@$HOST:/workspace/coderepair/data/toolcall.db "$LOCAL/data/toolcall_7b.db" 2>/dev/null || true

echo "DONE 7B pipeline"

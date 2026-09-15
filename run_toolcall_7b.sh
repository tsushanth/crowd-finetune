#!/usr/bin/env bash
set -euo pipefail
# Run 7B toolcall LoRA end-to-end on toolcall-7b box (51027615) via ssh7.vast.ai:27614

KEY="$HOME/.ssh/runpod_cuda"
HOST="ssh7.vast.ai"
PORT="27614"
H="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $PORT"
R="root@$HOST"
REPO="/workspace/coderepair"

echo "== HEAD =="
$H $R "cd $REPO && git log --oneline -1 && grep -E 'VLLM_USE|no.*vllm|no-vllm' deploy_toolcall_7b.sh | head -3"

echo "== ensure peft/transformers/accelerate =="
$H $R "cd $REPO && python3 -m pip install -q -U peft transformers accelerate"

echo "== train 7B LoRA (499 SFT rows, 3 epochs, 8-bit) =="
$H $R "cd $REPO && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
python3 -m training.train_lora --data data/toolcall_sft.jsonl \
  --base Qwen/Qwen2.5-7B-Instruct --out outputs/toolcall-7b-lora \
  --epochs 3 --max-len 512 --batch 1 --grad-accum 8 --lr 1e-4" 2>&1 | tee /tmp/train_7b.log | tail -5

echo "== merge LoRA -> dense =="
$H $R "cd $REPO && python3 -m training.reasoning.merge \
  --base Qwen/Qwen2.5-7B-Instruct --adapter outputs/toolcall-7b-lora \
  --output $REPO/outputs/toolcall-7b-lora-merged"

echo "== merged exists? =="
$H $R "cd $REPO && ls -la outputs/toolcall-7b-lora-merged | head -4"

echo "== eval tuned vs base vs gpt-4o-mini vs haiku (bench-56 + full-555) =="
$H $R "cd $REPO && env DB_PATH=$REPO/data/toolcall.db \
python3 -m training.eval_toolcall --bench data/bench.jsonl --samples 56 \
  --adapter outputs/toolcall-7b-lora --competitor --gpu-rate 1.0 \
  --local-tokens-per-sec 420" 2>&1 | tee /tmp/eval_56_7b.log | tail -3
$H $R "cd $REPO && env DB_PATH=$REPO/data/toolcall.db \
python3 -m training.eval_toolcall --bench data/toolcall_corpus.jsonl --samples 200 \
  --adapter outputs/toolcall-7b-lora --competitor --gpu-rate 1.0 \
  --local-tokens-per-sec 420" 2>&1 | tee /tmp/eval_full_7b.log | tail -3

echo "== pull eval data =="
rsync -az -e "ssh -i $KEY -p $PORT" $R:/workspace/coderepair/data/toolcall.db \
  "/Users/sushanthtiruvaipati/Documents/Default Project/crowd-finetune/data/toolcall_7b.db"

echo ""
echo "=== DONE ==="
echo "Adapter:  outputs/toolcall-7b-lora (merged -> outputs/toolcall-7b-lora-merged)"
echo "Logs:     /tmp/train_7b.log /tmp/eval_56_7b.log /tmp/eval_full_7b.log"

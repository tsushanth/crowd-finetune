#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

BASE="${BASE:-Qwen/Qwen2.5-3B-Instruct}"
OUT="outputs"
GRPO_LIMIT="${GRPO_LIMIT:-128}"
GRPO_MAX="${GRPO_MAX_COMPLETION:-1024}"
GRPO_GENS="${GRPO_GENS:-8}"
GRPO_BATCH="${GRPO_BATCH:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SFT_DIR="$SCDIR/$OUT/code-repair-sft"
SFT_MERGED="$SCDIR/$OUT/code-repair-sft-merged"
GRPO_DIR="$SCDIR/$OUT/code-repair-grpo"
GRPO_MERGED="$SCDIR/$OUT/code-repair-grpo-merged"

echo "[1/8] deps"
python -m pip install -q -r training/coderepair/requirements.txt

echo "[2/8] repair SFT set present: data/code_repair_sft.jsonl ($(wc -l < training/coderepair/data/code_repair_sft.jsonl) rows)"

echo "[3/8] SFT on repair traces"
python -m training.coderepair.train_sft --base "$BASE" --data data/code_repair_sft.jsonl --output "$SFT_DIR" --device auto

echo "[4/8] merge SFT adapter"
python -m training.reasoning.merge \
  --base "$BASE" \
  --adapter "$SFT_DIR" \
  --output "$SFT_MERGED"

if [ -z "${SKIP_GRPO:-}" ]; then
echo "[5/8] GRPO RL on repair prompts (unit-test reward, ${GRPO_LIMIT} prompts)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m training.coderepair.train_grpo \
    --base "$SFT_MERGED" \
    --data data/code_repair_sft.jsonl \
    --limit "$GRPO_LIMIT" --gens "$GRPO_GENS" --batch "$GRPO_BATCH" --max-completion "$GRPO_MAX" \
    --device auto

  GRPO_CKPT="$(ls -d "$GRPO_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)"
  test -n "$GRPO_CKPT"

  echo "[6/8] merge GRPO ($GRPO_CKPT)"
  python -m training.reasoning.merge \
    --base "$SFT_MERGED" \
    --adapter "$GRPO_CKPT" \
    --output "$GRPO_MERGED"
fi

echo "[7/8] held-out pass@1"
echo "-- repair eval (buggy fn + failing tests -> fix):"
python -m training.coderepair.eval_code --data data/code_repair_eval.jsonl --model "$BASE" --out data/code_repair_eval_base.json
python -m training.coderepair.eval_code --data data/code_repair_eval.jsonl --model "$SFT_MERGED" --out data/code_repair_eval_sft.json
if [ -z "${SKIP_GRPO:-}" ]; then
  python -m training.coderepair.eval_code --data data/code_repair_eval.jsonl --model "$GRPO_MERGED" --out data/code_repair_eval_grpo.json
fi
echo "-- generate eval (docstring only, no regression check):"
python -m training.coderepair.eval_code --data data/code_eval.jsonl --model "$BASE" --out data/code_eval_base.json
python -m training.coderepair.eval_code --data data/code_eval.jsonl --model "$SFT_MERGED" --out data/code_eval_sft.json
if [ -z "${SKIP_GRPO:-}" ]; then
  python -m training.coderepair.eval_code --data data/code_eval.jsonl --model "$GRPO_MERGED" --out data/code_eval_grpo.json
fi
echo "-- MBPP generate eval (out-of-domain):"
python -m training.coderepair.eval_code --data data/code_mbpp_eval.jsonl --model "$BASE" --out data/code_mbpp_eval_base.json
python -m training.coderepair.eval_code --data data/code_mbpp_eval.jsonl --model "$SFT_MERGED" --out data/code_mbpp_eval_sft.json
if [ -z "${SKIP_GRPO:-}" ]; then
  python -m training.coderepair.eval_code --data data/code_mbpp_eval.jsonl --model "$GRPO_MERGED" --out data/code_mbpp_eval_grpo.json
fi

echo "[8/8] competitor baseline on repair eval"
python -m training.coderepair.eval_code --data data/code_repair_eval.jsonl --model "$SFT_MERGED" --competitor --out data/code_repair_eval_table.json

echo "DONE. Compare the repair pass@1 columns."
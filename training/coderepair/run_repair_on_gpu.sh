#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

BASE="${BASE:-Qwen/Qwen2.5-3B-Instruct}"
OUT="outputs"
SEED="${SEED:-42}"
WANDB_ARGS=""
if [ -n "${WANDB:-}" ]; then WANDB_ARGS="--wandb"; fi
GRPO_LIMIT="${GRPO_LIMIT:-128}"
GRPO_MAX="${GRPO_MAX_COMPLETION:-1024}"
GRPO_GENS="${GRPO_GENS:-8}"
GRPO_BATCH="${GRPO_BATCH:-1}"
SFT_BATCH="${SFT_BATCH:-2}"
SFT_GA="${SFT_GA:-16}"
SFT_SEQ="${SFT_SEQ:-2048}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SFT_DIR="$SCDIR/$OUT/code-repair-sft"
SFT_MERGED="$SCDIR/$OUT/code-repair-sft-merged"
GRPO_DIR="$SCDIR/$OUT/code-repair-grpo"
GRPO_MERGED="$SCDIR/$OUT/code-repair-grpo-merged"
DPO_DIR="$SCDIR/$OUT/code-repair-dpo"
DPO_MERGED="$SCDIR/$OUT/code-repair-dpo-merged"
ROLLOUT_LIMIT="${ROLLOUT_LIMIT:-128}"
ROLLOUT_GENS="${ROLLOUT_GENS:-8}"

echo "[1/10] deps"
python -m pip install -q -r training/coderepair/requirements.txt

echo "[2/10] repair SFT set present: data/code_repair_sft.jsonl ($(wc -l < training/coderepair/data/code_repair_sft.jsonl) rows)"

echo "[3/10] SFT on repair traces"
python -m training.coderepair.train_sft \
  --base "$BASE" --data data/code_repair_sft.jsonl --output "$SFT_DIR" \
  --device auto --batch "$SFT_BATCH" --grad-accum "$SFT_GA" --seq-length "$SFT_SEQ" \
  --seed "$SEED" $WANDB_ARGS

echo "[4/10] merge SFT adapter"
python -m training.reasoning.merge \
  --base "$BASE" \
  --adapter "$SFT_DIR" \
  --output "$SFT_MERGED"

if [ -z "${SKIP_GRPO:-}" ]; then
echo "[5/10] GRPO RL on repair prompts (unit-test reward, ${GRPO_LIMIT} prompts)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m training.coderepair.train_grpo \
    --base "$SFT_MERGED" \
    --data data/code_repair_sft.jsonl \
    --limit "$GRPO_LIMIT" --gens "$GRPO_GENS" --batch "$GRPO_BATCH" --max-completion "$GRPO_MAX" \
    --device auto --seed "$SEED" $WANDB_ARGS

  GRPO_CKPT="$(ls -d "$GRPO_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)"
  test -n "$GRPO_CKPT"

  echo "[6/10] merge GRPO ($GRPO_CKPT)"
  python -m training.reasoning.merge \
    --base "$SFT_MERGED" \
    --adapter "$GRPO_CKPT" \
    --output "$GRPO_MERGED"
fi

if [ -z "${SKIP_DPO:-}" ]; then
  if [ -d "$GRPO_MERGED" ]; then
    ROLLOUT_MODEL="$GRPO_MERGED"
  else
    ROLLOUT_MODEL="$SFT_MERGED"
  fi

  echo "[7/10] collect rollouts (${ROLLOUT_LIMIT} prompts x ${ROLLOUT_GENS}) from ${ROLLOUT_MODEL}"
  python -m training.coderepair.collect_rollouts \
    --base "$ROLLOUT_MODEL" \
    --data data/code_repair_eval.jsonl \
    --limit "$ROLLOUT_LIMIT" --gens "$ROLLOUT_GENS" \
    --seed "$SEED" --output "data/repair_rollouts.jsonl"

  echo "[7.5/10] build DPO pairs"
  python -m training.coderepair.make_dpo \
    --input "data/repair_rollouts.jsonl" --output "data/repair_dpo_pairs.jsonl"

  echo "[8/10] DPO on SFT checkpoint"
  python -m training.coderepair.train_dpo \
    --base "$SFT_MERGED" \
    --data "data/repair_dpo_pairs.jsonl" \
    --output "$DPO_DIR" \
    --seed "$SEED" $WANDB_ARGS

  DPO_CKPT="$(ls -d "$DPO_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)"
  test -n "$DPO_CKPT"

  echo "[8.5/10] merge DPO ($DPO_CKPT)"
  python -m training.reasoning.merge \
    --base "$SFT_MERGED" \
    --adapter "$DPO_CKPT" \
    --output "$DPO_MERGED"
fi

echo "[9/10] held-out pass@1"
echo "-- repair eval (buggy fn + failing tests -> fix):"
python -m training.coderepair.eval_code --data data/code_repair_eval.jsonl --model "$BASE" --out data/code_repair_eval_base.json
python -m training.coderepair.eval_code --data data/code_repair_eval.jsonl --model "$SFT_MERGED" --out data/code_repair_eval_sft.json
if [ -z "${SKIP_GRPO:-}" ]; then
  python -m training.coderepair.eval_code --data data/code_repair_eval.jsonl --model "$GRPO_MERGED" --out data/code_repair_eval_grpo.json
fi
if [ -z "${SKIP_DPO:-}" ]; then
  python -m training.coderepair.eval_code --data data/code_repair_eval.jsonl --model "$DPO_MERGED" --out data/code_repair_eval_dpo.json
fi

echo "-- generate eval (docstring only, no regression check):"
python -m training.coderepair.eval_code --data data/code_eval.jsonl --model "$BASE" --out data/code_eval_base.json
python -m training.coderepair.eval_code --data data/code_eval.jsonl --model "$SFT_MERGED" --out data/code_eval_sft.json
if [ -z "${SKIP_GRPO:-}" ]; then
  python -m training.coderepair.eval_code --data data/code_eval.jsonl --model "$GRPO_MERGED" --out data/code_eval_grpo.json
fi
if [ -z "${SKIP_DPO:-}" ]; then
  python -m training.coderepair.eval_code --data data/code_eval.jsonl --model "$DPO_MERGED" --out data/code_eval_dpo.json
fi

echo "-- MBPP generate eval (out-of-domain):"
python -m training.coderepair.eval_code --data data/code_mbpp_eval.jsonl --model "$BASE" --out data/code_mbpp_eval_base.json
python -m training.coderepair.eval_code --data data/code_mbpp_eval.jsonl --model "$SFT_MERGED" --out data/code_mbpp_eval_sft.json
if [ -z "${SKIP_GRPO:-}" ]; then
  python -m training.coderepair.eval_code --data data/code_mbpp_eval.jsonl --model "$GRPO_MERGED" --out data/code_mbpp_eval_grpo.json
fi
if [ -z "${SKIP_DPO:-}" ]; then
  python -m training.coderepair.eval_code --data data/code_mbpp_eval.jsonl --model "$DPO_MERGED" --out data/code_mbpp_eval_dpo.json
fi

echo "[10/10] competitor baseline on repair eval"
python -m training.coderepair.eval_code --data data/code_repair_eval.jsonl --model "$SFT_MERGED" --competitor --out data/code_repair_eval_table.json

echo "DONE. Compare the repair/generate/MBPP pass@1 columns."

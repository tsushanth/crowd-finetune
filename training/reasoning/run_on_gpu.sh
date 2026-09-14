#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

BASE="${BASE:-Qwen/Qwen2.5-3B-Instruct}"
DATA="${DATA:-data/reasoning_sft.jsonl}"
SEED="${SEED:-42}"
OUT="outputs"
WANDB_ARGS=""
if [ -n "${WANDB:-}" ]; then WANDB_ARGS="--wandb"; fi

echo "[1/8] deps"
python -m pip install -q -r training/reasoning/requirements.txt

echo "[2/8] SFT on distilled traces (${DATA}) seed=${SEED}"
python -m training.reasoning.train_sft --base "$BASE" --data "$DATA" --seed "$SEED" $WANDB_ARGS

echo "[3/8] merge SFT adapter"
python -m training.reasoning.merge \
  --base "$BASE" \
  --adapter "$OUT/reasoning-sft" \
  --output "$OUT/reasoning-sft-merged"

GRPO_LIMIT="${GRPO_LIMIT:-400}"
GRPO_MAX="${GRPO_MAX_COMPLETION:-1024}"

if [ -z "${SKIP_GRPO:-}" ]; then
  echo "[4/8] GRPO RL (${GRPO_LIMIT} prompts) on merged SFT checkpoint seed=${SEED}"
  python -m training.reasoning.train_grpo \
    --base "$OUT/reasoning-sft-merged" \
    --limit "$GRPO_LIMIT" --gens 8 --batch 4 --max-completion "$GRPO_MAX" \
    --seed "$SEED" $WANDB_ARGS

  GRPO_CKPT="$(ls -d "$SCDIR/$OUT/reasoning-grpo"/checkpoint-* 2>/dev/null | sort -V | tail -1)"
  test -n "$GRPO_CKPT"

  echo "[5/8] merge GRPO ($GRPO_CKPT)"
  python -m training.reasoning.merge \
    --base "$OUT/reasoning-sft-merged" \
    --adapter "$GRPO_CKPT" \
    --output "$OUT/reasoning-grpo-merged"
fi

if [ -z "${SKIP_DPO:-}" ]; then
  if [ -d "$SCDIR/$OUT/reasoning-grpo-merged" ]; then
    ROLLOUT_MODEL="$OUT/reasoning-grpo-merged"
  else
    ROLLOUT_MODEL="$OUT/reasoning-sft-merged"
  fi
  ROLLOUT_LIMIT="${ROLLOUT_LIMIT:-128}"
  ROLLOUT_GENS="${ROLLOUT_GENS:-8}"

  echo "[6/8] collect rollouts (${ROLLOUT_LIMIT} prompts x ${ROLLOUT_GENS}) from ${ROLLOUT_MODEL}"
  python -m training.reasoning.collect_rollouts \
    --base "$ROLLOUT_MODEL" \
    --limit "$ROLLOUT_LIMIT" --gens "$ROLLOUT_GENS" \
    --seed "$SEED" --output "data/grpo_rollouts.jsonl"

  echo "[6.5/8] build DPO pairs"
  python -m training.reasoning.make_dpo \
    --input "data/grpo_rollouts.jsonl" --output "data/dpo_pairs.jsonl"

  echo "[7/8] DPO on SFT checkpoint seed=${SEED}"
  python -m training.reasoning.train_dpo \
    --base "$OUT/reasoning-sft-merged" \
    --data "data/dpo_pairs.jsonl" \
    --seed "$SEED" $WANDB_ARGS

  DPO_CKPT="$(ls -d "$SCDIR/$OUT/reasoning-dpo"/checkpoint-* 2>/dev/null | sort -V | tail -1)"
  test -n "$DPO_CKPT"

  echo "[7.5/8] merge DPO ($DPO_CKPT)"
  python -m training.reasoning.merge \
    --base "$OUT/reasoning-sft-merged" \
    --adapter "$DPO_CKPT" \
    --output "$OUT/reasoning-dpo-merged"
fi

echo "[8/9] held-out eval: base vs SFT vs GRPO vs DPO (GSM8K, limit ${EVAL_LIMIT:-100})"
python -m training.reasoning.eval_judge --model "$BASE"                         --limit "${EVAL_LIMIT:-100}" --output "data/eval_base.json"
python -m training.reasoning.eval_judge --model "$OUT/reasoning-sft-merged"    --limit "${EVAL_LIMIT:-100}" --output "data/eval_sft.json"
if [ -z "${SKIP_GRPO:-}" ]; then
  python -m training.reasoning.eval_judge --model "$OUT/reasoning-grpo-merged"   --limit "${EVAL_LIMIT:-100}" --output "data/eval_grpo.json"
fi
if [ -z "${SKIP_DPO:-}" ]; then
  python -m training.reasoning.eval_judge --model "$OUT/reasoning-dpo-merged"   --limit "${EVAL_LIMIT:-100}" --output "data/eval_dpo.json"
fi

echo "[9/9] held-out eval (MATH-500, limit ${MATH_LIMIT:-100}, strict)"
python -m training.reasoning.eval_judge --model "$BASE"                         --dataset HuggingFaceH4/MATH-500 --limit "${MATH_LIMIT:-100}" --strict --output "data/eval_base_math.json"
python -m training.reasoning.eval_judge --model "$OUT/reasoning-sft-merged"    --dataset HuggingFaceH4/MATH-500 --limit "${MATH_LIMIT:-100}" --strict --output "data/eval_sft_math.json"
if [ -z "${SKIP_GRPO:-}" ]; then
  python -m training.reasoning.eval_judge --model "$OUT/reasoning-grpo-merged"   --dataset HuggingFaceH4/MATH-500 --limit "${MATH_LIMIT:-100}" --strict --output "data/eval_grpo_math.json"
fi
if [ -z "${SKIP_DPO:-}" ]; then
  python -m training.reasoning.eval_judge --model "$OUT/reasoning-dpo-merged"   --dataset HuggingFaceH4/MATH-500 --limit "${MATH_LIMIT:-100}" --strict --output "data/eval_dpo_math.json"
fi

python - <<'PY'
import json
from pathlib import Path

out = Path("training/reasoning/data")
for name in sorted(p.name for p in out.glob("eval_*.json")):
    d = json.loads((out / name).read_text())
    met = d.get("metrics", {})
    print(
        f"{name}: acc={d['accuracy']:.4f} n={d['n']} "
        f"tokens={met.get('avg_len_tokens')} ppl={met.get('ppl')} "
        f"rm={ {k: v for k, v in met.items() if k.startswith('rm_')} }"
    )
PY

echo "DONE. Compare the eval_*.json accuracies + metrics."
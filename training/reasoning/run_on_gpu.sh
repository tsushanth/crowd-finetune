#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

BASE="${BASE:-Qwen/Qwen2.5-3B-Instruct}"
DATA="${DATA:-data/reasoning_sft.jsonl}"
OUT="outputs"
SFT_BATCH="${SFT_BATCH:-2}"
SFT_SEQ="${SFT_SEQ:-1024}"

echo "[1/6] deps"
python -m pip install -q -r training/reasoning/requirements.txt

echo "[2/6] SFT on distilled traces (${DATA})"
python -m training.reasoning.train_sft --base "$BASE" --data "$DATA" --batch "$SFT_BATCH" --grad-accum 16 --seq-length "$SFT_SEQ"

echo "[3/6] merge SFT adapter"
python -m training.reasoning.merge \
  --base "$BASE" \
  --adapter "$OUT/reasoning-sft" \
  --output "$OUT/reasoning-sft-merged"

GRPO_LIMIT="${GRPO_LIMIT:-400}"
GRPO_MAX="${GRPO_MAX_COMPLETION:-1024}"

if [ -z "${SKIP_GRPO:-}" ]; then
  echo "[4/6] GRPO RL (${GRPO_LIMIT} prompts) on merged SFT checkpoint"
  python -m training.reasoning.train_grpo \
    --base "$OUT/reasoning-sft-merged" \
    --limit "$GRPO_LIMIT" --gens 8 --batch 4 --max-completion "$GRPO_MAX"

  GRPO_CKPT="$(ls -d "$SCDIR/$OUT/reasoning-grpo"/checkpoint-* 2>/dev/null | sort -V | tail -1)"
  test -n "$GRPO_CKPT"

  echo "[5/6] merge GRPO ($GRPO_CKPT)"
  python -m training.reasoning.merge \
    --base "$OUT/reasoning-sft-merged" \
    --adapter "$GRPO_CKPT" \
    --output "$OUT/reasoning-grpo-merged"
fi

echo "[6/6] held-out eval: base vs SFT vs GRPO (GSM8K, limit 100)"
python -m training.reasoning.eval_judge --model "$BASE"                         --limit 100 --output "data/eval_base.json"
python -m training.reasoning.eval_judge --model "$OUT/reasoning-sft-merged"    --limit 100 --output "data/eval_sft.json"
if [ -z "${SKIP_GRPO:-}" ]; then
  python -m training.reasoning.eval_judge --model "$OUT/reasoning-grpo-merged"   --limit 100 --output "data/eval_grpo.json"
fi

echo "[7/7] held-out eval: base vs SFT vs GRPO (MATH-500, limit ${MATH_LIMIT:-100}, strict)"
python -m training.reasoning.eval_judge --model "$BASE"                         --dataset HuggingFaceH4/MATH-500 --limit "${MATH_LIMIT:-100}" --strict --output "data/eval_base_math.json"
python -m training.reasoning.eval_judge --model "$OUT/reasoning-sft-merged"    --dataset HuggingFaceH4/MATH-500 --limit "${MATH_LIMIT:-100}" --strict --output "data/eval_sft_math.json"
if [ -z "${SKIP_GRPO:-}" ]; then
  python -m training.reasoning.eval_judge --model "$OUT/reasoning-grpo-merged"   --dataset HuggingFaceH4/MATH-500 --limit "${MATH_LIMIT:-100}" --strict --output "data/eval_grpo_math.json"
fi

python - <<'PY'
import json
from pathlib import Path

out = Path("training/reasoning/data")
for name in ("eval_base.json", "eval_sft.json", "eval_grpo.json",
             "eval_base_math.json", "eval_sft_math.json", "eval_grpo_math.json"):
    if not (out / name).exists():
        continue
    d = json.loads((out / name).read_text())
    print(f"{name}: accuracy={d['accuracy']:.4f} n={d['n']} -> {name}")
PY

echo "DONE. Compare the six eval_*.json accuracies."
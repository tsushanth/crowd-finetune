#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

BASE="${BASE:-Qwen/Qwen2.5-3B-Instruct}"
OUT="outputs"
EVAL_SIZE="${EVAL_SIZE:-24}"
GRPO_LIMIT="${GRPO_LIMIT:-128}"
GRPO_MAX="${GRPO_MAX_COMPLETION:-1024}"

echo "[1/9] deps"
python -m pip install -q -r training/coderepair/requirements.txt

echo "[2/9] materialize dataset: train rows (tests visible) + eval rows (tests hidden)"
python -m training.coderepair.datasets --datasets "${DATASETS:-humaneval}" --eval-size "$EVAL_SIZE"

if [ -z "${SKIP_DISTILL:-}" ]; then
  echo "[3/9] teacher distill (needs BASE_LLM_API_KEY); sandbox-verify each trace"
  python -m training.coderepair.distill --dataset "${DATASETS:-humaneval}"
else
  echo "[3/9] SKIP_DISTILL set; code_sft.jsonl must already exist"
fi

echo "[4/9] SFT on test-verified traces"
python -m training.coderepair.train_sft --base "$BASE"

echo "[5/9] merge SFT adapter"
python -m training.reasoning.merge \
  --base "$BASE" \
  --adapter "$OUT/code-sft" \
  --output "$OUT/code-sft-merged"

if [ -z "${SKIP_GRPO:-}" ]; then
  echo "[6/9] GRPO RL (unit-test reward, ${GRPO_LIMIT} prompts)"
  python -m training.coderepair.train_grpo \
    --base "$OUT/code-sft-merged" \
    --limit "$GRPO_LIMIT" --gens 8 --batch 4 --max-completion "$GRPO_MAX"

  GRPO_CKPT="$(ls -d "$SCDIR/$OUT/code-grpo"/checkpoint-* 2>/dev/null | sort -V | tail -1)"
  test -n "$GRPO_CKPT"

  echo "[7/9] merge GRPO ($GRPO_CKPT)"
  python -m training.reasoning.merge \
    --base "$OUT/code-sft-merged" \
    --adapter "$GRPO_CKPT" \
    --output "$OUT/code-grpo-merged"
fi

echo "[8/9] held-out pass@1: base vs SFT vs GRPO"
python -m training.coderepair.eval_code --model "$BASE"                    --out "data/code_eval_base.json"
python -m training.coderepair.eval_code --model "$OUT/code-sft-merged"    --out "data/code_eval_sft.json"
if [ -z "${SKIP_GRPO:-}" ]; then
  python -m training.coderepair.eval_code --model "$OUT/code-grpo-merged" --out "data/code_eval_grpo.json"
fi

echo "[9/9] baseline comparison: tuned-3B vs GPT-4o-mini vs Claude Haiku"
python -m training.coderepair.eval_code --model "$OUT/code-sft-merged" --competitor --out "data/code_eval_table.json"

echo "DONE. Compare the pass@1 columns."
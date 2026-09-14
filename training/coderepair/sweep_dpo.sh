#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

BASE="${BASE:-Qwen/Qwen2.5-3B-Instruct}"
SEEDS="${SEEDS:-1 2 3}"
DATA="${DATA:-data/repair_dpo_pairs.jsonl}"
OUT="outputs"
SFT="$OUT/code-repair-sft-merged"

[ -d "$SCDIR/$SFT" ] || { echo "missing $SFT — run run_repair_on_gpu.sh first"; exit 1; }
[ -f "$SCDIR/$DATA" ] || [ -f "$SCDIR/data/repair_rollouts.jsonl" ] || {
  echo "no $DATA or data/repair_rollouts.jsonl — collect rollouts first"; exit 1;
}
[ -f "$SCDIR/$DATA" ] || {
  echo "building DPO pairs from existing rollouts"
  python -m training.coderepair.make_dpo --output data/repair_dpo_pairs.jsonl
}

for s in $SEEDS; do
  echo "== seed $s: DPO + merge =="
  python -m training.coderepair.train_dpo \
    --base "$SFT" --data "$DATA" \
    --output "outputs/code-repair-dpo-seed$s" --seed "$s"
  CKPT="$(ls -d "$SCDIR/$OUT/code-repair-dpo-seed$s"/checkpoint-* 2>/dev/null | sort -V | tail -1)"
  test -n "$CKPT"
  python -m training.reasoning.merge \
    --base "$SFT" --adapter "$CKPT" \
    --output "$OUT/code-repair-dpo-seed$s-merged"
done

echo "== eval matrix: base, SFT, DPO seed 1..3 (repair eval) =="
python -m training.coderepair.eval_code --data data/code_repair_eval.jsonl --model "$BASE" --out "data/code_repair_eval_base.json"
python -m training.coderepair.eval_code --data data/code_repair_eval.jsonl --model "$SFT" --out "data/code_repair_eval_sft.json"
for s in $SEEDS; do
  python -m training.coderepair.eval_code \
    --data data/code_repair_eval.jsonl \
    --model "$OUT/code-repair-dpo-seed$s-merged" \
    --out "data/code_repair_eval_dpo_seed$s.json"
done

python - "${SEEDS}" <<'PY'
import json
import sys
from pathlib import Path

seeds = sys.argv[1].split()
out = Path("training/coderepair/data")

def load(name):
    d = json.loads((out / name).read_text())
    rows = d.get("rows_table") or []
    for r in rows:
        if "pass" in r:
            return r["pass"], []
    return 0.0, []

models = {}
models["base"] = load("code_repair_eval_base.json")
models["sft"] = load("code_repair_eval_sft.json")
for s in seeds:
    models[f"dpo{s}"] = load(f"code_repair_eval_dpo_seed{s}.json")

print(f"\nrepair pass@1 table")
for name, (p, _) in models.items():
    print(f"  {name:6} {p:.4f}")

ps = [models[f"dpo{s}"][0] for s in seeds]
spread = max(ps) - min(ps)
print(f"\nDPO seed reproducibility: min={min(ps):.4f} max={max(ps):.4f} spread={spread:.4f}")
PY

echo "DONE. Compare code_repair_eval_{base,sft,dpo_seed*}.json pass@1."

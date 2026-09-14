#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

BASE="${BASE:-Qwen/Qwen2.5-3B-Instruct}"
SEEDS="${SEEDS:-1 2 3}"
DATA="${DATA:-data/dpo_pairs.jsonl}"
OUT="outputs"
LIMIT="${EVAL_LIMIT:-100}"
SFT="$OUT/reasoning-sft-merged"

[ -d "$SCDIR/$SFT" ] || { echo "missing $SFT — run run_on_gpu.sh first"; exit 1; }
[ -f "$SCDIR/$DATA" ] || [ -f "$SCDIR/data/grpo_rollouts.jsonl" ] || {
  echo "no $DATA or data/grpo_rollouts.jsonl — collect rollouts first"; exit 1
}
[ -f "$SCDIR/data/dpo_pairs.jsonl" ] || {
  echo "building DPO pairs from existing rollouts"
  python -m training.reasoning.make_dpo --output data/dpo_pairs.jsonl
}

for s in $SEEDS; do
  echo "== seed $s: DPO + merge =="
  python -m training.reasoning.train_dpo \
    --base "$SFT" --data "$DATA" \
    --output "outputs/reasoning-dpo-seed$s" --seed "$s"
  CKPT="$(ls -d "$SCDIR/$OUT/reasoning-dpo-seed$s"/checkpoint-* 2>/dev/null | sort -V | tail -1)"
  test -n "$CKPT"
  python -m training.reasoning.merge \
    --base "$SFT" --adapter "$CKPT" \
    --output "$OUT/reasoning-dpo-seed$s-merged"
done

echo "== eval matrix: base, SFT, DPO seed 1..3 =="
python -m training.reasoning.eval_judge --model "$BASE" --limit "$LIMIT" --output "data/eval_base.json"
python -m training.reasoning.eval_judge --model "$SFT" --limit "$LIMIT" --output "data/eval_sft.json"
for s in $SEEDS; do
  python -m training.reasoning.eval_judge \
    --model "$OUT/reasoning-dpo-seed$s-merged" \
    --limit "$LIMIT" --output "data/eval_dpo_seed$s.json"
done

python - "$LIMIT" "$SEEDS" <<'PY'
import json
import sys
from pathlib import Path

limit = int(sys.argv[1])
seeds = sys.argv[2].split()
out = Path("training/reasoning/data")

def load(name):
    d = json.loads((out / name).read_text())
    return d["accuracy"], [r["ok"] for r in d["rows"]]

models = {}
models["base"] = load("eval_base.json")
models["sft"] = load("eval_sft.json")
for s in seeds:
    models[f"dpo{s}"] = load(f"eval_dpo_seed{s}.json")

print(f"\naccuracy table (n={limit})")
for name, (acc, _) in models.items():
    print(f"  {name:6} {acc:.4f}")

accs = [models[f"dpo{s}"][0] for s in seeds]
spread = max(accs) - min(accs)
print(f"\nDPO seed reproducibility: min={min(accs):.4f} max={max(accs):.4f} spread={spread:.4f}")

names = list(models)
print("\npairwise win matrix (row model beats col model, /" + str(limit) + ")")
print(f"{'i\\j':8}" + "".join(f"{c:>8}" for c in names))
for a in names:
    row_ok = models[a][1]
    cells = []
    for b in names:
        col_ok = models[b][1]
        wins = sum(1 for x, y in zip(row_ok, col_ok) if x and not y)
        losses = sum(1 for x, y in zip(row_ok, col_ok) if not x and y)
        cells.append(f"{wins}-{losses:<6}")
    print(f"{a:8}" + "".join(f"{c:>8}" for c in cells))
PY

echo "DONE. Compare eval_base/eval_sft/eval_dpo_seed{1..3}.json accuracies."
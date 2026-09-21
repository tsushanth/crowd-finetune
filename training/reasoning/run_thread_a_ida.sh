#!/usr/bin/env bash
# Thread A: one IDA round. Round 0 already exists (742-trace SFT + GRPO-outcome champion,
# both on Hugging Face). This round: use the round-0 champion as a SELF-teacher on fresh
# GSM8K questions, merge the new verified traces with the original 742, retrain SFT+GRPO,
# and evaluate on the same fixed test sets report 6 used (so round 0 vs round 1 is a
# direct, reusable comparison -- report 6's full_grpo_outcome_merged_*.json are round 0).
#
# Run from the repo root on the GPU box. Progress in /root/progress.log; /root/fail on
# error; /root/all_done at the end. Needs /root/.hftoken.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R=training/reasoning
BASE=Qwen/Qwen2.5-3B-Instruct
step() { echo "[$(date +%H:%M:%S)] $1" >> /root/progress.log; }
fail() { echo "$1" > /root/fail; step "FAIL $1"; exit 1; }
run()  { local log=$1; shift; "$@" >> "$log" 2>&1 || fail "$log"; }
rm -f /root/fail /root/all_done; : > /root/progress.log

step "rehydrate round-0 sft adapter + grpo-outcome adapter from HF"
run /root/rehydrate.log python - <<'PYEOF'
from huggingface_hub import snapshot_download
tok = open("/root/.hftoken").read().strip()
snapshot_download("sushanth9/prm-track-sft-adapter", local_dir="training/reasoning/outputs/reasoning-sft", token=tok)
snapshot_download("sushanth9/prm-track-grpo-outcome", local_dir="training/reasoning/outputs/grpo_outcome_r0", token=tok)
snapshot_download("sushanth9/prm-track-prm", local_dir="training/reasoning/outputs/prm_real", token=tok)  # for the gaming diagnostic only
print("rehydrated")
PYEOF

step "merge round-0 champion (sft-merged, then grpo-outcome adapter on top)"
run /root/merge0.log python -m training.reasoning.merge --base $BASE --adapter outputs/reasoning-sft --output outputs/sft-merged
run /root/merge0.log python -m training.reasoning.merge --base outputs/sft-merged --adapter outputs/grpo_outcome_r0 --output outputs/round0_champion

step "fresh distillation pool (300 q, disjoint from every prior thread)"
run /root/pool.log python -m training.reasoning.pool_utils --count 300 --out data/ida_round1_pool.jsonl

step "self-distill round-1 traces from the round-0 champion"
run /root/distill.log python -m training.reasoning.distill_local --teacher outputs/round0_champion \
    --pool data/ida_round1_pool.jsonl --batch 32 --out data/ida_round1.jsonl

step "merge round-0 (742) + round-1 verified traces"
run /root/merge_traces.log python - <<'PYEOF'
import json
from training.reasoning import prm as P
rows = P.dedupe_rows(["training/reasoning/data/reasoning_sft.jsonl", "training/reasoning/data/ida_round1.jsonl"])
open("training/reasoning/data/ida_sft_round1.jsonl", "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
print(f"{len(rows)} unique traces -> ida_sft_round1.jsonl")
PYEOF

step "sft on round-1 merged traces"; run /root/sft.log python -m training.reasoning.train_sft --base $BASE --data data/ida_sft_round1.jsonl --batch 2 --grad-accum 16 --seq-length 1024 --output outputs/ida_sft_round1
run /root/sft.log python -m training.reasoning.merge --base $BASE --adapter outputs/ida_sft_round1 --output outputs/ida_round1_sft_merged

step "fresh grpo pool (400 q, further along, still disjoint)"
run /root/pool.log python -m training.reasoning.pool_utils --count 400 --skip 300 --out data/ida_round1_grpo_pool.jsonl

step "grpo round 1 (outcome reward, matching round 0's recipe)"
run /root/grpo.log python -m training.reasoning.train_grpo --base outputs/ida_round1_sft_merged --prompts-file data/ida_round1_grpo_pool.jsonl \
    --limit 400 --gens 8 --batch 4 --max-completion 1024 --reward outcome --output outputs/ida_round1_grpo
ckpt=$(ls -d $PWD/$R/outputs/ida_round1_grpo/checkpoint-* 2>/dev/null | sort -V | tail -1)
[ -n "$ckpt" ] || fail "no checkpoint for ida_round1_grpo"
run /root/grpo.log python -m training.reasoning.merge --base outputs/ida_round1_sft_merged --adapter "$ckpt" --output outputs/ida_round1_champion

step "eval round-1 champion"
run /root/eval.log python -m training.reasoning.eval_judge --model outputs/ida_round1_champion --limit 1319 --batch 96 --save-text --output data/full_ida_round1_champion_gsm8k.json
run /root/eval.log python -m training.reasoning.eval_judge --model outputs/ida_round1_champion --dataset HuggingFaceH4/MATH-500 --limit 500 --strict --batch 64 --save-text --output data/full_ida_round1_champion_math.json
run /root/diag.log python -m training.reasoning.score_eval_chains data/full_ida_round1_champion_gsm8k.json --prm outputs/prm_real --out data/diag_ida_round1_champion_gsm8k.json

step "done"; echo ALL_DONE > /root/all_done

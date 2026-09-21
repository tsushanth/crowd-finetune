#!/usr/bin/env bash
# Follow-up session: isolate the length_reward confound flagged in report 6, plus a seed-replication
# check. Reuses the SFT adapter and PRM already on Hugging Face (sushanth9/prm-track-sft-adapter,
# sushanth9/prm-track-prm) instead of retraining them, so this session skips the most expensive phase
# of the previous one (real-negative generation + PRM training, ~$1.19 last time).
#
# New arms:
#   outcome_nolr        outcome reward,       length_reward dropped, seed 42 (isolates length_reward
#                        for the outcome-only arm; compare against session 1's `outcome`)
#   prm_nolr             prm reward,           length_reward dropped, seed 42 (isolates length_reward
#                        for the PRM arm; compare against session 1's `prm`, and against outcome_nolr
#                        for the cleanest PRM-vs-exact-match comparison with no length confound at all)
#   prm_outcome_s1       prm+outcome reward,   length_reward included, seed 1  (replication check for
#                        the one signal session 1 found: prm+outcome shorter than outcome on MATH-500)
#
# Run from the repo root on the GPU box. Progress in /root/progress.log; /root/fail on error;
# /root/all_done at the end. Needs /root/.hftoken (see the session-2 message in the conversation for
# how it's placed) so it can pull the private HF adapter/PRM repos.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R=training/reasoning
BASE=Qwen/Qwen2.5-3B-Instruct
step() { echo "[$(date +%H:%M:%S)] $1" >> /root/progress.log; }
fail() { echo "$1" > /root/fail; step "FAIL $1"; exit 1; }
run()  { local log=$1; shift; "$@" >> "$log" 2>&1 || fail "$log"; }
rm -f /root/fail /root/all_done; : > /root/progress.log

step "rehydrate sft adapter + prm from HF"
run /root/rehydrate.log python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
tok = open("/root/.hftoken").read().strip()
snapshot_download("sushanth9/prm-track-sft-adapter", local_dir="training/reasoning/outputs/reasoning-sft", token=tok)
snapshot_download("sushanth9/prm-track-prm", local_dir="training/reasoning/outputs/prm_real", token=tok)
print("rehydrated")
PYEOF
step "merge sft"; run /root/sft.log python -m training.reasoning.merge --base $BASE --adapter outputs/reasoning-sft --output outputs/sft-merged
step "pool";      run /root/pool.log python -m training.reasoning.make_grpo_pool --count 400

declare -A ARGS
ARGS[outcome_nolr]="--reward outcome --no-length-reward --seed 42"
ARGS[prm_nolr]="--reward prm --no-length-reward --prm outputs/prm_real --seed 42"
ARGS[prm_outcome_s1]="--reward prm+outcome --prm outputs/prm_real --seed 1"

for name in outcome_nolr prm_nolr prm_outcome_s1; do
  step "grpo $name"
  run /root/grpo_$name.log python -m training.reasoning.train_grpo --base outputs/sft-merged --prompts-file data/grpo_pool.jsonl --limit 400 \
      --gens 8 --batch 4 --max-completion 1024 ${ARGS[$name]} --output outputs/grpo_$name
  ckpt=$(ls -d $PWD/$R/outputs/grpo_$name/checkpoint-* 2>/dev/null | sort -V | tail -1)
  [ -n "$ckpt" ] || fail "no checkpoint for $name"
  run /root/grpo_$name.log python -m training.reasoning.merge --base outputs/sft-merged --adapter "$ckpt" --output outputs/grpo_${name}_merged
done

for m in grpo_outcome_nolr_merged grpo_prm_nolr_merged grpo_prm_outcome_s1_merged; do
  step "eval $m"
  run /root/eval.log python -m training.reasoning.eval_judge --model outputs/$m --limit 1319 --batch 96 --save-text --output data/full_${m}_gsm8k.json
  run /root/eval.log python -m training.reasoning.eval_judge --model outputs/$m --dataset HuggingFaceH4/MATH-500 --limit 500 --strict --batch 64 --save-text --output data/full_${m}_math.json
  run /root/diag.log python -m training.reasoning.score_eval_chains data/full_${m}_gsm8k.json --prm outputs/prm_real --out data/diag_${m}_gsm8k.json
done
step "done"; echo ALL_DONE > /root/all_done

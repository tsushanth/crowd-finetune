#!/usr/bin/env bash
# Thread C: same pipeline, harder domain. Competition MATH instead of GSM8K -- base SFT
# scores 0.35-0.38 there (vs 0.79-0.80 on GSM8K), a much sparser outcome-reward signal,
# which is the actual condition under which process rewards are supposed to matter more.
# Train pool is nlile/hendrycks-MATH-benchmark's train split filtered to numeric answers
# (its test split is byte-identical to HuggingFaceH4/MATH-500 -- confirmed no leakage --
# so the untouched 500-question test set stays the eval target throughout).
#
# Four disjoint local slices were pre-built on the laptop (data/math_{distill,prm,grpo,val}_pool.jsonl)
# and synced to this box; math_val_pool.jsonl is unused in this pass (scope cut -- see report).
#
# Run from the repo root on the GPU box. Progress in /root/progress.log; /root/fail on
# error; /root/all_done at the end. Needs /root/.hftoken and /root/.openrouter (a labeled
# shell-sourceable file: BASE_LLM_API_KEY=... and BASE_LLM_URL=... on their own lines --
# labeled, not positional, since .env's own field order is not something to rely on).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R=training/reasoning
BASE=Qwen/Qwen2.5-3B-Instruct
step() { echo "[$(date +%H:%M:%S)] $1" >> /root/progress.log; }
fail() { echo "$1" > /root/fail; step "FAIL $1"; exit 1; }
run()  { local log=$1; shift; "$@" >> "$log" 2>&1 || fail "$log"; }
rm -f /root/fail /root/all_done; : > /root/progress.log
set -a; source /root/.openrouter; set +a

step "distill MATH traces (external teacher, deepseek-v3.2, verified against reference)"
run /root/distill.log python -m training.reasoning.distill --local-file data/math_distill_pool.jsonl \
    --limit 400 --workers 6 --teacher-model deepseek/deepseek-v3.2 --temperature 0.5 --out data/math_sft.jsonl
n=$(wc -l < training/reasoning/data/math_sft.jsonl 2>/dev/null || echo 0)
[ "$n" -ge 40 ] || fail "only $n verified MATH traces, too few to train on"

step "sft on math traces ($n verified)"
run /root/sft.log python -m training.reasoning.train_sft --base $BASE --data data/math_sft.jsonl --batch 2 --grad-accum 16 --seq-length 1024 --output outputs/math_sft
run /root/sft.log python -m training.reasoning.merge --base $BASE --adapter outputs/math_sft --output outputs/math_sft_merged

step "real-negative PRM data on MATH (800 questions, 8 rollouts/prefix)"
run /root/gen.log python -m training.reasoning.gen_real_negatives --policy $PWD/$R/outputs/math_sft_merged \
    --local-file data/math_prm_pool.jsonl --exclude data/math_sft.jsonl --questions 800 --max-wrong 500 --rollouts 8 \
    --batch 96 --output data/math_prm_real.jsonl

step "train math-specific PRM (synthetic-corruption base = math_sft.jsonl, not GSM8K)"
run /root/prm.log python -m training.reasoning.train_prm --data data/math_sft.jsonl --extra data/math_prm_real.jsonl \
    --train-layers 8 --epochs 2 --seed 42 --output outputs/math_prm

for arm in outcome prm; do
  step "grpo $arm (MATH prompts, no length_reward)"
  extra=""; [ "$arm" = "prm" ] && extra="--prm outputs/math_prm"
  run /root/grpo_$arm.log python -m training.reasoning.train_grpo --base outputs/math_sft_merged --prompts-file data/math_grpo_pool.jsonl \
      --limit 400 --gens 8 --batch 4 --max-completion 1024 --reward $arm --no-length-reward $extra --seed 42 --output outputs/math_grpo_$arm
  ckpt=$(ls -d $PWD/$R/outputs/math_grpo_$arm/checkpoint-* 2>/dev/null | sort -V | tail -1)
  [ -n "$ckpt" ] || fail "no checkpoint for math_grpo_$arm"
  run /root/grpo_$arm.log python -m training.reasoning.merge --base outputs/math_sft_merged --adapter "$ckpt" --output outputs/math_grpo_${arm}_merged
done

for m in math_sft_merged math_grpo_outcome_merged math_grpo_prm_merged; do
  step "eval $m"
  run /root/eval.log python -m training.reasoning.eval_judge --model outputs/$m --dataset HuggingFaceH4/MATH-500 --limit 500 --strict --batch 64 --save-text --output data/full_${m}_math.json
  run /root/eval.log python -m training.reasoning.eval_judge --model outputs/$m --limit 1319 --batch 96 --save-text --output data/full_${m}_gsm8k.json
  run /root/diag.log python -m training.reasoning.score_eval_chains data/full_${m}_math.json --prm outputs/math_prm --out data/diag_${m}_math.json
done

step "done"; echo ALL_DONE > /root/all_done

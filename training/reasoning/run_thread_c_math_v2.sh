#!/usr/bin/env bash
# Retry of Thread C (report 10), after fixing the two bugs that caused its 0.0 MATH-500 result:
#   1. teacher.py routed deepseek/* through a dead "R1 guard-bracket" prompt that deepseek-v3.2
#      never followed, so distilled traces came back with no <reasoning>/<answer> tags at all.
#      Fixed: all teacher calls now use the plain tag prompt every other model in this pipeline
#      already uses (formats.TEACHER_SYSTEM). Verified: 100-question local dry run went from
#      12/100 verified traces (report 10's actual number) to 77/100.
#   2. formats.exact_match only recognized plain decimal/integer answers; MATH answers expressed
#      as fractions or \frac{}{} LaTeX were discarded even when correct. Fixed with
#      formats.numeric_match (fraction/LaTeX-aware, tolerance-based), used everywhere in this
#      pipeline that used to call exact_match/extract_last_number.
#   3. New: a format-compliance gate right after SFT (check_format_compliance.py) stops the run
#      before the expensive PRM/GRPO stages if the SFT model still isn't using the tag format,
#      instead of discovering that at the very end as report 10 did.
#
# data/math_sft.jsonl (the verified traces) is pre-built locally via distill.py against the
# fixed teacher and synced in -- distillation is pure API calls, no GPU needed for that step.
# data/math_{grpo,prm}_pool.jsonl are pre-built via build_math_pool.py (widened fraction/LaTeX
# filter, same 400/800 sizes as report 10, disjoint from math_sft.jsonl and from MATH-500).
#
# Run from the repo root on the GPU box. Progress in /root/progress.log; /root/fail on error;
# /root/all_done at the end. Needs /root/.hftoken and /root/.openrouter (labeled KEY=value file).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R=training/reasoning
BASE=Qwen/Qwen2.5-3B-Instruct
step() { echo "[$(date +%H:%M:%S)] $1" >> /root/progress.log; }
fail() { echo "$1" > /root/fail; step "FAIL $1"; exit 1; }
run()  { local log=$1; shift; "$@" >> "$log" 2>&1 || fail "$log"; }
rm -f /root/fail /root/all_done; : > /root/progress.log
set -a; source /root/.openrouter; set +a

n=$(wc -l < training/reasoning/data/math_sft.jsonl 2>/dev/null || echo 0)
step "starting with $n pre-distilled, verified MATH traces (data/math_sft.jsonl)"
[ "$n" -ge 200 ] || fail "only $n verified MATH traces, expected several hundred with the fixed teacher prompt"

step "sft on math traces ($n verified)"
run /root/sft.log python -m training.reasoning.train_sft --base $BASE --data data/math_sft.jsonl --batch 2 --grad-accum 16 --seq-length 1024 --output outputs/math_sft_v2
run /root/sft.log python -m training.reasoning.merge --base $BASE --adapter outputs/math_sft_v2 --output outputs/math_sft_v2_merged

step "format-compliance gate: sample MATH-500 (n=60) before spending on PRM/GRPO"
run /root/gate_eval.log python -m training.reasoning.eval_judge --model outputs/math_sft_v2_merged --dataset HuggingFaceH4/MATH-500 --limit 60 --strict --batch 60 --save-text --output data/gate_math_sft_v2.json
run /root/gate.log python -m training.reasoning.check_format_compliance data/gate_math_sft_v2.json --min-rate 0.4

step "real-negative PRM data on MATH (800 questions, 8 rollouts/prefix)"
run /root/gen.log python -m training.reasoning.gen_real_negatives --policy $PWD/$R/outputs/math_sft_v2_merged \
    --local-file data/math_prm_pool.jsonl --exclude data/math_sft.jsonl --questions 800 --max-wrong 500 --rollouts 8 \
    --batch 96 --output data/math_prm_real_v2.jsonl

step "train math-specific PRM v2 (synthetic-corruption base = math_sft.jsonl, not GSM8K)"
run /root/prm.log python -m training.reasoning.train_prm --data data/math_sft.jsonl --extra data/math_prm_real_v2.jsonl \
    --train-layers 8 --epochs 2 --seed 42 --output outputs/math_prm_v2

for arm in outcome prm; do
  step "grpo $arm v2 (MATH prompts, no length_reward)"
  extra=""; [ "$arm" = "prm" ] && extra="--prm outputs/math_prm_v2"
  run /root/grpo_$arm.log python -m training.reasoning.train_grpo --base outputs/math_sft_v2_merged --prompts-file data/math_grpo_pool.jsonl \
      --limit 400 --gens 8 --batch 4 --max-completion 1024 --reward $arm --no-length-reward $extra --seed 42 --output outputs/math_grpo_v2_$arm
  ckpt=$(ls -d $PWD/$R/outputs/math_grpo_v2_$arm/checkpoint-* 2>/dev/null | sort -V | tail -1)
  [ -n "$ckpt" ] || fail "no checkpoint for math_grpo_v2_$arm"
  run /root/grpo_$arm.log python -m training.reasoning.merge --base outputs/math_sft_v2_merged --adapter "$ckpt" --output outputs/math_grpo_v2_${arm}_merged
done

for m in math_sft_v2_merged math_grpo_v2_outcome_merged math_grpo_v2_prm_merged; do
  step "eval $m"
  run /root/eval.log python -m training.reasoning.eval_judge --model outputs/$m --dataset HuggingFaceH4/MATH-500 --limit 500 --strict --batch 64 --save-text --output data/full_${m}_math.json
  run /root/eval.log python -m training.reasoning.eval_judge --model outputs/$m --limit 1319 --batch 96 --save-text --output data/full_${m}_gsm8k.json
  run /root/diag.log python -m training.reasoning.score_eval_chains data/full_${m}_math.json --prm outputs/math_prm_v2 --out data/diag_${m}_math.json
done

step "done"; echo ALL_DONE > /root/all_done

#!/usr/bin/env bash
# Resume run_thread_c_math_v2.sh from the GRPO step: SFT, the format-compliance gate, real-negative
# generation and PRM training already completed successfully before the torch/trl version mismatch
# (torch 2.4.0 lacked FSDPModule, needed by pinned trl==1.13.0) was fixed by upgrading to
# torch/torchvision/torchaudio 2.6.0/0.21.0/2.6.0 (cu124).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R=training/reasoning
step() { echo "[$(date +%H:%M:%S)] $1" >> /root/progress.log; }
fail() { echo "$1" > /root/fail; step "FAIL $1"; exit 1; }
run()  { local log=$1; shift; "$@" >> "$log" 2>&1 || fail "$log"; }
rm -f /root/fail

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

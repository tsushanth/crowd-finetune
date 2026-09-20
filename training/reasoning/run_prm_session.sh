#!/usr/bin/env bash
# One GPU session: pinned SFT -> PRM data + PRM -> validation -> 3 GRPO arms -> full evals -> gaming diagnostics.
# Run from the repo root on the GPU box. Progress in /root/progress.log; /root/fail on error; /root/all_done at the end.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R=training/reasoning
BASE=Qwen/Qwen2.5-3B-Instruct
step() { echo "[$(date +%H:%M:%S)] $1" >> /root/progress.log; }
fail() { echo "$1" > /root/fail; step "FAIL $1"; exit 1; }
run()  { local log=$1; shift; "$@" >> "$log" 2>&1 || fail "$log"; }
rm -f /root/fail /root/all_done; : > /root/progress.log

step "sft";        run /root/sft.log python -m training.reasoning.train_sft --base $BASE --data data/reasoning_sft.jsonl --batch 2 --grad-accum 16 --seq-length 1024
step "merge sft";  run /root/sft.log python -m training.reasoning.merge --base $BASE --adapter outputs/reasoning-sft --output outputs/sft-merged
step "pool";       run /root/pool.log python -m training.reasoning.make_grpo_pool --count 400
step "gen real negatives (pinned SFT)"
run /root/gen.log python -m training.reasoning.gen_real_negatives --policy $PWD/$R/outputs/sft-merged --questions 1400 --max-wrong 900 --rollouts 8 --batch 128 --output data/prm_real_pin.jsonl
step "prm real";   run /root/prm.log python -m training.reasoning.train_prm --extra data/prm_real_pin.jsonl --train-layers 8 --epochs 2 --seed 0 --output outputs/prm_real
step "prm synthetic-only (comparison)"; run /root/prm.log python -m training.reasoning.train_prm --train-layers 8 --epochs 2 --seed 0 --output outputs/prm_syn
step "validate PRMs on pinned-SFT samples"
run /root/val.log python -m training.reasoning.validate_prm --policy $PWD/$R/outputs/sft-merged --no-system --prm outputs/prm_real --count 150 --samples 4 --output data/val_pin_real.json
run /root/val.log python -m training.reasoning.validate_prm --reuse data/val_pin_real.json --prm outputs/prm_syn --output data/val_pin_syn.json

for arm in outcome prm+outcome prm; do
  name=${arm//+/_}
  step "grpo $arm"
  run /root/grpo_$name.log python -m training.reasoning.train_grpo --base outputs/sft-merged --prompts-file data/grpo_pool.jsonl --limit 400 \
      --gens 8 --batch 4 --max-completion 1024 --reward $arm --prm outputs/prm_real --output outputs/grpo_$name
  ckpt=$(ls -d $PWD/$R/outputs/grpo_$name/checkpoint-* 2>/dev/null | sort -V | tail -1)
  [ -n "$ckpt" ] || fail "no checkpoint for $arm"
  run /root/grpo_$name.log python -m training.reasoning.merge --base outputs/sft-merged --adapter "$ckpt" --output outputs/grpo_${name}_merged
done

for m in sft-merged grpo_outcome_merged grpo_prm_outcome_merged grpo_prm_merged; do
  step "eval $m"
  run /root/eval.log python -m training.reasoning.eval_judge --model outputs/$m --limit 1319 --batch 96 --save-text --output data/full_${m}_gsm8k.json
  run /root/eval.log python -m training.reasoning.eval_judge --model outputs/$m --dataset HuggingFaceH4/MATH-500 --limit 500 --strict --batch 64 --save-text --output data/full_${m}_math.json
  run /root/diag.log python -m training.reasoning.score_eval_chains data/full_${m}_gsm8k.json --prm outputs/prm_real --out data/diag_${m}_gsm8k.json
done
step "done"; echo ALL_DONE > /root/all_done

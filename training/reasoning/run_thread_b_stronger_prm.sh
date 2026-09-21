#!/usr/bin/env bash
# Thread B: does a materially stronger PRM change the GRPO picture? Bigger backbone
# (Qwen2.5-1.5B vs 0.5B), more/better-labeled real negatives (2000 questions, 16 rollouts
# per prefix vs 1400/8 before), then ONE confirmatory GRPO arm on the *same* 400-prompt
# pool report 6/7 used, so it's directly comparable to their outcome_nolr/prm_nolr arms
# (rehydrated from HF / pulled locally, not rerun).
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

step "rehydrate sft adapter from HF"
run /root/rehydrate.log python - <<'PYEOF'
from huggingface_hub import snapshot_download
tok = open("/root/.hftoken").read().strip()
snapshot_download("sushanth9/prm-track-sft-adapter", local_dir="training/reasoning/outputs/reasoning-sft", token=tok)
print("rehydrated")
PYEOF
step "merge sft"; run /root/sft.log python -m training.reasoning.merge --base $BASE --adapter outputs/reasoning-sft --output outputs/sft-merged
step "pool (reuse report 6/7's exact 400 GRPO prompts, deterministic rebuild)"
run /root/pool.log python -m training.reasoning.make_grpo_pool --count 400

step "bigger real-negative data: 2000 questions, 16 rollouts/prefix, excluding the GRPO pool too"
run /root/gen.log python -m training.reasoning.gen_real_negatives --policy $PWD/$R/outputs/sft-merged \
    --exclude data/reasoning_sft.jsonl data/grpo_pool.jsonl --questions 2000 --max-wrong 1200 --rollouts 16 \
    --batch 128 --output data/prm_real_big.jsonl

step "train the bigger PRM (Qwen2.5-1.5B, top 10 of 28 layers)"
run /root/prm.log python -m training.reasoning.train_prm --base Qwen/Qwen2.5-1.5B-Instruct \
    --extra data/prm_real_big.jsonl --train-layers 10 --epochs 2 --seed 42 --output outputs/prm_big

step "validate the bigger PRM on the same pinned-SFT samples report 6/7 used"
run /root/val.log python -m training.reasoning.validate_prm --reuse data/val_pin_real.json --prm outputs/prm_big --output data/val_prm_big.json

step "grpo: prm reward (bigger PRM), no length_reward, same pool/settings as report 7"
run /root/grpo.log python -m training.reasoning.train_grpo --base outputs/sft-merged --prompts-file data/grpo_pool.jsonl --limit 400 \
    --gens 8 --batch 4 --max-completion 1024 --reward prm --no-length-reward --prm outputs/prm_big --seed 42 --output outputs/grpo_prm_big_nolr
ckpt=$(ls -d $PWD/$R/outputs/grpo_prm_big_nolr/checkpoint-* 2>/dev/null | sort -V | tail -1)
[ -n "$ckpt" ] || fail "no checkpoint for grpo_prm_big_nolr"
run /root/grpo.log python -m training.reasoning.merge --base outputs/sft-merged --adapter "$ckpt" --output outputs/grpo_prm_big_nolr_merged

step "eval"
run /root/eval.log python -m training.reasoning.eval_judge --model outputs/grpo_prm_big_nolr_merged --limit 1319 --batch 96 --save-text --output data/full_grpo_prm_big_nolr_merged_gsm8k.json
run /root/eval.log python -m training.reasoning.eval_judge --model outputs/grpo_prm_big_nolr_merged --dataset HuggingFaceH4/MATH-500 --limit 500 --strict --batch 64 --save-text --output data/full_grpo_prm_big_nolr_merged_math.json
run /root/diag.log python -m training.reasoning.score_eval_chains data/full_grpo_prm_big_nolr_merged_gsm8k.json --prm outputs/prm_big --out data/diag_grpo_prm_big_nolr_merged_gsm8k.json

step "done"; echo ALL_DONE > /root/all_done

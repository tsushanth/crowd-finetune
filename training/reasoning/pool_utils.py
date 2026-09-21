"""Shared helper: build a fresh GSM8K-train question pool excluding every question set
already used anywhere in this project, so each new thread trains/distills on genuinely
unseen data. Extend EXCLUDE_FILES as new threads consume more of the dataset.
"""
import json
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent

# Every local file whose "question" column has already been used for something (SFT traces,
# PRM real-negative sampling, GRPO prompts). Missing files are skipped so this still runs
# before every one of them has been synced to a given box.
EXCLUDE_FILES = [
    "data/reasoning_sft.jsonl",       # 742 original SFT traces
    "data/prm_real.jsonl", "data/prm_real_v2.jsonl", "data/prm_real_pin.jsonl",  # PRM data (1400 q pools)
    "data/grpo_pool.jsonl",           # report 6/7's 400 GRPO prompts
]


def excluded_questions(extra_files=()):
    seen = set()
    for rel in list(EXCLUDE_FILES) + list(extra_files):
        p = ROOT / rel
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                seen.add(json.loads(line)["question"])
    return seen


def fresh_gsm8k_pool(count, skip=0, extra_exclude_files=()):
    """[{"question", "answer"}] of `count` fresh GSM8K train rows, in dataset order,
    skipping the first `skip` fresh (non-excluded) rows too (so two callers asking for
    disjoint slices never collide even if both exclude the same base sets)."""
    seen = excluded_questions(extra_exclude_files)
    fresh = [ex for ex in load_dataset("openai/gsm8k", "main", split="train") if ex["question"] not in seen]
    return fresh[skip: skip + count]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pool = fresh_gsm8k_pool(args.count, args.skip)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps({"question": e["question"], "answer": e["answer"]}) for e in pool) + "\n")
    print(f"{len(excluded_questions())} excluded questions; wrote {len(pool)} fresh rows -> {out}")

"""Build the GRPO prompt pool: GSM8K train questions the SFT/PRM pipeline has never used.

Excludes (1) the 742 SFT trace questions and (2) the first --fresh-skip fresh questions, which
gen_real_negatives.py used as PRM training data (it takes the first N fresh questions in dataset order).
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="+", default=["data/reasoning_sft.jsonl"])
    ap.add_argument("--fresh-skip", type=int, default=1400)
    ap.add_argument("--count", type=int, default=400)
    ap.add_argument("--out", default="data/grpo_pool.jsonl")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    seen = set()
    for p in args.exclude:
        for line in (root / p).read_text().splitlines():
            if line.strip():
                seen.add(json.loads(line)["question"])
    fresh = [ex for ex in load_dataset("openai/gsm8k", "main", split="train") if ex["question"] not in seen]
    pool = fresh[args.fresh_skip: args.fresh_skip + args.count]
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps({"question": e["question"], "answer": e["answer"]}) for e in pool) + "\n")
    print(f"{len(seen)} excluded traces; {len(fresh)} fresh train questions; "
          f"pool = fresh[{args.fresh_skip}:{args.fresh_skip + args.count}] -> {len(pool)} rows -> {out}")


if __name__ == "__main__":
    main()

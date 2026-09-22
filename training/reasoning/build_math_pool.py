"""Build disjoint MATH train pools for distillation/GRPO/PRM/validation.

Widens report 10's numeric-only filter (which kept ~64% of rows, plain integers/decimals
only) to also accept simple fractions and \\frac{}{} LaTeX answers, since formats.numeric_match
can now score those correctly. Excludes any row whose problem text collides with MATH-500
(the eval set), verified via a set of stripped-whitespace problem strings.
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from . import formats

ROOT = Path(__file__).resolve().parent


def is_scorable(answer: str) -> bool:
    answer = answer.strip()
    return bool(answer) and formats.extract_numeric_token(answer) == answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distill", type=int, default=1200)
    parser.add_argument("--grpo", type=int, default=400)
    parser.add_argument("--real-neg", type=int, default=800)
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    eval_problems = {row["problem"].strip() for row in math500}

    train = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")
    rows = []
    for row in train:
        problem, answer = row["problem"].strip(), str(row["answer"]).strip()
        if problem in eval_problems or not is_scorable(answer):
            continue
        rows.append({"question": problem, "answer": answer})

    print(f"{len(rows)} of {len(train)} train rows are scorable and non-overlapping with MATH-500 "
          f"({100 * len(rows) / len(train):.1f}%)")

    random.Random(args.seed).shuffle(rows)
    need = args.distill + args.grpo + args.real_neg + args.val
    if len(rows) < need:
        raise SystemExit(f"only {len(rows)} usable rows, need {need}")

    out = Path(ROOT / args.out_dir)
    out.mkdir(exist_ok=True)
    i = 0
    for name, n in (("math_distill_pool", args.distill), ("math_grpo_pool", args.grpo),
                     ("math_prm_pool", args.real_neg), ("math_val_pool", args.val)):
        chunk = rows[i:i + n]
        i += n
        with open(out / f"{name}.jsonl", "w") as fh:
            for r in chunk:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {len(chunk)} rows to {name}.jsonl")


if __name__ == "__main__":
    main()

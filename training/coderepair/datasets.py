import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"

TEST_TMPL = "\n\nUnit tests the implementation must satisfy:\n```python\n{body}\n```"


def humaneval_test_lines(test_src):
    lines = []
    for line in test_src.splitlines():
        line = line.strip()
        if line.startswith("assert"):
            lines.append(line)
    return lines or [test_src.strip()]


def mbpp_entry_point(code):
    m = re.search(r"\bdef\s+(\w+)\s*\(", code)
    return m.group(1) if m else None


def load_humaneval(offset=0, limit=None):
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows = []
    for ex in ds.select(range(offset, min(limit or len(ds), len(ds)))):
        rows.append(
            {
                "source": ex["task_id"],
                "prompt": ex["prompt"],
                "entry_point": ex["entry_point"],
                "tests": humaneval_test_lines(ex["test"]),
                "imports": [],
            }
        )
    return rows


def load_mbpp(offset=0, limit=None):
    ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
    rows = []
    for ex in ds.select(range(offset, min(limit or len(ds), len(ds)))):
        rows.append(
            {
                "source": "mbpp/" + str(ex["task_id"]),
                "prompt": ex["prompt"],
                "entry_point": mbpp_entry_point(ex["code"]),
                "tests": list(ex["test_list"]),
                "imports": list(ex["test_imports"]),
            }
        )
    return rows


LOADERS = {"humaneval": load_humaneval, "mbpp": load_mbpp}


def render_question(row, include_tests=True):
    question = row["prompt"]
    if include_tests and row["tests"]:
        question += TEST_TMPL.format(body="\n".join(row["tests"]))
    return question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="humaneval")
    parser.add_argument("--train-out", default="data/code_train.jsonl")
    parser.add_argument("--eval-out", default="data/code_eval.jsonl")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eval-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random

    all_rows = []
    for name in args.datasets.split(","):
        name = name.strip()
        all_rows.extend(LOADERS[name](offset=args.offset, limit=args.limit or None))

    random.Random(args.seed).shuffle(all_rows)
    if args.eval_size > 0:
        eval_rows, train_rows = all_rows[: args.eval_size], all_rows[args.eval_size:]
    else:
        eval_rows, train_rows = [], all_rows

    train_out = ROOT / args.train_out
    eval_out = ROOT / args.eval_out
    train_out.parent.mkdir(parents=True, exist_ok=True)

    with train_out.open("w") as fh:
        for row in train_rows:
            payload = {**row, "question": render_question(row, include_tests=True)}
            fh.write(json.dumps(payload) + "\n")
    with eval_out.open("w") as fh:
        for row in eval_rows:
            payload = {**row, "question": render_question(row, include_tests=False)}
            fh.write(json.dumps(payload) + "\n")
    print(
        f"{len(train_rows)} train rows (tests in prompt) -> {train_out}\n"
        f"{len(eval_rows)} eval rows (original prompt, tests hidden) -> {eval_out}"
    )


if __name__ == "__main__":
    main()
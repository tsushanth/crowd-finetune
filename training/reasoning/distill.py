import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset

from . import formats
from .teacher import Teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--config-name", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default="data/reasoning_sft.jsonl")
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    teacher = Teacher(args.teacher_model)
    out = Path(__file__).resolve().parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, args.config_name, split=args.split)
    n = min(args.limit, len(ds) - args.skip)
    examples = list(ds.select(range(args.skip, args.skip + n)))

    def work(example):
        reference = formats.extract_last_number(example["answer"])
        return teacher.trace(
            example["question"],
            reference,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    rows = []
    with out.open("w") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for trace in pool.map(work, examples, chunksize=1):
            if trace:
                rows.append(trace)
                fh.write(json.dumps(trace) + "\n")
                fh.flush()
    print(f"kept {len(rows)}/{n} verified traces -> {out}")


if __name__ == "__main__":
    main()
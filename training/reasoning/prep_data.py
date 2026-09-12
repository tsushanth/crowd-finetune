import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", default="data/reasoning_sft.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    seen = {}
    for path in args.inputs:
        for line in (root / path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            seen.setdefault(row["question"], row)

    rows = list(seen.values())
    random.Random(args.seed).shuffle(rows)
    if args.max_rows:
        rows = rows[: args.max_rows]

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    reasoning_len = sum(len(r["reasoning"]) for r in rows)
    print(
        f"{len(rows)} unique verified traces -> {out} "
        f"(avg reasoning {reasoning_len // len(rows)} chars)"
    )


if __name__ == "__main__":
    main()
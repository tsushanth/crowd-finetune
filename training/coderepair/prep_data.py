import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", default="data/code_sft.jsonl")
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
            key = row.get("source") or row.get("question", "")[:80]
            seen.setdefault(key, row)

    rows = list(seen.values())
    random.Random(args.seed).shuffle(rows)
    if args.max_rows:
        rows = rows[: args.max_rows]

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"{len(rows)} unique test-verified traces -> {out}")


if __name__ == "__main__":
    main()
import argparse
import json
import random
from pathlib import Path

from . import formats


def emit_rows(row, hybrid):
    long_row = {**row, "control": formats.CONTROL_THINK}
    if not hybrid:
        return [long_row]
    short_row = {
        "question": row["question"],
        "reasoning": "",
        "answer": row["answer"],
        "control": formats.CONTROL_NO_THINK,
    }
    return [long_row, short_row]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", default="data/reasoning_sft.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="emit a terse (no_think) AND a reasoning (think) row per verified trace "
        "so train_sft.py learns one checkpoint that serves both modes",
    )
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

    emitted = []
    for row in rows:
        emitted.extend(emit_rows(row, args.hybrid))
    random.Random(args.seed).shuffle(emitted)

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in emitted:
            fh.write(json.dumps(row) + "\n")
    if args.hybrid:
        think = sum(1 for r in emitted if r["control"] == formats.CONTROL_THINK)
        ters = len(emitted) - think
        print(
            f"{len(rows)} unique verified traces -> {out} "
            f"(hybrid: {think} think + {ters} no_think rows, "
            f"avg reasoning {sum(len(r['reasoning']) for r in emitted if r['reasoning']) // max(think, 1)} chars)"
        )
    else:
        reasoning_len = sum(len(r["reasoning"]) for r in emitted)
        print(
            f"{len(emitted)} unique verified traces -> {out} "
            f"(avg reasoning {reasoning_len // len(emitted)} chars)"
        )


if __name__ == "__main__":
    main()
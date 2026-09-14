import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/repair_rollouts.jsonl")
    parser.add_argument("--output", default="data/repair_dpo_pairs.jsonl")
    parser.add_argument(
        "--require-correct",
        action="store_true",
        default=True,
        help="keep pairs only when chosen is correct and rejected is wrong",
    )
    parser.add_argument(
        "--no-require-correct",
        dest="require_correct",
        action="store_false",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    pairs = []
    for line in (root / args.input).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        comps = [c for c in row["completions"] if c["text"].strip()]
        if len(comps) < 2:
            continue
        comps.sort(key=lambda c: c["advantage"], reverse=True)
        chosen, rejected = comps[0], comps[-1]
        if chosen["advantage"] <= rejected["advantage"]:
            continue
        if chosen["text"] == rejected["text"]:
            continue
        if args.require_correct and not (
            chosen["scores"]["test"] >= 1.0 and rejected["scores"]["test"] < 1.0
        ):
            continue
        pairs.append(
            {
                "question": row["question"],
                "prompt": row["prompt"],
                "chosen": chosen["text"],
                "rejected": rejected["text"],
                "advantage_diff": round(
                    chosen["advantage"] - rejected["advantage"], 4
                ),
            }
        )

    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    print(f"{len(pairs)} DPO pairs -> {out}")


if __name__ == "__main__":
    main()

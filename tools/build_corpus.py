import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = ROOT / "data" / "seed.tsv"
OUT_PATH = ROOT / "data" / "corpus.jsonl"
COLS = ("corpus_id", "niche", "question", "source", "golden_answer", "is_gold")


def fail(msg: str) -> None:
    print(msg)
    sys.exit(1)


def main() -> None:
    rows = []
    seen = set()
    with SEED_PATH.open() as f:
        for lineno, fields in enumerate(csv.reader(f, delimiter="\t"), start=1):
            if not fields:
                continue
            if len(fields) != len(COLS):
                fail(f"line {lineno}: expected {len(COLS)} columns, got {len(fields)}")
            row = dict(zip(COLS, [field.strip() for field in fields]))
            if not row["corpus_id"]:
                fail(f"line {lineno}: missing corpus_id")
            if row["corpus_id"] in seen:
                fail(f"duplicate corpus_id: {row['corpus_id']}")
            seen.add(row["corpus_id"])
            if not row["question"]:
                fail(f"line {lineno}: missing question")
            if not row["golden_answer"]:
                fail(f"line {lineno}: missing golden_answer")
            if row["is_gold"] not in ("0", "1"):
                fail(f"line {lineno}: is_gold not in {{0,1}}: {row['is_gold']!r}")
            row["is_gold"] = int(row["is_gold"])
            rows.append(row)
    with OUT_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
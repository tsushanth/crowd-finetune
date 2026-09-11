import argparse
import json
import os
import random
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "corpus.jsonl"
BENCH_PATH = ROOT / "data" / "bench.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".split_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept", type=int, default=5)
    args = ap.parse_args()
    if args.kept < 1:
        raise SystemExit(f"--kept must be >= 1, got {args.kept}")
    corpus = load_jsonl(CORPUS_PATH)
    rng = random.Random(42)
    shuffled = list(corpus)
    rng.shuffle(shuffled)
    bench = shuffled[:: args.kept]
    write_jsonl(BENCH_PATH, bench)
    print(f"corpus total: {len(corpus)}")
    print(f"bench count: {len(bench)}")
    print(f"bench corpus_ids: {', '.join(r['corpus_id'] for r in bench)}")
    print("note: corpus.jsonl is kept intact; bench items stay playable and are")
    print("      excluded from training only via export --exclude-bench")


if __name__ == "__main__":
    main()
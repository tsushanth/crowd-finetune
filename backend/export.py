import json
from pathlib import Path

_BENCH_PATH = Path(__file__).resolve().parent.parent / "data" / "bench.jsonl"


def _load_bench_ids() -> set[str]:
    if not _BENCH_PATH.exists():
        return set()
    ids = set()
    for line in _BENCH_PATH.read_text().splitlines():
        line = line.strip()
        if line:
            ids.add(json.loads(line)["corpus_id"])
    return ids


def export_sft_jsonl(
    conn, path: str | Path, limit: int = 0, exclude_bench: bool = False, niche: str | None = None
) -> int:
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    niche_clause = " AND i.niche = ?" if niche else ""
    params = (niche,) if niche else ()
    rows = conn.execute(
        f"SELECT a.question, a.corrected_answer, i.corpus_id "
        f"FROM accepted_samples a JOIN items i ON i.id = a.item_id "
        f"WHERE 1 = 1{niche_clause} "
        f"ORDER BY a.id {limit_clause}",
        params,
    ).fetchall()
    bench_ids = _load_bench_ids() if exclude_bench else set()
    if bench_ids:
        rows = [r for r in rows if r["corpus_id"] not in bench_ids]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {"instruction": row["question"], "output": row["corrected_answer"]}
                )
                + "\n"
            )
    return len(rows)


def export_weighted(
    conn,
    path: str | Path,
    max_copies: int = 3,
) -> int:
    rows = conn.execute(
        """
        SELECT a.question, a.corrected_answer, p.reliability
        FROM accepted_samples a
        JOIN submissions s ON s.id = a.submission_id
        JOIN players p ON p.id = s.player_id
        ORDER BY a.id
        """
    ).fetchall()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    meta = []
    with out.open("w") as f:
        for row in rows:
            copies = max(1, min(max_copies, round(row["reliability"] * max_copies)))
            for _ in range(copies):
                f.write(
                    json.dumps(
                        {"instruction": row["question"], "output": row["corrected_answer"]}
                    )
                    + "\n"
                )
            written += copies
            meta.append({"reliability": round(row["reliability"], 3), "copies": copies})
    return written, meta


if __name__ == "__main__":
    import argparse

    from . import db

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude-bench", action="store_true")
    ap.add_argument("--niche", default=None)
    ap.add_argument("--out", default="data/crowd_sft.jsonl")
    args = ap.parse_args()
    conn = db.connect()
    count = export_sft_jsonl(
        conn,
        args.out,
        limit=args.limit,
        exclude_bench=args.exclude_bench,
        niche=args.niche,
    )
    print(f"exported {count} samples to {args.out}")
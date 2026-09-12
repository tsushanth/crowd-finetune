import argparse
import hashlib
import json
import random
import sqlite3
import sys
import time

import httpx

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config, db, export, filter as flt, gold

ARCHETYPES = {
    "novice": {"detect": 0.55, "false_alarm": 0.18, "weight": 0.25},
    "casual": {"detect": 0.70, "false_alarm": 0.08, "weight": 0.40},
    "reliable": {"detect": 0.82, "false_alarm": 0.03, "weight": 0.25},
    "veteran": {"detect": 0.92, "false_alarm": 0.01, "weight": 0.10},
}

LABELS_PATH = config.BASE_DIR / "data" / "candidate_labels.json"


def read_conn() -> sqlite3.Connection:
    p = Path(config.DB_PATH)
    if not p.is_absolute():
        p = config.BASE_DIR / p
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def pick_robots(n: int, rng: random.Random) -> list[dict]:
    keys, weights = zip(*[(k, v["weight"]) for k, v in ARCHETYPES.items()])
    robots = []
    for i in range(n):
        name = rng.choices(keys, weights)[0]
        robots.append(
            {
                "uid": f"{name}-{i:02d}",
                "detect": ARCHETYPES[name]["detect"],
                "false_alarm": ARCHETYPES[name]["false_alarm"],
            }
        )
    return robots


def fp(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def candidate_truth(rconn: sqlite3.Connection) -> dict[int, bool]:
    rows = rconn.execute(
        "SELECT id, question, golden_answer, base_response FROM items WHERE is_gold = 0"
    ).fetchall()
    cache = {}
    if LABELS_PATH.exists():
        cache = json.loads(LABELS_PATH.read_text())
    changed = False
    truth = {}
    for row in rows:
        label = cache.get(str(row["id"]))
        key = fp(row["base_response"])
        if label and label["fp"] == key:
            truth[row["id"]] = bool(label["base_correct"])
            continue
        base_correct = False
        if row["golden_answer"]:
            base_correct = bool(
                gold.judge("verify_gold", row["question"], row["base_response"], row["golden_answer"])["pass"]
            )
        cache[str(row["id"])] = {"fp": key, "base_correct": base_correct}
        truth[row["id"]] = base_correct
        changed = True
        print(f"  labelled item {row['id']}: base_correct={base_correct}")
    if changed:
        LABELS_PATH.write_text(json.dumps(cache, indent=2))
    return truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--sessions", type=int, default=60)
    ap.add_argument("--robots", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    client = httpx.Client(base_url=args.base_url, timeout=60)
    try:
        health = client.get("/health")
        health.raise_for_status()
    except Exception as exc:
        print(f"server not reachable at {args.base_url}: {exc}")
        sys.exit(1)

    rconn = read_conn()
    items = rconn.execute("SELECT COUNT(*) AS n, SUM(is_gold) AS gold FROM items").fetchone()
    if not items["n"]:
        print("DB has no items — start the server first so warm() populates the corpus.")
        sys.exit(1)
    print(
        f"corpus: {items['n']} items ({items['gold']} gold); "
        f"posterior={flt.posterior(rconn):.4f}"
    )

    print("labelling candidate base answers via judge...")
    truth = candidate_truth(rconn)
    n_wrong = sum(1 for v in truth.values() if not v)
    print(f"candidates: {len(truth)} items, base-wrong: {n_wrong}")

    robots = pick_robots(args.robots, rng)
    sessions_per = max(1, (args.sessions + args.robots - 1) // args.robots)
    print(f"robots: {[(r['uid'], r['detect'], r['false_alarm']) for r in robots]}")
    print(f"playing {sessions_per} sessions each so {len(robots) * sessions_per} total...")

    t0 = time.time()
    for robot in robots:
        for _ in range(sessions_per):
            start = client.post("/session/start", json={"tg_user_id": robot["uid"]})
            start.raise_for_status()
            session_id = start.json()["session_id"]
            served = {
                r["item_id"]: r["served_type"]
                for r in rconn.execute(
                    "SELECT item_id, served_type FROM session_items WHERE session_id = ?",
                    (session_id,),
                ).fetchall()
            }
            for item in start.json()["items"]:
                srv = served[item["item_id"]]
                if srv == "control_wrong":
                    prob = robot["detect"]
                elif srv == "control_right":
                    prob = robot["false_alarm"]
                else:
                    prob = robot["detect"] if not truth.get(item["item_id"], False) else robot["false_alarm"]
                client.post(
                    "/session/answer",
                    json={
                        "session_id": session_id,
                        "item_id": item["item_id"],
                        "marked_wrong": rng.random() < prob,
                    },
                ).raise_for_status()
            client.post("/session/end", json={"session_id": session_id}).raise_for_status()
    elapsed = time.time() - t0

    stats = rconn.execute("SELECT * FROM filter_stats WHERE id = 1").fetchone()
    accepted = rconn.execute("SELECT COUNT(*) AS n FROM accepted_samples").fetchone()["n"]
    distinct = rconn.execute(
        "SELECT COUNT(DISTINCT item_id) AS n FROM accepted_samples"
    ).fetchone()["n"]
    players = rconn.execute("SELECT COUNT(*) AS n FROM players").fetchone()["n"]
    subs = rconn.execute("SELECT COUNT(*) AS n FROM submissions").fetchone()["n"]
    top = rconn.execute(
        "SELECT tg_user_id, points FROM players ORDER BY points DESC LIMIT 3"
    ).fetchall()

    print(f"\nran {len(robots) * sessions_per} sessions in {elapsed:.1f}s "
          f"({len(robots) * sessions_per / elapsed:.1f}/s)")
    print(f"players={players} submissions={subs}")
    print(f"filter_stats: hits={stats['taint_hits']} false_alarms={stats['taint_false_alarms']} "
          f"answered={stats['taint_answered']} accepted_count={stats['accepted_count']} "
          f"posterior={(stats['taint_hits'] + 1) / (stats['taint_hits'] + stats['taint_false_alarms'] + 2):.4f}")
    print(f"accepted_samples={accepted} (distinct items {distinct})")
    print("top players:", [dict(t) for t in top])

    if not args.no_export:
        wconn = db.connect()
        rows = export.export_sft_jsonl(wconn, "data/crowd_sft.jsonl", exclude_bench=True)
        written, _meta = export.export_weighted(wconn, "data/crowd_sft_weighted.jsonl")
        wconn.close()
        print(f"exported {rows} rows (bench-excluded) to data/crowd_sft.jsonl; "
              f"{written} weighted lines")

    client.close()
    rconn.close()


if __name__ == "__main__":
    main()
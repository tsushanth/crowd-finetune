import json
import os
import sqlite3
import sys
import tempfile
import threading

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPO = ROOT


def make_fixture_corpus(path: Path) -> None:
    rows = []
    for i in range(1, 11):
        is_gold = i <= 5
        rows.append(
            {
                "corpus_id": f"TST-{i:03d}",
                "niche": "demo",
                "question": f"fixture question number {i}",
                "source": "integration-test",
                "golden_answer": f"stub golden answer {i}" if is_gold else None,
                "is_gold": int(is_gold),
            }
        )
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class StubHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        if not str(self.headers.get("Authorization", "")).startswith("Bearer "):
            self.send_error(401)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        messages = body.get("messages", [])
        system = messages[0]["content"] if messages else ""
        if body.get("response_format", {}).get("type") == "json_object":
            content = '{"verdict":"pass","score":1.0}'
        elif "subtly-wrong" in system:
            content = "stub tainted answer"
        else:
            content = "stub correct answer for the given question"
        payload = {"choices": [{"message": {"content": content}}]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="crowdcheck_it_"))
    corpus = tmp / "corpus.jsonl"
    make_fixture_corpus(corpus)

    server = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    os.environ.update(
        {
            "DB_PATH": str(tmp / "it.db"),
            "CORPUS_PATH": str(corpus),
            "BASE_LLM_URL": f"http://127.0.0.1:{port}/v1",
            "BASE_LLM_API_KEY": "test-key",
            "BASE_LLM_MODEL": "stub-base",
            "JUDGE_MODEL": "stub-judge",
            "REQUIRE_TG_AUTH": "false",
            "FILTER_THETA": "0.8",
            "MIN_EVAL_DELTA": "0.03",
        }
    )

    from fastapi.testclient import TestClient

    from backend import db, export, game

    conn = db.connect()
    with TestClient(game.app) as client:
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["ok"], "health check failed"

        warmed = conn.execute(
            "SELECT is_gold, base_verified, tainted_reference FROM items ORDER BY id"
        ).fetchall()
        assert len(warmed) == 10, f"warm() inserted {len(warmed)} items, want 10"
        gold = [r for r in warmed if r["is_gold"] == 1]
        cand = [r for r in warmed if r["is_gold"] == 0]
        assert len(gold) == 5 and len(cand) == 5, "warm() gold/candidate split wrong"
        assert all(
            r["base_verified"] == 1 and r["tainted_reference"] is not None for r in gold
        ), "gold items not promoted to playable controls"

        start = client.post("/session/start", json={"tg_user_id": "perfect-demo"})
        assert start.status_code == 200, f"session start failed: {start.text}"
        payload = start.json()
        session_id = payload["session_id"]
        served_items = payload["items"]
        assert len(served_items) == 10, "session should serve 10 items"

        served = dict(
            conn.execute(
                "SELECT item_id, served_type FROM session_items WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        )
        controls = [i for i, t in served.items() if t.startswith("control")]
        candidates = [i for i, t in served.items() if t == "candidate"]
        assert len(controls) == 5 and len(candidates) == 5, "session control/candidate split wrong"

        marks = {}
        for item_id, served_type in served.items():
            if served_type == "control_wrong":
                marks[item_id] = True
            elif served_type == "control_right":
                marks[item_id] = False
            else:
                marks[item_id] = item_id in candidates[:3]

        points = 0
        for item_id, mark in marks.items():
            r = client.post(
                "/session/answer",
                json={"session_id": session_id, "item_id": item_id, "marked_wrong": mark},
            )
            assert r.status_code == 200, f"answer for item {item_id}: {r.text}"
            ans = r.json()
            assert ans["is_control"] == (item_id in controls), "answer is_control echo wrong"
            if item_id not in controls:
                assert ans["actual_wrong"] is None, "candidate answer should not leak truth"
            else:
                expected = 1 if served[item_id] == "control_wrong" else 0
                assert ans["actual_wrong"] == expected, "answer actual_wrong echo wrong"
            if mark and ans["actual_wrong"] == 1:
                points += ans["points_awarded"]
        assert points == 60, f"perfect catcher should score 60, got {points}"

        ended = client.post("/session/end", json={"session_id": session_id})
        assert ended.status_code == 200, f"session end failed: {ended.text}"
        summary = ended.json()
        assert summary["accepted_samples"] == 3, f"expected 3 accepted samples, got {summary}"
        assert summary["posterior"] == 0.8, f"posterior should be 0.8 after one clean session, got {summary['posterior']}"

        n_acc = conn.execute("SELECT COUNT(*) AS n FROM accepted_samples").fetchone()["n"]
        assert n_acc == 3, "accepted_samples table should hold 3 rows"
        acc = [row["id"] for row in conn.execute("SELECT id FROM accepted_samples").fetchall()]
        assert all(
            row["corrected_answer"] for row in conn.execute("SELECT corrected_answer FROM accepted_samples").fetchall()
        ), "accepted samples must carry a corrected answer"
        assert n_acc == len(acc), "accepted id list mismatch"

        exported = export.export_sft_jsonl(conn, tmp / "sft.jsonl")
        assert exported == 3, f"export should write 3 rows, wrote {exported}"
        sft_rows = [json.loads(l) for l in (tmp / "sft.jsonl").read_text().splitlines()]
        assert len(sft_rows) == 3 and all("instruction" in r and "output" in r for r in sft_rows), "sft export shape wrong"

        lb = client.get("/leaderboard").json()
        assert lb and lb[0]["tg_user_id"] == "perfect-demo" and lb[0]["points"] == 60, f"leaderboard wrong: {lb}"

        grouped = client.post("/stage2/group", json={"method": "interleaved", "n_groups": 3})
        assert grouped.status_code == 200, f"group failed: {grouped.text}"
        groups = grouped.json()
        assert isinstance(groups["round_id"], int), "group round not created"
        assert [p for grp in groups["groups"] for p in grp], "no players assigned to groups"

        cur = conn.execute(
            "INSERT INTO eval_runs (model_tag, base_bench, tuned_bench, delta, verdict) VALUES (?,?,?,?,?)",
            ("stub-run", 0.5, 0.5, 0.0, "release"),
        )
        eval_run_id = cur.lastrowid
        conn.commit()

        settled = client.post(
            "/stage2/settle",
            json={"eval_run_id": eval_run_id, "sample_ids": acc, "rate": 0.05},
        )
        assert settled.status_code == 200, f"settle failed: {settled.text}"
        result = settled.json()
        assert result["credited"] == 3 and result["paid_players"] == 1, f"settle wrong: {result}"

        payout = conn.execute(
            "SELECT p.reward_balance AS balance, COUNT(pay.id) AS n_payouts "
            "FROM players p LEFT JOIN payouts pay ON pay.player_id = p.id "
            "WHERE p.tg_user_id = 'perfect-demo' GROUP BY p.id"
        ).fetchone()
        assert payout["n_payouts"] == 1, "player should receive exactly one payout"
        assert abs(payout["balance"] - 0.12) < 1e-6, f"reward balance wrong: {payout['balance']}"

    server.shutdown()
    print(
        f"integration_test PASS: warmed 10 items (5 gold->controls); perfect session "
        f"score=60; gate accepted 3/3 candidates at posterior 0.8; export=3 rows; "
        f"group+settle credited 3 samples to 1 player; reward=${payout['balance']:.2f}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"integration_test FAIL: {exc}")
        raise
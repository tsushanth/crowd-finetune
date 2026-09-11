import sqlite3

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

from . import config, db, filter as flt, gold, grouping, reward
from . import auth

CONTROL_WRONG, CONTROL_RIGHT, CANDIDATE = "control_wrong", "control_right", "candidate"
SESSION_SIZE = 10
CONTROLS_PER_SESSION = 5
WRONG_CONTROLS = 3
POINTS_PER_CATCH = 20
SECONDS_PER_ITEM = 120

conn_holder = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    conn = db.connect()
    db.init_schema(conn)
    db.ensure_filter_row(conn)
    gold.warm(conn)
    conn_holder["conn"] = conn
    yield
    conn.close()


app = FastAPI(title="CrowdCheck QA game", lifespan=lifespan)


class StartBody(BaseModel):
    tg_user_id: str


class AnswerBody(BaseModel):
    session_id: int
    item_id: int
    marked_wrong: bool


class EndBody(BaseModel):
    session_id: int


class GroupBody(BaseModel):
    method: str = "interleaved"
    n_groups: int = 3


class SettleBody(BaseModel):
    eval_run_id: int
    sample_ids: list[int]
    rate: float = 0.05


def authenticated(request: Request) -> dict | None:
    if not config.REQUIRE_TG_AUTH:
        return None
    init_data = request.headers.get("X-Telegram-InitData", "")
    verified = auth.verify(init_data, config.TELEGRAM_BOT_TOKEN)
    if not verified:
        raise HTTPException(status_code=401, detail="Invalid Telegram initData")
    return verified


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/session/start")
def start_session(request: Request, body: StartBody):
    verified = authenticated(request)
    tg_user_id = (
        auth.user_id_from_message(verified)
        if verified
        else body.tg_user_id
    )
    conn = conn_holder["conn"]
    player = db.get_or_create_player(conn, tg_user_id)
    controls = conn.execute(
        "SELECT * FROM items WHERE is_gold = 1 AND base_verified = 1 AND tainted_reference IS NOT NULL ORDER BY RANDOM()",
    ).fetchall()[:CONTROLS_PER_SESSION]
    candidates = conn.execute(
        "SELECT * FROM items WHERE is_gold = 0 ORDER BY RANDOM()",
    ).fetchall()[:SESSION_SIZE - CONTROLS_PER_SESSION]
    served = []
    for idx, item in enumerate(controls):
        served_type = CONTROL_WRONG if idx < WRONG_CONTROLS else CONTROL_RIGHT
        served.append((item, served_type))
    for item in candidates:
        served.append((item, CANDIDATE))
    cur = conn.execute("INSERT INTO sessions (player_id) VALUES (?)", (player["id"],))
    session_id = cur.lastrowid
    items_out = []
    for item, served_type in served:
        conn.execute(
            "INSERT INTO session_items (session_id, item_id, served_type) VALUES (?,?,?)",
            (session_id, item["id"], served_type),
        )
        shown = (
            item["tainted_reference"]
            if served_type == CONTROL_WRONG
            else item["base_response"]
        )
        items_out.append(
            {
                "item_id": item["id"],
                "question": item["question"],
                "shown_answer": shown,
                "seconds": SECONDS_PER_ITEM,
            }
        )
    conn.commit()
    return {
        "session_id": session_id,
        "player_points": player["points"],
        "items": items_out,
    }


@app.post("/session/answer")
def answer(request: Request, body: AnswerBody):
    authenticated(request)
    conn = conn_holder["conn"]
    item = db.get_item(conn, body.item_id)
    served_type = conn.execute(
        "SELECT served_type FROM session_items WHERE session_id = ? AND item_id = ?",
        (body.session_id, body.item_id),
    ).fetchone()["served_type"]
    is_control = served_type in (CONTROL_WRONG, CONTROL_RIGHT)
    actual_wrong = 1 if served_type == CONTROL_WRONG else (0 if served_type == CONTROL_RIGHT else None)
    points = 0
    shown = (
        item["tainted_reference"]
        if served_type == CONTROL_WRONG
        else item["base_response"]
    )
    cur = conn.execute(
        "INSERT INTO submissions (session_id, player_id, item_id, is_control, shown_answer, marked_wrong, actual_wrong, points_awarded) VALUES (?,?,?,?,?,?,?,?)",
        (
            body.session_id,
            conn.execute("SELECT player_id FROM sessions WHERE id = ?", (body.session_id,)).fetchone()["player_id"],
            body.item_id,
            int(is_control),
            shown,
            int(body.marked_wrong),
            actual_wrong,
            0,
        ),
    )
    sub_id = cur.lastrowid
    player_id = conn.execute(
        "SELECT player_id FROM sessions WHERE id = ?", (body.session_id,)
    ).fetchone()["player_id"]

    if served_type == CONTROL_WRONG:
        if body.marked_wrong:
            points = POINTS_PER_CATCH
            conn.execute("UPDATE players SET taint_hits = taint_hits + 1, points = points + ? WHERE id = ?", (points, player_id))
            conn.execute("UPDATE filter_stats SET taint_hits = taint_hits + 1 WHERE id = 1")
        conn.execute("UPDATE filter_stats SET taint_answered = taint_answered + 1 WHERE id = 1")
    elif served_type == CONTROL_RIGHT:
        if body.marked_wrong:
            conn.execute("UPDATE players SET taint_false_alarms = taint_false_alarms + 1 WHERE id = ?", (player_id,))
            conn.execute("UPDATE filter_stats SET taint_false_alarms = taint_false_alarms + 1 WHERE id = 1")
    else:
        if body.marked_wrong:
            conn.execute("UPDATE players SET candidates_marked_wrong = candidates_marked_wrong + 1 WHERE id = ?", (player_id,))

    if points:
        conn.execute("UPDATE submissions SET points_awarded = ? WHERE id = ?", (points, sub_id))
    conn.commit()
    return {
        "session_id": body.session_id,
        "item_id": body.item_id,
        "points_awarded": points,
        "actual_wrong": actual_wrong,
        "is_control": is_control,
    }


@app.post("/session/end")
def end_session(request: Request, body: EndBody):
    authenticated(request)
    conn = conn_holder["conn"]
    session_points = conn.execute(
        "SELECT COALESCE(SUM(points_awarded), 0) FROM submissions WHERE session_id = ?",
        (body.session_id,),
    ).fetchone()[0]
    conn.execute(
        "UPDATE sessions SET ended_at = datetime('now'), completed = 1 WHERE id = ?",
        (body.session_id,),
    )
    conn.execute(
        "UPDATE players SET sessions_completed = sessions_completed + 1 WHERE id = (SELECT player_id FROM sessions WHERE id = ?)",
        (body.session_id,),
    )
    new_accepts = flt.accept_candidates(conn)
    conn.commit()
    return {
        "session_id": body.session_id,
        "session_points": session_points,
        "accepted_samples": len(new_accepts),
        "posterior": flt.posterior(conn),
    }


@app.get("/leaderboard")
def leaderboard(limit: int = 10):
    conn = conn_holder["conn"]
    rows = conn.execute(
        "SELECT tg_user_id, points FROM players ORDER BY points DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


@app.post("/stage2/group")
def make_groups(body: GroupBody):
    return grouping.assign_groups(conn_holder["conn"], method=body.method, n_groups=body.n_groups)


@app.post("/stage2/settle")
def settle_rewards(body: SettleBody):
    result = reward.settle(
        conn_holder["conn"],
        eval_run_id=body.eval_run_id,
        accepted_sample_ids=body.sample_ids,
        rate=body.rate,
    )
    return result
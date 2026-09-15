from . import config, gold


def posterior(conn) -> float:
    row = conn.execute(
        "SELECT taint_hits, taint_false_alarms FROM filter_stats WHERE id = 1"
    ).fetchone()
    hits = row["taint_hits"] if row else 0
    false_alarms = row["taint_false_alarms"] if row else 0
    return (hits + 1) / (hits + false_alarms + 2)


def player_posterior(hits: int, false_alarms: int) -> float:
    return (hits + 1) / (hits + false_alarms + 2)


def accept_candidates(conn, theta: float | None = None) -> list[int]:
    theta = theta or config.FILTER_THETA
    subs = conn.execute(
        """
        SELECT s.*, i.question, i.golden_answer,
               p.taint_hits AS player_hits, p.taint_false_alarms AS player_false_alarms
        FROM submissions s
        JOIN items i ON i.id = s.item_id
        JOIN players p ON p.id = s.player_id
        WHERE s.is_control = 0 AND s.marked_wrong = 1 AND s.accepted = 0
        """
    ).fetchall()
    subs = [
        sub for sub in subs
        if player_posterior(sub["player_hits"], sub["player_false_alarms"]) >= theta
    ]
    accepted = []
    for sub in subs:
        conn.execute("UPDATE submissions SET accepted = 1 WHERE id = ?", (sub["id"],))
        corrected = sub["golden_answer"] or gold.base_answer(sub["question"])
        conn.execute(
            "INSERT INTO accepted_samples (submission_id, item_id, question, corrected_answer) VALUES (?,?,?,?)",
            (sub["id"], sub["item_id"], sub["question"], corrected),
        )
        conn.execute(
            "UPDATE filter_stats SET accepted_count = accepted_count + 1 WHERE id = 1"
        )
        accepted.append(sub["id"])
    conn.commit()
    return accepted
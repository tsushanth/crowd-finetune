from . import config


def credit_release(conn, eval_run_id: int, accepted_sample_ids: list[int]) -> list[dict]:
    credited = []
    for sample_id in accepted_sample_ids:
        row = conn.execute(
            """
            SELECT a.id AS sample_id, s.player_id
            FROM accepted_samples a
            JOIN submissions s ON s.id = a.submission_id
            WHERE a.id = ?
            """,
            (sample_id,),
        ).fetchone()
        if not row:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO release_samples (eval_run_id, accepted_sample_id) VALUES (?,?)",
            (eval_run_id, sample_id),
        )
        credited.append(dict(row))
    conn.commit()
    return credited


def distribute(conn, eval_run_id: int, rate: float | None = None) -> list[dict]:
    rate = rate if rate is not None else 0.05
    teams = conn.execute(
        """
        SELECT p.id AS player_id, p.reliability, COUNT(*) AS n_samples,
               SUM(p.reliability * ?) AS credit
        FROM release_samples rs
        JOIN accepted_samples a ON a.id = rs.accepted_sample_id
        JOIN submissions s ON s.id = a.submission_id
        JOIN players p ON p.id = s.player_id
        WHERE rs.eval_run_id = ?
        GROUP BY p.id
        """,
        (rate, eval_run_id),
    ).fetchall()
    for team in teams:
        amount = round(team["credit"], 4)
        conn.execute(
            "INSERT INTO payouts (player_id, eval_run_id, amount, payout_type) VALUES (?,?,?,?)",
            (team["player_id"], eval_run_id, amount, "sample_reward"),
        )
        conn.execute(
            "UPDATE players SET reward_balance = reward_balance + ? WHERE id = ?",
            (amount, team["player_id"]),
        )
    conn.commit()
    return [dict(t) for t in teams]


def settle(conn, eval_run_id: int, accepted_sample_ids: list[int], rate: float | None = None) -> dict:
    credited = credit_release(conn, eval_run_id, accepted_sample_ids)
    teams = distribute(conn, eval_run_id, rate)
    return {"credited": len(credited), "paid_players": len(teams), "teams": teams}
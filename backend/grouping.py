import random

from typing import Sequence

from . import db

EPSILON = 0.15

_METHODS = ("random", "interleaved", "epsilon_greedy")


def reliability(conn, player_id: int) -> float:
    row = conn.execute(
        "SELECT taint_hits, taint_false_alarms FROM players WHERE id = ?", (player_id,)
    ).fetchone()
    if not row:
        return 0.0
    hits = row["taint_hits"] + 1
    false_alarms = row["taint_false_alarms"] + 1
    return hits / (hits + false_alarms)


def sync_reliability(conn) -> None:
    for row in conn.execute("SELECT id FROM players").fetchall():
        conn.execute(
            "UPDATE players SET reliability = ? WHERE id = ?",
            (reliability(conn, row["id"]), row["id"]),
        )
    conn.commit()


def rank_players(conn) -> list[int]:
    rows = conn.execute(
        "SELECT id FROM players ORDER BY reliability DESC, points DESC"
    ).fetchall()
    return [r["id"] for r in rows]


def _chunk(players: Sequence[int], n_groups: int) -> list[list[int]]:
    size = max(1, (len(players) + n_groups - 1) // n_groups)
    return [players[i:i + size] for i in range(0, len(players), size)]


def assign_random(conn, n_groups: int) -> list[list[int]]:
    players = [r["id"] for r in conn.execute("SELECT id FROM players").fetchall()]
    random.shuffle(players)
    return _chunk(players, n_groups)


def assign_interleaved(conn, n_groups: int) -> list[list[int]]:
    groups: list[list[int]] = [[] for _ in range(n_groups)]
    for i, player_id in enumerate(rank_players(conn)):
        groups[i % n_groups].append(player_id)
    return groups


def assign_epsilon_greedy(conn, n_groups: int) -> list[list[int]]:
    groups: list[list[int]] = [[] for _ in range(n_groups)]
    for i, player_id in enumerate(rank_players(conn)):
        target = random.randrange(n_groups) if random.random() < EPSILON else i % n_groups
        groups[target].append(player_id)
    return groups


def assign_groups(conn, method: str = "interleaved", n_groups: int = 3) -> dict:
    method = method if method in _METHODS else "interleaved"
    sync_reliability(conn)
    groups = {
        "random": assign_random,
        "interleaved": assign_interleaved,
        "epsilon_greedy": assign_epsilon_greedy,
    }[method](conn, n_groups)
    if not any(groups):
        return {"round_id": None, "groups": [], "scores": {}}
    cur = conn.execute(
        "INSERT INTO rounds (method, n_groups, eval_metric) VALUES (?,?,?)",
        (method, n_groups, "bench"),
    )
    round_id = cur.lastrowid
    scores = {}
    for group_id, members in enumerate(groups):
        for player_id in members:
            weight = reliability(conn, player_id)
            scores[player_id] = weight
            conn.execute(
                "INSERT INTO team_assignments (round_id, player_id, group_id, weight) VALUES (?,?,?,?)",
                (round_id, player_id, group_id, round(weight, 4)),
            )
    conn.commit()
    return {"round_id": round_id, "groups": groups, "scores": scores}


def winners(conn, round_id: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT group_id, AVG(weight) AS mean_weight, COUNT(*) AS members,
               SUM(CASE WHEN weight >= 0.75 THEN 1 ELSE 0 END) AS strong_members
        FROM team_assignments WHERE round_id = ?
        GROUP BY group_id ORDER BY mean_weight DESC
        """,
        (round_id,),
    ).fetchall()
    return [dict(r) for r in rows]
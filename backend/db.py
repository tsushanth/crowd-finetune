import sqlite3

from pathlib import Path

from . import config

_SCHEMA = Path(__file__).resolve().parent / "schema.sql"


def connect() -> sqlite3.Connection:
    db_path = Path(config.DB_PATH)
    if not db_path.is_absolute():
        db_path = config.BASE_DIR / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA.read_text())
    _migrate(conn)
    conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(players)").fetchall()}
    for col, ddl in (
        ("reliability", "ALTER TABLE players ADD COLUMN reliability REAL NOT NULL DEFAULT 0"),
        ("reward_balance", "ALTER TABLE players ADD COLUMN reward_balance REAL NOT NULL DEFAULT 0"),
    ):
        if col not in cols:
            conn.execute(ddl)


def get_or_create_player(conn: sqlite3.Connection, tg_user_id: str) -> dict:
    row = conn.execute(
        "SELECT * FROM players WHERE tg_user_id = ?", (tg_user_id,)
    ).fetchone()
    if row:
        return dict(row)
    cur = conn.execute(
        "INSERT INTO players (tg_user_id) VALUES (?)", (tg_user_id,)
    )
    conn.commit()
    return dict(
        conn.execute("SELECT * FROM players WHERE id = ?", (cur.lastrowid,)).fetchone()
    )


def get_item(conn: sqlite3.Connection, item_id: int) -> dict:
    return dict(conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone())


def ensure_filter_row(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT OR IGNORE INTO filter_stats (id) VALUES (1)")
    conn.commit()
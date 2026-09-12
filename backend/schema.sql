PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS players (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tg_user_id TEXT UNIQUE NOT NULL,
  points INTEGER NOT NULL DEFAULT 0,
  sessions_completed INTEGER NOT NULL DEFAULT 0,
  taint_hits INTEGER NOT NULL DEFAULT 0,
  taint_false_alarms INTEGER NOT NULL DEFAULT 0,
  candidates_marked_wrong INTEGER NOT NULL DEFAULT 0,
  reliability REAL NOT NULL DEFAULT 0,
  reward_balance REAL NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  corpus_id TEXT UNIQUE NOT NULL,
  niche TEXT NOT NULL,
  question TEXT NOT NULL,
  source TEXT NOT NULL,
  golden_answer TEXT,
  is_gold INTEGER NOT NULL DEFAULT 0,
  base_verified INTEGER NOT NULL DEFAULT 0,
  base_response TEXT,
  tainted_reference TEXT,
  payload TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  player_id INTEGER NOT NULL REFERENCES players(id),
  started_at TEXT NOT NULL DEFAULT (datetime('now')),
  ended_at TEXT,
  completed INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS session_items (
  session_id INTEGER NOT NULL REFERENCES sessions(id),
  item_id INTEGER NOT NULL REFERENCES items(id),
  served_type TEXT NOT NULL,
  PRIMARY KEY (session_id, item_id)
);

CREATE TABLE IF NOT EXISTS submissions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL REFERENCES sessions(id),
  player_id INTEGER NOT NULL REFERENCES players(id),
  item_id INTEGER NOT NULL REFERENCES items(id),
  is_control INTEGER NOT NULL,
  shown_answer TEXT NOT NULL,
  marked_wrong INTEGER NOT NULL,
  actual_wrong INTEGER,
  points_awarded INTEGER NOT NULL DEFAULT 0,
  accepted INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS accepted_samples (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  submission_id INTEGER NOT NULL REFERENCES submissions(id),
  item_id INTEGER NOT NULL REFERENCES items(id),
  question TEXT NOT NULL,
  corrected_answer TEXT NOT NULL,
  expert_verified INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS filter_stats (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  taint_answered INTEGER NOT NULL DEFAULT 0,
  taint_hits INTEGER NOT NULL DEFAULT 0,
  taint_false_alarms INTEGER NOT NULL DEFAULT 0,
  accepted_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS eval_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_tag TEXT NOT NULL,
  base_bench REAL NOT NULL,
  tuned_bench REAL NOT NULL,
  delta REAL NOT NULL,
  verdict TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS rounds (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  method TEXT NOT NULL,
  n_groups INTEGER NOT NULL,
  eval_metric TEXT NOT NULL DEFAULT 'bench',
  winner_candidate TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS team_assignments (
  round_id INTEGER NOT NULL REFERENCES rounds(id),
  player_id INTEGER NOT NULL REFERENCES players(id),
  group_id INTEGER NOT NULL,
  weight REAL NOT NULL DEFAULT 1.0,
  PRIMARY KEY (round_id, player_id)
);

CREATE TABLE IF NOT EXISTS release_samples (
  eval_run_id INTEGER NOT NULL REFERENCES eval_runs(id),
  accepted_sample_id INTEGER NOT NULL REFERENCES accepted_samples(id),
  PRIMARY KEY (eval_run_id, accepted_sample_id)
);

CREATE TABLE IF NOT EXISTS payouts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  player_id INTEGER NOT NULL REFERENCES players(id),
  eval_run_id INTEGER NOT NULL REFERENCES eval_runs(id),
  amount REAL NOT NULL,
  payout_type TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
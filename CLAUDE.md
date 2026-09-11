# CrowdCheck QA — project context & handoff

Carry this forward as the canonical state of the project. Keep this file
updated when you change architecture, mechanics, or decisions.

## Objective

An individual wants to crowdsource "post-processing" improvements to open
models: a gamified adversarial-QA loop where humans catch deliberate base-model
errors, the good catches become LoRA fine-tune data, and the improved model is
released — proven by a held-out eval, and only then do contributors get paid.

Research conclusion (do not re-litigate without new evidence): a full
"Wikipedia-style" correction of model output is NOT viable (frontier models
are right on most objective facts, and the tail is dominated by rare/hallucinated
claims you cannot verify cheaply). The viable path is niche adversarial/verification
data: pick a niche where correctness is decidable, the base model is wrong on a
meaningful share, and someone wants the fine-tuned model. Mechanics are lifted
from two papers:

- GAP (arXiv:2410.04038): gamified adversarial QA on an open model
  (0.147→0.477 GPT-judge score on Qwen2-7B); ~50k players → ~3.7k usable QAs,
  ~93% of crowd output discarded by the gate.
- Crowd-SFT (arXiv:2506.04063): interleaved/synergized grouping + reputation
  up to +55% on convergence from the same human data.

## The game mechanic (Stage 1, GAP-style)

- Each session = 10 items: 5 tainted controls (3 `control_wrong` + 2
  `control_right`) + 5 candidate items. Player is blind to the split.
- 120s per item; base model's answer is shown pre-filled; player says
  right/wrong. 20 pts ONLY for a correct catch ("mark wrong") on a tainted
  control. Points (XP) are the gamification; reward_balance is separate cash.
- Tainted answers are the base answer with a subtle wrong inserted; on
  `control_wrong` items they look convincing (reCAPTCHA-style captchas).
- Acceptance gate: `posterior = (taint_hits+1)/(taint_hits+taint_false_alarms+2)`
  over player's control history (Laplace smoothing, stored per-player in
  players and aggregated in filter_stats). A candidate marked wrong is accepted
  when the player's posterior >= FILTER_THETA (0.8). This is the PRIMARY quality
  defence; the reputation engine (Stage 2) is secondary (see Crowd-SFT caveat).
- Release gate: train SFT LoRA, run held-out bench, release only if
  tuned − base >= MIN_EVAL_DELTA (0.03). Bench = 1-in-5 held-out corpus items
  that stay playable in-game but are excluded from TRAINING via
  `export --exclude-bench`.

## Stage 2 — reputation, grouping, eval-proven payouts (Crowd-SFT-style)

- players gain `reliability` = (hits+1)/(hits+fa+2), `reward_balance`.
- New tables: `rounds`, `team_assignments`, `release_samples`, `payouts`.
- Grouping methods: random / interleaved (alternate strong + novice on each
  team) / epsilon_greedy (random placement w.p. ε so newcomers can prove
  themselves). See `backend/grouping.py`.
- `export_weighted` upsamples proven contributors' samples into training
  (weight capped at 3 copies).
- `reward.settle(eval_run_id, sample_ids, rate=0.05)` credits ONLY the accepted
  samples that shipped in a RELEASED eval run, paying contributors
  `n × rate × reliability` into `reward_balance`, logged in `payouts`.
  Whoever's samples measurably improved the model gets paid; nobody else.
- Crowd-SFT caveat (documented in spec): reward-attribution accuracy drops as
  the player pool passes ~100. The control gate, not reputation, remains the
  primary defence.

## Architecture — file map

```
backend/
  schema.sql     # full SQLite schema (Stage 1 + 2 tables, player columns)
  config.py      # env-driven config; DB_PATH, CORPUS_PATH, BASE_LLM_*, JUDGE_*,
                 # FILTER_THETA, MIN_EVAL_DELTA, TELEGRAM_BOT_TOKEN, REQUIRE_TG_AUTH
  db.py          # connect(check_same_thread=False), init_schema, _migrate,
                 # get_or_create_player, get_item, ensure_filter_row
  gold.py        # chat() OpenAI-compat, base_answer, judge, taint_answer,
                 # load_corpus (corpus.jsonl with demo fallback), warm() at startup
                 # (calls the base+judge LLMs — needs a real BASE_LLM_API_KEY)
  filter.py      # posterior(), accept_candidates(conn, theta)
  game.py        # FastAPI app; endpoints: /health, /session/start|answer|end,
                 # /leaderboard, /stage2/group, /stage2/settle
                 # authenticated() = 401 on bad/missing X-Telegram-InitData when
                 # REQUIRE_TG_AUTH=true; derives tg_user_id from verified user
  auth.py        # Telegram initData HMAC verify + user helpers (stdlib only)
  grouping.py    # reliability, sync_reliability, rank_players, assign_*,
                 # assign_groups, winners
  reward.py      # credit_release, distribute, settle
  export.py      # export_sft_jsonl(conn, path, limit, exclude_bench),
                 # export_weighted(conn, path); CLI --exclude-bench flag
data/
  seed.tsv       # source-of-truth corpus sheet (corpus_id<niche<question<source<golden_answer<is_gold)
  corpus.jsonl   # ACTIVE corpus: 22 EU-AI-Act compliance items (see caveats)
  bench.jsonl    # deterministic 1-in-5 holdout (tools/split_bench.py)
  demo_corpus.jsonl  # fictional building-code QA example (refer only)
tools/
  build_corpus.py  # seed.tsv → corpus.jsonl (validated, exit 1 on bad rows)
  split_bench.py   # corpus.jsonl → bench.jsonl WITHOUT shrinking corpus
                   # (bench stays playable; excluded only at export time)
miniapp/index.html # single-file Telegram miniapp (works in browser too);
                   # sends X-Telegram-InitData when inside Telegram
prompts/judge.md   # judge-model grading rubric
training/
  lora-config.yaml # LLaMA-Factory LoRA-SFT config
  eval.py          # held-out bench eval, release-gate; records eval_runs
deploy/
  Caddyfile        # auto-HTTPS, serves miniapp/, /api/* → 127.0.0.1:8000
  crowdcheck.service  # systemd unit (venv uvicorn, EnvironmentFile=<REPO>/.env)
  setup.sh         # placeholder-guarded installer (needs caddy installed)
  DEPLOY.md        # step-by-step BotFather → token → Mini App URL → deploy
.gitignore, .env.example, spec-stage1.md, README.md, CLAUDE.md
```

## Work state

Verified by me end-to-end:
- `python3 -m py_compile backend/*.py tools/*.py` — compiles clean.
- Full loop smoke test (temp DB): corpus 22 items loaded → 14 accepted after
  gate → epsilon-greedy groups made → settle credited 14 samples / 1 player →
  export all 14 vs exclude-bench 11 (3 bench items' samples correctly held out).
- `tools/build_corpus.py` round-trip regenerates corpus byte-identical.
- `tools/split_bench.py --kept 5` → 5 bench corpus_ids: EU-022, EU-013,
  EU-019, EU-005, EU-004; corpus.jsonl stays full (22).
- auth self-test (Telegram spec reimplemented independently): valid initData →
  verified; tampered / wrong-token / expired / empty → all rejected. NOTE:
  build test initData with `quote(value, safe="")` (encodeURIComponent → `%20`),
  NOT Python's `urlencode` (`+`); `auth.py` uses plain `unquote`, which matches
  Telegram's decoder exactly — a `+`-encoded value fails verification by design.
- uvicorn curl checks (prior run): REQUIRE_TG_AUTH=false → dev mode accepts
  body tg_user_id (200); true → no header/garbage → 401; valid header → session
  created for the initData user id (body ignored).

The `db.py` fix `sqlite3.connect(..., check_same_thread=False)` is required —
FastAPI creates the connection in the lifespan thread but executes sync
endpoints in a threadpool; without it every request 500s.

A `.venv` exists in the repo (created via `uv`; has fastapi/uvicorn + deps).
`BASE_LLM_API_KEY` is NOT set in this environment, so `gold.warm()` will fail at
server startup unless you stub the endpoint, point CORPUS_PATH at a corpus with
no items, or set a real key. The smoke tests above deliberately avoid touching
game.py's lifespan.

## Runbook

```bash
cd /Users/sushanthtiruvaipati/Documents/Default Project/crowd-finetune
python3 -m py_compile backend/*.py tools/*.py
python3 tools/split_bench.py --kept 5
DB_PATH=/tmp/<name>.db python3  - <<'EOF'  # inject a smoke test like the ones above
EOF
# server (dev):  REQUIRE_TG_AUTH=false DB_PATH=data/crowd.db python3 -m uvicorn backend.game:app --reload --port 8000
# export:        python -m backend.export --exclude-bench
# train:         llamafactory-cli train training/lora-config.yaml
# eval gate:     GPU=1 python training/eval.py --base Qwen/Qwen2.5-7B-Instruct --adapter outputs/crowd-lora
```

## Next moves (in rough priority)

1. Run the full pipeline against a REAL API key once — validate warm(),
   judge grading on the EU-AI-Act corpus, and confirm the demo play loop from
   the miniapp in a browser.
2. Validate/swap the EU-AI-Act corpus facts flagged as low-confidence
   (EU-018 substantial-modification→Art 25; EU-015 deepfake disclosure
   user↔deployer mapping; EU-009 third fine tier 7.5M/1.5% Art 99(5)) against
   the consolidated OJ text. Best done as `data/seed.tsv` edits + rebuild.
3. Wire `training/eval.py` (currently records eval_runs) to call
   `reward.settle` automatically on a passing release run so payouts are
   eval-gated by construction rather than manual.
4. Optional hardening exercises: concurrency test on the shared sqlite conn;
   a `data/crowd_sft_valid.jsonl` validation step before LLaMA-Factory; an
   integration test that drives game.py with a stubbed LLM endpoint so CI can
   run the full loop without a key.

## Work-state markers

- Stage 1 (game + gate + export + bench/release): DONE, smoke-tested.
- Stage 2 (reputation/grouping + eval-proven payouts): DONE, smoke-tested.
- Deploy (Telegram initData auth + Caddy/systemd configs): DONE, files written,
  auth unit-verified; nothing deployed to a real host yet.
- Not started: real-key pipeline run, corpus fact audit, eval→settle wiring,
  any actual model training or release.

## Caveats & decisions to respect

- No code comments (user convention). Docs carry the rationale; keep them in
  sync if you change mechanics.
- Zero new dependencies so far (stdlib sqlite/json/hmac; fastapi+uvicorn+httpx
  only at the server layer). Don't add frameworks casually.
- `.env.example` is the contract for config; update it with any new env key.
- Rewards: XP is gamification; `reward_balance` (USD at `rate`) is real and
  capped by `n × rate × reliability`. Lantern-rule: pay only samples that
  shipped in a released, eval-proven run.
- Do not silently restore the "shrink corpus.jsonl on split" behavior — the
  bench-vs-train design is: bench items stay playable, excluded from training
  only via `export --exclude-bench`.
# CrowdCheck QA — MVP scaffold

Crowdsourced adversarial QA → LoRA fine-tune loop (GAP mechanism), with a
Bayesian acceptance gate and a held-out eval release gate.

## Structure

```
backend/
  schema.sql        # SQLite schema (players, items, sessions, submissions, …)
  db.py             # sqlite3 connection + helpers
  gold.py           # base-model answers, judge grading, taint generation
  filter.py         # Bayesian acceptance of adversarial samples (posterior >= θ)
  game.py           # FastAPI endpoints for the miniapp
  export.py         # accepted_samples → LLaMA-Factory train JSONL
data/
  demo_corpus.jsonl # demo niche items (fictional building-code QA) — REPLACE
miniapp/index.html  # single-file Telegram miniapp (works in browser too)
prompts/judge.md    # judge-model rubric (grading modes)
training/
  lora-config.yaml  # LLaMA-Factory LoRA-SFT config
  eval.py           # held-out bench, judge-graded, release-gate
spec-stage1.md      # mechanics, tuning numbers, success criteria
```

## Setup

```bash
cd crowd-finetune
python3 -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn httpx pydantic
cp .env.example .env
```

Set `BASE_LLM_API_KEY` (OpenAI-compatible endpoint). `BASE_LLM_MODEL` is the
model you will fine-tune (a cheap open model served via an OpenAI-compatible
endpoint, e.g. vLLM/LM Studio/Together). `JUDGE_MODEL` grades answers.

## Replace the demo corpus

`data/corpus.jsonl` is the active corpus (already seeded with a realistic
EU-AI-Act compliance niche; `data/demo_corpus.jsonl` is a fictional example).
Manage it from the seed sheet:

```bash
python3 tools/build_corpus.py            # data/seed.tsv → data/corpus.jsonl (validated)
python3 tools/split_bench.py --kept 5    # deterministic 1-in-5 holdout → data/bench.jsonl
```

`is_gold: 1` items become the tainted controls (reCAPTCHA-style quality gates)
once the base model is verified correct on them.

Niche must satisfy: decidable correctness (golden answers you can grade),
the base model is wrong on a meaningful share, someone wants the final model.

## Run

```bash
uvicorn backend.game:app --reload --port 8000
```

Warm-up (on first startup) answers every corpus item with the base model,
judge-verifies gold items, and generates subtle-wrong tainted answers. Then
open `miniapp/index.html` in a browser (demo users) or ship it as a Telegram
miniapp pointing at the backend.

## Data → training

```bash
python -m backend.export --exclude-bench  # skips accepted samples on bench.jsonl
python -m backend.export                   # full export (debug/comparison)
```

`--exclude-bench` keeps the held-out bench items playable in the game but
out of the training set (default `data/crowd_sft.jsonl`, alpaca format).

Register `crowd_sft` as an alpaca template dataset in LLaMA-Factory's
`dataset_info.json`, then:

```bash
llamafactory-cli train training/lora-config.yaml
```

## Release gate

Hold out ~20% of corpus items into `data/bench.jsonl` (via `tools/split_bench.py`;
excluded from training). After training:

```bash
GPU=1 python training/eval.py --base Qwen/Qwen2.5-7B-Instruct \
    --adapter outputs/crowd-lora
```

Releases only if tuned − base ≥ `MIN_EVAL_DELTA` (default 0.03). Runs are
recorded in `eval_runs`.

## Stage 2 — reputation + eval-proven payouts

```bash
# balance strong + new contributors onto mixed teams (Crowd-SFT interleaved)
curl -X POST localhost:8000/stage2/group -H 'Content-Type: application/json' \
     -d '{"method":"interleaved","n_groups":3}'

# replicate proven contributors' samples in the training set
python -c "from backend import db, export; cmd=db.connect(); print(export.export_weighted(cmd, 'data/crowd_sft_weighted.jsonl'))"

# after a passing eval run, pay only contributors whose samples shipped
curl -X POST localhost:8000/stage2/settle -H 'Content-Type: application/json' \
     -d '{"eval_run_id":1,"sample_ids":[1,2,3],"rate":0.05}'
```

`rate` is $/accepted-sample credit; settlement writes to `payouts` and
`players.reward_balance`. Details in `spec-stage1.md`.

## Cost notes

Judge/base inference is the only real spend (~cents per session at small
scale); LoRA training is a few tens of dollars per run on a rented GPU. For
a Telegram miniapp you must serve the miniapp and backend over HTTPS
(Telegram requires it) — any static host + a small VPS works.

## Deploy (Telegram mini app)

See `deploy/DEPLOY.md` for a step-by-step Caddy (auto-HTTPS) + systemd
deployment. Set `TELEGRAM_BOT_TOKEN` and `REQUIRE_TG_AUTH=true` in `.env` to
enable server-side Telegram initData verification; leave `REQUIRE_TG_AUTH`
false for dev/demo mode with no auth.
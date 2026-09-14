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

## DeepCogito reasoning track (`training/reasoning/`)

Learning track reproducing the open "reasoning post-training" recipe (Open-R1 /
Cogito-style): SFT on teacher-distilled reasoning traces, then GRPO RL with a
verifiable reward, then iterative refinement (IDA-lite). Runs on a single
rented ~24GB GPU (Qwen2.5-3B default). Isolated from the backend; its deps
(torch/trl/vllm) are training-only and do not touch the stdlib-only game loop.

```
training/reasoning/
  formats.py     # <reasoning>/<answer> tags, parse + number-match helpers
  rewards.py     # GRPO reward funcs (exact-match + format compliance)
  teacher.py     # OpenAI-compatible teacher (OpenRouter; R1-0528 guard-block CoT)
  distill.py     # teacher traces for GSM8K/MATH -> data/reasoning_*.jsonl
                 #   (verifies each trace against the gold final answer)
  prep_data.py   # merge + dedupe teacher batches -> data/reasoning_sft.jsonl
  train_sft.py   # LoRA SFT on distilled traces
  train_grpo.py  # GRPO RL with verifiable reward (LoRA)
  merge.py       # merge LoRA adapter -> dense checkpoint for the next stage
  eval_judge.py  # held-out accuracy (local HF model or OpenAI-compatible endpoint)
  run_on_gpu.sh  # one-shot: SFT -> merge -> GRPO -> merge -> eval table
  serve.sh       # vLLM entry point for evals / chat
  requirements.txt
```

Runbook (from repo root; the scripts are `python -m` modules):

```bash
# macOS / any box with an OpenRouter key: build the SFT dataset
python -m training.reasoning.distill --limit 200          # teacher traces (needs BASE_LLM_API_KEY)
python -m training.reasoning.prep_data --inputs data/reasoning_sft.jsonl data/reasoning_sft_more.jsonl --max-rows 400
# rented CUDA box (RunPod/Vast); repo synced to the box first
bash training/reasoning/run_on_gpu.sh                    # SFT -> merge -> GRPO -> merge -> eval table
# serve the final checkpoint for evals / chat
training/reasoning/serve.sh
```

The month-1 deliverable is a before/after accuracy table (base vs SFT vs GRPO)
on the same test split — a DeepCogito-style reasoning model trained on a
rented GPU. Stage-2 process supervision reuses this same loop later.

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
  corpus.jsonl   # ACTIVE corpus: 38 EU-AI-Act compliance items, 24 is_gold
                 # (7 verified+tainted, usable as control_right/control_wrong;
                 # see caveats — golden_answer phrasing matters a lot here)
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
  lora-config.yaml # LLaMA-Factory LoRA-SFT config (rented-GPU path)
  train_lora.py    # minimal PEFT LoRA-SFT (r=64/alpha=128/all-linear) that
                   # runs ON THIS MAC via MPS — mirrors lora-config.yaml;
                   # `python -m training.train_lora --data data/crowd_sft.jsonl
                   # --base Qwen/Qwen2.5-3B-Instruct --out outputs/crowd-lora`
  eval.py          # held-out bench eval, release-gate; records eval_runs and
                   # auto-settles rewards on release (see "Work-state markers")
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
- Full loop smoke test (temp DB, back when the corpus was 22 items): 14
  accepted after gate → epsilon-greedy groups made → settle credited 14
  samples / 1 player → export all 14 vs exclude-bench 11 (3 bench items'
  samples correctly held out). NOTE: corpus has since grown to 38 items /
  24 gold (EU-023..EU-038 added); re-run split_bench and re-validate gate
  numbers against the current corpus.jsonl before trusting these.
- `tools/build_corpus.py` round-trip regenerates corpus byte-identical.
- `tools/split_bench.py --kept 5` writes data/bench.jsonl without touching
  corpus.jsonl (bench stays playable; excluded only at export time).
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
A real API key IS configured in `.env` (OpenRouter) and the real-key pipeline
run is done — but `.env` is gitignored and NOT in the repo, so a fresh clone
needs it before `gold.warm()` at server startup. The smoke tests above
deliberately avoid touching game.py's lifespan.

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

1. Confirm the demo play loop from the miniapp with a REAL browser (I
   simulated the exact HTTP calls and static-served index.html; a headless or
   desktop browser click-through is the remaining gap), ideally against the
   deployed Caddy TLS endpoint.
2. Real 7B release pass: export the same accepted samples and run
   LLaMA-Factory/trl training + `training/eval.py --base Qwen/Qwen2.5-7B-Instruct`
   on the rented GPU (this ties into the reasoning track's box). The local 3B
   run proves the mechanics; the 7B run is the production pairing (game base).
3. Grow data volume: real play sessions, then re-check bench; the 17-row
   train set is demonstrative, not yet consequential for the domain model.
4. Optional hardening exercises: concurrency test on the shared sqlite conn;
   a `data/crowd_sft_valid.jsonl` validation step before training; an
   integration test that drives game.py with a stubbed LLM endpoint so CI can
   run the full loop without a key; a headless-browser e2e for the miniapp.

## Work-state markers

- Stage 1 (game + gate + export + bench/release): DONE, smoke-tested.
- Stage 2 (reputation/grouping + eval-proven payouts): DONE, smoke-tested.
- Deploy (Telegram initData auth + Caddy/systemd configs): DONE, files written,
  auth unit-verified; nothing deployed to a real host yet.
- Real-key pipeline run: DONE (OpenRouter, qwen/qwen-2.5-7b-instruct base +
  openai/gpt-4o-mini judge). warm() runs clean end-to-end; see corpus-authoring
  caveat below for what the run actually taught us.
- Corpus fact audit: DONE for the 3 originally-flagged items (EU-009 fixed
  1.5%→1%, EU-015 fixed a misleading "(the user under the Act)" gloss,
  EU-018 corroborated-not-verbatim, left as-is).
- eval.py→settle wiring on release: DONE (commit 6996994, plus the
  base/tuned aliasing bug the wiring caught).
- Judge-rubric strictness audit: DONE (see caveat at bottom). `prompts/judge.md`
  now grades ONLY the decisive facts the question asks about: self-volunteered
  extras (including a WRONG volunteer citation) no longer sink a correct core
  answer; `gold.py`'s base system prompt no longer says "cite the controlling
  rule" (it was inducing the citations that got penalized). Empirically
  re-verified against the exact previously-failing pattern: same response
  scores 1.0 pass now (was 0.5/fail before the fix), a decisive-fact error
  still scores 0.0, fabricated in-topic extras score 0.8 borderline (still
  blocked for gold use). Net effect: verified gold grew 7→10, more
  control_right supply.
- Live miniapp full-loop test: DONE (real server: poisoned-and-killed a stray
  IPv6 `http.server` that was squatting :8000 so `localhost` missed uvicorn).
  Drove the exact UI sequence (start → 10 answers → end × sessions +
  leaderboard + static-serve of index.html = 200). The gate flips exactly
  where designed: perfect-catch players push global posterior 0.5→0.8 in one
clean session and acceptance begins; a sloppy all-flag player adds 2 false
   alarms + 5 accepted at posterior 0.84 and co-punishes on payouts.
   NOTE: test "players" (expert-demo/detective-demo/hunter-demo/novice-demo/
   sloppy-demo) are simulated harnesses whose control decisions were
  DB-guided (perfect detectives) — deliberately not naive, to exercise the
  mechanics. Real naive behaviour is what the control gate exists for.
- Actual LoRA training + release-gate pass: DONE on this Mac (MPS).
  `train_lora.py` was written because LLaMA-Factory isn't installable here;
  several real bugs surfaced and fixed along the way:
    - MPS OOM was an accumulation bug in my first loop (one giant graph);
      rewrote as per-batch grad accumulation + PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0.
    - transformers 5.x `apply_chat_template` returns a dict, not a tensor
      (`enc["input_ids"]` now), in eval.py.
    - reward.distribute read the possibly-stale `players.reliability` column →
      payouts were $0.00 unless /stage2/group had run; now computes reliability
      inline from live taint_hits/false_alarms.
    - bench regenerated (~1-in-5 of the grown 38-item corpus → 8 items:
      EU-012/030/028/025/032/019/022/018). Adapter trained on 17 exported rows
      (8 unique pairs) → outputs/crowd-lora (gitignored via outputs/).
    - eval result: base=0.05 tuned=0.3375 delta=+0.2875 -> RELEASE. eval_run 1
      recorded; auto-settle paid 2 players (hunter 0.929rel/20samples→$0.929,
      sloppy 0.571rel/5samples→$0.143).
- Not started: miniapp click-TEST driving a real browser (I simulated the
  exact HTTP calls; no headless browser on hand), Telegram deploy to a real
  host, training the real 7B (game base) on a GPU box — the local release
  proves the mechanism on Qwen2.5-3B, not yet on the 7B the game serves.

## Local training runbook (this Mac, no GPU box needed)

```bash
REQUIRE_TG_AUTH=false .venv/bin/python -m uvicorn backend.game:app --port 8000  # warms live items
# play real sessions, then:
python3 tools/split_bench.py --kept 5
.venv/bin/python -m backend.export --exclude-bench
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
  .venv/bin/python -m training.train_lora --data data/crowd_sft.jsonl --base Qwen/Qwen2.5-3B-Instruct --out outputs/crowd-lora
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
  .venv/bin/python -m training.eval --base Qwen/Qwen2.5-3B-Instruct --adapter outputs/crowd-lora --max-new 128
```

Feature caveats confirmed live: 16GB MPS trains/evaluates Qwen2.5-3B LoRA fine
(via PEFT, not trl — the trl path needs the GPU box per the reasoning track);
a 7B in the same pattern would need the rented GPU.

## Work-state markers (reasoning track)

- Distillation: DONE for real. `data/reasoning_sft.jsonl` = 235 verified
  GSM8K reasoning traces (56 via deepseek-r1-0528 + 203 via
  deepseek-v3.2, deduped, avg reasoning ~274 chars). Teacher hits
  OpenRouter via existing BASE_LLM_* env. cogito-v2.1-671b is NOT on
  OpenRouter (404); default `TEACHER_MODEL` is now deepseek-r1-0528.
- trl 1.13/transformers 5.17 API deltas (verified by running):
  `SFTConfig`/`GRPOConfig` use `max_length` (not `max_seq_length`), no
  `warmup_ratio`/`max_prompt_length`; trainers take `processing_class=`
  (not `tokenizer=`); GRPO requires `generation_batch_size % num_generations
  == 0`; reward funcs receive `prompts=/completions=/completion_ids=`
  keyword args plus one kwarg per dataset column — `rewards.py` matches this.
- Smoke tests: SFT and GRPO both run end-to-end on the Mac (CPU-forced;
  small Qwen2.5-0.5B). Mac MPS + `pin_memory` segfaults in the trainer
  weight-load path (local-only quirk; box is CUDA, bf16 auto-enabled via
  `torch.cuda.is_available()`).
- eval_judge.py is now schema-aware: auto-detects question column
  (`problem`/`question`), auto-picks the match mode per row — "number"
  (last-number equality via `<answer>` parsing; GSM8K-style `\n#### 42`
  and pure-numeric refs) vs "string" (cleaned full-string equality for
  symbolic MATH-500 answers). Verified no-model against real GSM8K +
  MATH-500 rows (MATH first-100: 61 number / 39 string). `run_on_gpu.sh`
  step [7/7] adds a MATH-500 eval (limit 100, `MATH_LIMIT` env) so the
  before/after table is GSM8K + MATH ≥ twice the signal.
- DONE — real GPU run (Vast 4090, 24GB, ~$2.2 total, month-1 deliverable):
  `eval_{base,sft,grpo}.json` + `_math` in `training/reasoning/data/`:
    - GSM8K(100): base 0.77 -> SFT 0.82 -> GRPO 0.79
    - MATH-500(100): 0.39 / 0.39 / 0.38 (absolute MATH value INFLATED by
      loose prose-tail number matching; only relative deltas meaningful here)
  Takeaway: SFT on 235 distilled traces is a solid +5 on GSM8K. GRPO on top
  slightly regressed (82->79) and is a format-drift effect: exact-match reward
  on only 128 rows taught terser bare-number replies, flipping 7 rows (2 W, 5 L).
- Box env lessons: runpod pytorch 2.4 image needs `pip install -U "torch>=2.5"`
  for transformers 5.17, then `pip uninstall torchvision torchaudio` (stale
  builtins crash under new torch). `run_on_gpu.sh` needs SCDIR-relative paths
  (BASH_SOURCE-based REPO/SCDIR; checkpoints live under the *script* dir, not
  repo). datasets 5.x refuses `openai/gsm8k` without explicit config name ->
  eval_judge falls back to first config. transformers 5.17: use
  `apply_chat_template(tokenize=False)` then tokenizer() for generate().
- Ops: Vast key `~/.config/vastai/vast_api_key` + CLIs in `/tmp/pv` venv
  (vastai 1.7.0). Instances sometimes exit with "resources unavailable";
  keep them (do NOT destroy) and retry `vastai start instance <id>` — storage
  only bills while stopped and the disk (with all checkpoints) persists.

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
- **Corpus authoring: don't embed a citation in `golden_answer` unless the
  question explicitly asks for one.** Found live getting control_right supply
  from 2→7: the base model reliably answers yes/no questions correctly but
  almost always volunteers its own citation anyway (and gets the specific
  article/annex number wrong more often than not — this model hallucinates
  citation numbers confidently even when the underlying fact is right). The
  judge rubric's "no unverifiable additions" scoring then penalizes that
  self-volunteered wrong citation as if it contradicted the reference, even
  though the actual yes/no answer was correct — so a factually-correct
  response scores 0.5 ("borderline") instead of 1.0 ("pass") for a reason
  that has nothing to do with what the question asked. Verified directly via
  `gold.judge(...)` on the SAME base_response with only golden_answer changed:
  embedding "Yes, under Article 17." scored 0.5/fail; rewriting to the
  substantive fact with no citation to contradict ("Yes, providers of
  high-risk AI systems must establish and maintain a quality management
  system...") scored 1.0/pass on the identical model output. Also found:
  "which article/annex is X" questions (free-recall of a specific number) are
  NOT an easy category for this model — 0/6 verified in one batch, all wrong
  numbers, all confidently stated. Yes/no-restatement questions (this model
  confirming a stated fact rather than recalling a citation cold) verify far
  more reliably — that's the actual "easy" category, not citation lookups.

## code-repair track (branch `niche/code-repair`, worktree crowd-finetune-coderepair)

Phase 1 (docstring → code) and Phase 2 (buggy fn + failing tests → patch)
both built and smoke-verified end-to-end. The reward is a MECHANICAL
unit-test pass/fail (no LLM judge). Given a docstring/signature + unit tests
(HumanEval/MBPP-style), the model generates a full function def inside
`<reasoning>/<answer>` tags; a hardened subprocess sandbox runs the tests per
completion, isolated from the training loop. SFT on test-verified teacher
traces, then GRPO with the test-pass reward, then optional DPO on own
rollouts as a more stable alternative to GRPO.

Files: `training/coderepair/{sandbox,formats,rewards,teacher,datasets,distill,distill_repair,
make_buggy,prep_data,train_sft,train_grpo,train_dpo,collect_rollouts,make_dpo,
eval_code}.py`, `run_on_gpu.sh`, `run_repair_on_gpu.sh`, `sweep_dpo.sh`,
`requirements.txt`. Reuses `training/reasoning/{formats,merge,requirements}`.

Real numbers from the CPU smoke (this Mac, no GPU box):
- sandbox battery 15/15: correct/wrong/syntax/loop(timeout)/socket/fs-write/
  os.system/os.fork/subprocess/urllib/ctypes.native/env-write/tempfile all
  behave; 0 host-filesystem escapes. Wrapper isolates candidate stdout via
  `os.dup2` -> /dev/null (a printing candidate corrupted result parsing before
  the fix). CPU infinite loops are bounded by the subprocess timeout, not the
  rlimit (rlimit is a second net, do not rely on alarm alone).
- datasets: `openai/openai_humaneval` NEEDS the `openai/` namespace (bare id
  404s on hub 5.x). 164 problems; CURRENT split re-run at eval-size 50 ->
  114 train / 50 eval (was 156/8; uncommitted in the working tree). The first
  8 eval rows are unchanged (same seed-42 shuffle). Train questions embed the
  rendered tests; eval questions keep the original prompt only (tests hidden).
  MBPP loader written but not part of the smoke. NOTE: the old 10 smoke
  traces (HumanEval/17/30/52/66/121/123/124/127/137/18) are ALL now eval-50
  rows -> they are NOT re-usable as SFT data (eval leakage); the final SFT set
  is built only from traces of the current 114-row train split.
- make_buggy.py: AST mutators (wrong_operator/off_by_one/wrong_var/invert_cond/
  drop_guard), sandbox-verifies each buggy variant compiles + fails >=1 test.
  `--variants N` generates up to N unique buggy variants per source problem
  (default 1), yielding source keys like `HumanEval/157#v1`. Deduplication is
  by `(source, bug_type, buggy_code)` so multi-variant rows are preserved.
  `--data` accepts comma-separated files for combined HumanEval+MBPP input.
  `build_row()` adds `"system": formats.REPAIR_SYSTEM` so downstream
  training/eval uses the correct repair prompt (previously fell back to
  CODE_SYSTEM, a real eval bug).
- distill_repair.py: teacher distills fix traces under REPAIR_SYSTEM,
  sandbox-verified. Same ThreadPool pattern as distill.py.
- REPAIR_SYSTEM / build_repair_question: the repair prompt shows the buggy
  function + failing tests and asks for the corrected function.
- Phase-2 train/eval split: 142 train variants (from 77 phase-1 SFT traces,
  3 variants each, deduped), 27 eval (from 50 held-out eval rows, teacher-
  solved first then bug-injected).
- SFT: Qwen2.5-0.5B-Instruct, 4 traces, 1 epoch, CPU -> loss 1.633, ~78s.
  Then merge.py adapter -> dense checkpoint before GRPO (GRPOTrainer loads a
  dense base, NOT a LoRA dir).
- transformers 5.17 on this Mac defaults the trainer device to MPS and
  MPS+pin_memory segfaults in the weight-load path -> train_sft.py /
  train_grpo.py gained `--device {cpu,auto}` (use_cpu), default cpu. CUDA
  boxes: `--device auto` sets bf16=True via torch.cuda.is_available().
- DPO pipeline (`collect_rollouts.py`, `make_dpo.py`, `train_dpo.py`):
  RLAIF on own rollouts. `collect_rollouts` samples N completions/prompt from
  the best merged checkpoint (GRPO-merged or SFT-merged), scores them with the
  same sandbox rewards as GRPO (test_pass hard + format), computes group-
  normalized advantages. `make_dpo` picks chosen=highest-advantage correct /
  rejected=worst wrong. `train_dpo` does LoRA DPO on top of the merged SFT
  checkpoint. All three trainers take `--seed`/`--device`/`--wandb`.
- `sweep_dpo.sh`: 3-seed DPO sweep + pairwise win matrix on repair eval.
  Needs `outputs/code-repair-sft-merged` + rollouts already collected.
- eval baseline (eval-8, pass@1): Qwen2.5-3B base 0.625, openai/gpt-4o-mini
  0.75, anthropic/claude-3-haiku 0.75. Local 3B was CPU (48 tok/s); the
  tuned-model row needs `--adapter` (smoke compared base vs base).
- Path conventions (verified): datasets.py writes to repo-root `data/`;
  distill.py and train_*.py resolve `--data/--out` package-relative under
  `training/coderepair/` (data/code_sft.jsonl etc. live there);
  eval_code.py `--out` is now CWD-relative (default repos-root data/).
  merge.py `--output` must be an ABSOLUTE path (it resolves relative to
  training/reasoning/ and silently nests junk dirs otherwise).

FULL GPU RUN DONE (RTX 4090, runpod/pytorch 2.4.0 + torch 2.14.0+cu130,
transformers 5.17.0, trl 1.13.0; on-demand instance 50824587, ~$0.87/hr,
~1.5h incl. setup -> ~$1.3 billed + OpenRouter distill/competitor spend).
Pipeline data/code_train.jsonl (114) + data/code_eval.jsonl (50), SFT on the
77 test-verified traces, GRPO on 60 train prompts x 8 gens, eval as pass@1 on
the 50 hidden-test eval rows. REAL NUMBERS:
- SFT (Qwen2.5-3B-Instruct LoRA, 77 rows x 2 epochs, batch4/ga8): loss 0.2118.
  Slow-ish on 4090 (~54 min, 6 steps): eager attention at seq 2048; no flash
  attn installed = expected.
- GRPO: rewards/_reward_call mean 0.79 -> 0.83 across steps, std ~0.19-0.27
  (the test-pass reward DISCRIMINATES); completions mean_length ~274-299 (
  terminated, clipped_ratio 0, no max_completion clips); loss 0.875 -> 1.65;
  KL small (lr on the merge: ~1e-10 scale at the tail = fine).
- pass@1 on 50 eval rows: Qwen2.5-3B base 0.52, SFT-merged 0.52,
  GRPO-merged 0.52, openai/gpt-4o-mini 0.68, anthropic/claude-3-haiku 0.56.
  Coins flat: with 77 SFT traces + 60-prompt GRPO the tuned models neither
  gained nor lost vs base (0.52). gpt-4o-mini is the ceiling on this eval. A
  bigger trace set and/or repair-style (phase 2) is where headroom likely is;
  verdict: pipeline is sound, signal is absent at this data/RL scale.
- Bugs the box run caught (both fixed + committed): (1) train_sft.py /
  train_grpo.py defaulted `--device cpu` (the Mac MPS workaround) and
  run_on_gpu.sh step [4] didn't pass `--device auto` -> SFT trained the 3B on
  CPU (964% CPU spin, 0% GPU). Default changed to `auto`; pass `--device cpu`
  on this Mac. (2) GRPO OOM'd the 24GB 4090 with batch4 x gens8: policy +
  frozen ref = 2x 3B bf16 (~13GB) + a 32-rollout padded gen batch blew past
  24GB at the backward. Fixed: batch 1 (8 rollouts/step) +
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -> peaks ~16.9GB.
- Instance state: 50824587 STOPPED (storage-only billing; disk holds
  code-sft-merged / code-grpo-merged checkpoints + run.log under
  /workspace/coderepair/outputs*). Do NOT destroy until checkpoints are
  pulled or re-run is planned; stopping keeps them.
- Eval artifacts pulled back to repo-root data/code_eval_{base,sft,grpo,
  table}.json (gitignored). run_on_gpu.sh is re-runnable idempotently on the
  stopped box (start instance -> re-ssh; steps [1-5] are fast).
- Verified pre-run (unchanged): `bash -n` clean; datasets split reproduces
  byte-for-byte; eval_code CWD-relative outs; merge absolute outputs.
- Phase-2 GPU run provisioned to Vast box (instance 50824587) but the box
  is currently STOPPED/unreachable. Before restarting: `vastai start instance
  50824587`, then re-sync repo + the NEW data files (code_repair_train.jsonl
  142 rows, code_repair_eval.jsonl 27 rows) since git ignores them.
- Data volume is the main lever now: 142 repair train variants (was 51), 46
  SFT traces distilled from them (in progress; distill_repair.py timed out
  after 600s locally — run on the box or with smaller --workers). The DPO
  pipeline adds a second RL stage that may be more stable than GRPO.
- MBPP is wired into the pipeline at the datasets layer: run
  `datasets.py --datasets humaneval,mbpp` to get combined train/eval, then
  `distill.py --data data/code_train.jsonl` -> combined SFT -> make_buggy.py.
  The default make_buggy.py `--data` is `data/code_sft.jsonl`; comma-
  separated paths are accepted.
- Semi-open items: tuned model still needs to clear base 0.52; DPO step
  untested end-to-end on 3B; box restart + phase-2 rerun with new data.

## PHASE-2 FULL GPU RUN — RESULTS (instance 50824587, this session)

Pipeline ran END-TO-END on the 4090: SFT (code_repair_sft.jsonl) -> merge ->
GRPO (limit 142, 1136 steps) -> merge(ckpt-1136) -> collect_rollouts (142x8,
REPAIR_SYSTEM sandbox scoring) -> make_dpo (121 pairs) -> DPO (1 epoch, 16
steps) -> merge -> held-out eval matrix (repair 27 / generate 50 / MBPP 50) +
competitors. Real numbers (pass@1; tok/gen in parens):

```
dataset   base      sft       grpo      dpo
repair    0.778     0.370     0.370     0.481   (tok/gen 650/1236/1139/1217)
generate  0.560     0.340     0.400     0.300   (tok/gen 312/806/751/766)
mbpp      0.060     0.020     0.000     0.000   (tok/gen 217/657/601/785)
competitor repair: gpt-4o-mini 0.852, claude-3-haiku 0.630 (vs tuned 0.37-0.48)
```

VERDICT: the current code-repair SFT data (106-row auto-distilled traces) HURTS.
Every tuned stage regresses the base on ALL three sets while emitting 2-4x more
tokens (=2-7x cost). DPO recovers some of the damage on repair (0.481) but not
to base, let alone the APIs. The GRPO step overfit the 142-prompt repair unit
tests (reward-mean 0.79->0.83) but that does NOT transfer. The loop's own gate
correctly REJECTS this data: phase-2 needs BETTER (crowd-gated / correctness-
filtered, volume >> 142) repair traces before tuning is worth it. This mirrors
the toolcall track where crowd-gated data did move the needle.

Ops lessons from this run:
- Launch ONE phase-2 script per box. Two concurrent `run_repair_on_gpu.sh`
  (this session: mine `run_phase2.log` (GRPO_LIMIT=128 default) + the user's
  `repair_run.log` (GRPO_LIMIT=142 ROLLOUT_LIMIT=142) raced on the SAME
  output dirs; the user's survived, mine silently died. Result: read/watch
  `/workspace/coderepair/repair_run.log`, NOT run_phase2.log.
- DPO OOMs on the 4090 at `--batch 2 --seq-length 2048` (chosen+rejected+
  frozen-ref = 3 model passes, first-step alloc blew past 2.99GB free; "0/16"
  bar then crash). Works at `--batch 1 --grad-accum 8` (same eff. batch, ~11s/
  step, 16 steps ~3 min on 121 pairs).
- Path resolution gotchas hit live: train_dpo.py prepends its package dir to
  RELATIVE `--output` (-> nested `training/coderepair/training/coderepair/
  outputs/...`); always pass ABSOLUTE paths for `--output/--out/--adapter/--
  model` on the box and find adapters by `find -name '*dpo*' -exec test -f
  '{}/adapter_model.safetensors'`. eval_code `--data` is package-relative by
  default BUT absolute paths work if the file exists at the exact path (it
  root-resolves then falls back to as-given; generate/MBPP data lives at repo-
  root `data/code_{eval,mbpp_eval}.jsonl`, NOT under training/coderepair/data).
- `set -euo pipefail` + `ls -d dir/checkpoint-*` on empty glob exits non-zero
  and silently kills a continuation script inside `$(...)`; guard with
  `|| true`.
- All evals + competitor table are in `/workspace/coderepair/data/evals/`
  (gitigored); code-repair-dpo adapter (239MB) is at
  `training/coderepair/training/coderepair/outputs/code-repair-dpo`.
- Box still RUNNING (paid) for the vLLM throughput bench + toolcall cycle;
  artifacts worth pulling to repo `data/` before stopping if re-run planned.

## PHASE-2 FULL GPU RUN — DIAGNOSIS (this session)

The 106-row auto-distilled `code_repair_sft.jsonl` HURTS: base 0.778 → SFT 0.370
(repair eval), base 0.560 → SFT 0.340 (generate), base 0.060 → SFT 0.020 (MBPP).
Every tuning stage (SFT, GRPO, DPO) is worse than base. Diagnosis from trace
audit:

1. **Teacher over-reasoning**: deepseek-v3.2 emits 2000-3000 character reasoning
   essays per trace (median ~2400 chars) — the model learns to emit verbose
   prose before the code block, inflating tokens 2-4× and pushing completions
   past practical limits.

2. **Over-rewrite pathology**: the teacher rarely produces a minimal patch.
   Instead it rewrites the entire function body, often restructuring loops,
   renaming variables, or adding helper functions that were not in the original.
   The model learns "when you see a buggy function, rewrite everything" — which
   destroys working code on the generate/MBPP evals where the input is NOT buggy.

3. **Bug-type/reasoning mismatch**: the `bug_type` field says e.g. "wrong_var"
   but the teacher reasoning often discusses operators, conditions, or general
   algorithmic flaws. The model is not learning a consistent mapping from
   mutation type to fix pattern.

4. **No crowd gate**: the toolcall track succeeded because accepted samples
   were mechanically verified correct before entering SFT. The code-repair
   "teacher" is just an API call — no correctness filter at data-creation time
   beyond the sandbox pass. Sandbox-pass is necessary but NOT sufficient: the
   answer can be correct while the reasoning is wrong or the rewrite is
   excessive, and those bad habits get baked into the model.

5. **GPT-4o-mini ceiling is unreachable**: on repair eval, gpt-4o-mini scores
   0.852 — the tuned model peaks at 0.481 (DPO). Even perfect training data
   would struggle to close a 37-point gap on 27 eval items. The niche is
   competitive against frontier APIs in a way toolcall is not.

## Sunset recommendation (code-repair)

The code-repair pipeline (sandbox → AST mutators → teacher distill →
SFT/GRPO/DPO → eval) is mechanically sound. The data source is the blocker.
Fixing it would require:
- A minimal-patch teacher (not deepseek-v3.2's essayist)
- Crowd-gated acceptance of traces (only verified + concise reasoning)
- An order of magnitude more data (>> 142 variants, >> 106 traces)
- A separate "don't over-rewrite" regularization signal

None of these are on the current roadmap. **Recommended: archive the code-repair
files (keep the branch `niche/code-repair` for reference) and redirect effort to
toolcall**, which has already demonstrated a working flywheel (555-item corpus,
499 traces, tuned-3B beating GPT-4o-mini on full-domain eval).
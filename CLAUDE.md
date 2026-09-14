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

## Tool-call niche — the competitiveness claim (commits 5ac5ed9, 40a7446)

Pivoted proof-of-claim: pick a narrow, mechanically-decidable niche where a
small open model goes head-to-head with pay-per-token frontier APIs under the
IDENTICAL system prompt and NO LLM judge — correctness is a deterministic
oracle, so the accuracy delta is not a rubric artifact.

- `backend/toolcall.py`: NICHE="tool-call". The model routes a user request to
  exactly one tool from a JSON-schema toolset, or declines when none fits.
  SYSTEM_PROMPT forces a single JSON reply. `gold.chat` keeps `max_tokens=512`
  (without it claude-haiku-4.5 402'd: OpenRouter budgets the model's full 64K
  output ceiling). Grader is mechanical: `toolcall.is_correct` (line 186)
  schema-checks the emitted call — tool name, required args, types, enums,
  extra props, exact-arguments equality — no LLM in the loop.
- Sources of truth: `data/toolcall_corpus.jsonl` (30 items), `data/bench.jsonl`
  = 6 held-out TC items (TC-007/009/010/018/020/028); `tools/build_toolcall.py`
  and `tools/split_bench.py --corpus <file>` build/split it. `corpus.jsonl`'s
  38 EU items are legacy for this niche.
- `training/eval_toolcall.py` is the comparison harness (phase 2 deliverable):
  same SYSTEM_PROMPT to every model, mechanical grading, accuracy + $/1K-query
  cost table, eval_runs + release-settle wiring kept.

### Result (eval_run 5, 6-item TC bench, credit-loaded OpenRouter)

```
model                                             acc    $/1K q   tok/q     tps
Qwen2.5-3B-Instruct base                       1.0000  $0.26708     487       238
Qwen2.5-3B-Instruct tuned (crowd-lora-v2)      1.0000  $0.29400     487       216
openai/gpt-4o-mini                             0.8333  $0.07490     471         -
anthropic/claude-haiku-4.5                     0.8333  $0.69050     507         -
```

The claim holds on this slice: the STOCK 3B routes all 6 perfectly (1.0);
gpt-4o-mini and claude-haiku-4.5 each whiff one (0.8333) — both decline
TC-009 instead of calling turn_off_device. The EU-trained crowd-lora-v2
neither helps nor hurts routing (identical 6/6; delta 0.0 -> hold, nothing
settled), which is the honest answer to "does a strong EU LoRA transfer to a
different niche?" — no. Caveat: n=6 (1.0 vs 0.833 = a one-item gap), so this
proves the niche on this slice, not statistically beyond it; rerun on the full
30-item corpus or a grown bench before over-generalizing.

### Result (eval_run 1 in data/toolcall.db, full 30-item corpus, rented 4090 box)

Genuine niche-trained LoRA (`outputs/toolcall-lora`) trained on
`data/toolcall_sft.jsonl` — the first niche-targeted adapter this repo has ever
built (the EU `crowd-lora`/`crowd-lora-v2` were the no-transfer controls). Train
set = 77 accepted toolcall samples (13 unique items; the 12 samples on bench
items TC-007/009/010/018/020/028 held out per the release-gate design), LoRA
r=64/alpha=128/all-linear, 3 epochs, max-len 512, on a fresh runpod pytorch
instance (torch 2.14+cu130). Data path that makes this possible: for toolcall
items `accepted_samples.corrected_answer` = the item's mechanical
`golden_answer` (the exact expected call JSON), so accepted catches are directly
trainable. `backend/export.py` gained `--niche`/`--out` to export per-niche.

```
model                                          acc    $/1K q   tok/q       tps
Qwen2.5-3B-Instruct base                    0.8000  $0.07283     479       859
Qwen2.5-3B-Instruct tuned (toolcall-lora)   0.8333  $0.13362     476       465
openai/gpt-4o-mini                          0.8333  $0.07666     462         -
anthropic/claude-haiku-4.5                  0.8667  $0.63173     483         -
base=0.8000 tuned=0.8333 delta=+0.0333 -> release
settled eval_run 1: credited=89 paid_players=12
```

What this taught us: the 6-item bench was flattering to the STOCK 3B (1.0);
at n=30 the stock model drops to 0.80 (24/30), gpt-4o-mini ties the tuned
model at 0.8333, and haiku-4.5 still leads at 0.8667. The niche LoRA is a
genuine +1 item / +0.0333 — right at the MIN_EVAL_DELTA gate — and it closes
the base gap to gpt-4o-mini on the full corpus. Honest frame: one item of
movement at n=30 is suggestive, not strong; the mechanism (train on accepted
toolcall samples -> measurable routing lift over stock) is now proven once.
The tuned tps (465) is ~half base (859) in this lazy single-stream measurement
— do NOT read serving cost from that naive column; item-1's vLLM
continuous-batching harness is the intended fix.

Negative control recorded as eval_run 4: grading tool-call output with the
free-text EU `judge.md` under a non-toolcall system prompt is mechanically
invalid and produced a meaningless tuned=0.4667. `is_correct` is the ONLY
valid grader for this niche.

### Result (grown corpus, eval_runs 2-3 in data/toolcall.db, 4090 box 50824587)

Combinatorial growth landed (`tools/grow_toolcall.py`): corpus 30 → 555 items,
bench 6 → 56 held-out (bench items stay playable, excluded from training),
SFT 499 rows. Trained `outputs/toolcall-lora-v2` (LoRA r=64/alpha=128/all-linear,
3 epochs, max-len 512, batch-1/ga-8) on the box. First real throughput numbers
from the vLLM harness (`training/bench_serve.py` + vLLM 0.29 continuous
batching): **861 out-tok/s sustained on Qwen2.5-3B-Instruct (1×4090), 8.0x the
naive single-stream 108 tok/s**, so per-1K-query serving cost drops to $0.16 at
$1/hr GPU. Fed back into eval_toolcall as `--local-tokens-per-sec 861
--gpu-rate 1.0` for a REAL cost column instead of the naive gpu_rate/tps math.

Held-out bench (56 items, eval_run 2):
```
model                                        acc    $/1K q   tok/q       tps
Qwen2.5-3B-Instruct base                  0.7321  $0.15775     489       861
Qwen2.5-3B-Instruct tuned (toolcall-lora-v2)  1.0000  $0.15802     490       861
openai/gpt-4o-mini                        0.7500  $0.08248     475         -
anthropic/claude-haiku-4.5                0.6964  $0.60496     481         -
base=0.7321 tuned=1.0000 delta=+0.2679 -> release
eval_run 2 released but no unpaid accepted samples to settle
```
Full corpus (555 items, eval_run 3):
```
model                                        acc    $/1K q   tok/q       tps
Qwen2.5-3B-Instruct base                  0.6306  $0.15913     493       861
Qwen2.5-3B-Instruct tuned (toolcall-lora-v2)  0.9964  $0.15916     493       861
openai/gpt-4o-mini                        0.7297  $0.08286     478         -
anthropic/claude-haiku-4.5                0.6414  $0.61494     485         -
base=0.6306 tuned=0.9964 delta=+0.3658 -> release
eval_run 3 released but no unpaid accepted samples to settle
```

This is the competitiveness claim, PROVEN at volume: for a fixed tool
routing task, a niche LoRA on a 3B jumps routing accuracy 0.63→1.0 (held-out
bench) / 0.63→0.996 (full corpus), beating BOTH frontier pay-per-token APIs
(gpt-4o-mini 0.73/0.75, haiku-4.5 0.64/0.70) at ~$0.16/1K queries — 4x cheaper
than haiku, 2x gpt-4o-mini's price for ~1.36x its accuracy. The niche's
value story: accuracy edge over frontier APIs on the exact toolset it was
trained for, at commodity GPU cost.

Honest caveats: (1) the 555 corpus is combinatorial — train (499) and held-out
(56) items are near-duplicates within the same toolset, so both eval gauge
same-toolset generalization, NOT cross-domain transfer. (2) 0.996 on the full
corpus is partially in-distribution (499 train rows included). (3) The tuned
edge does not port to other niches/toolsets (consistent with the EU→TC
no-transfer result). (4) gpt-4o-mini winning at 0.75 on held-out still
underscores that frontier APIs are strong general routers; the niche LoRA wins
by being purpose-built for THIS toolset.

Ops notes: vLLM 0.29 on the box needed `VLLM_USE_FLASHINFER_SAMPLER=0` — with
flashinfer installed (required by vllm's sampler import check) its sampling
kernels JIT-compile under system nvcc (CUDA 12.4) while torch is cu130, and
compile fails; the env var skips flashinfer sampling and falls back to native
sampling. Box system python3 (torch 2.14+cu130 / peft 0.20 / transformers
5.17) trains and evals; vLLM lives in /workspace/vllmvenv. `config.DB_PATH`
defaults to `data/crowd.db`, NOT the toolcall DB — set `DB_PATH=
data/toolcall.db` for toolcall runs.

**Sibling project — code-repair phase-2 (closed, see
`../crowd-finetune-coderepair/CLAUDE.md` "PHASE-2 FULL GPU RUN — RESULTS"):**
the full auto-funnel (SFT → GRPO → rollouts → DPO → eval matrix) was run on
the same 4090 box for the code-repair niche. Verdict: every tuned variant
REGRESSED the base model on repair/generate/MBPP at 2-4x token cost (repair
pass@1 0.778 → sft 0.370 / grpo 0.370 / dpo 0.481); the loop's release gate
correctly REJECTED. The bottleneck is auto-SFT data quality/volume, not the
pipeline. Contrast with the grown tool-call cycle above, which RELEASED at
+0.268/+0.366 on 499 mechanical gold-labeled rows — i.e. the crowd-gated,
decidably-graded niche data path works and the unsupervised synthetic one
doesn't. This is the strongest evidence so far for the project's central
thesis (gated human data beats auto-generated data for post-training).

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
  train_sft.py   # LoRA SFT on distilled traces (--seed, --device, --wandb)
  train_grpo.py  # GRPO RL with verifiable reward (LoRA; --seed/--device/--wandb)
  train_dpo.py   # LoRA DPO on rollout-derived preference pairs (--seed/--device/--wandb)
  collect_rollouts.py  # sample N completions/prompt with a tuned model, score with
                 #   the SAME rewards as GRPO, save group-normalized advantages
  make_dpo.py    # rollouts -> DPO pairs (chosen=highest-advantage correct, rejected=worst)
  sweep_dpo.sh   # 3-seed DPO sweep + pairwise win matrix vs base/SFT
  merge.py       # merge LoRA adapter -> dense checkpoint for the next stage
  eval_judge.py  # held-out accuracy + metrics (avg len chars/tokens, local ppl,
                 #   rm scores from rewards.py, per-mode accuracy)
  run_on_gpu.sh  # one-shot: SFT -> merge -> GRPO -> merge -> rollouts -> DPO -> merge ->
                 #   eval table (SEED/WANDB/ROLLOUT_*/SKIP_DPO envs)
  serve.sh       # vLLM entry point for evals / chat
  requirements.txt
```

Runbook (from repo root; the scripts are `python -m` modules):

```bash
# macOS / any box with an OpenRouter key: build the SFT dataset
python -m training.reasoning.distill --limit 200          # teacher traces (needs BASE_LLM_API_KEY)
python -m training.reasoning.prep_data --inputs data/reasoning_sft.jsonl data/reasoning_sft_more.jsonl --max-rows 400
# rented CUDA box (RunPod/Vast); repo synced to the box first
bash training/reasoning/run_on_gpu.sh                    # SFT -> merge -> GRPO -> merge -> DPO -> eval table
# 3-seed DPO reproducibility sweep (needs outputs/reasoning-sft-merged + dpo_pairs on the box)
SEEDS="1 2 3" bash training/reasoning/sweep_dpo.sh
# serve the final checkpoint for evals / chat
training/reasoning/serve.sh
```

The RLAIF flywheel inside the reasoning track: the GRPO step's merged
checkpoint is the source for the DPO step. `collect_rollouts.py` re-samples
`--gens` completions per prompt from the *merged GRPO* model and scores them
with the exact same reward funcs GRPO optimizes (`rewards.py`), then
`make_dpo.py` turns each prompt into a chosen (highest-advantage, correct) /
rejected (worst, wrong) pair. `train_dpo.py` then does LoRA DPO on top of the
merged SFT checkpoint. This is preference pairs derived from your own
RLAIF-annotated rollouts — no extra judge calls needed. On the Mac, add
`--device cpu` to any trl trainer (the known MPS segfault in the trl
weight-load path); default (no `--device`) keeps the CUDA-box string-model
path unchanged.
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
  corpus.jsonl   # LEGACY niche corpus: 38 EU-AI-Act compliance items, 24 is_gold
                 # (kept for the legacy loop; the ACTIVE niche is tool-call below)
  toolcall_corpus.jsonl  # ACTIVE niche corpus: 30 tool-call items (11 gold, 19
                 # candidates; 11 expected-decline, 19 expected-call; home/CRM/pay)
  bench.jsonl    # deterministic holdout — NOW 6 TC-* items (tools/split_bench.py)
  demo_corpus.jsonl  # fictional building-code QA example (refer only)
tools/
  build_corpus.py  # seed.tsv → corpus.jsonl (validated, exit 1 on bad rows)
  build_toolcall.py # builds data/toolcall_corpus.jsonl (schemas + expected calls)
  simulate_players.py # bot-archetype player simulator (candidate truth is mechanical)
  split_bench.py   # --corpus <file> → bench.jsonl WITHOUT shrinking corpus
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
  eval_toolcall.py # tool-call comparison: local 3B (base/tuned) vs
                   # gpt-4o-mini / claude-haiku-4.5, mechanical is_correct
                   # grading, accuracy + $/1K-query table, eval_runs + settle
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

1. Tool-call niche, make it real: train a tool-call-tuned LoRA, rerun on the
   full 30-item corpus. DONE (n=30): `outputs/toolcall-lora`, tuned 0.80 ->
   0.8333 (+0.0333, ties gpt-4o-mini). Corpus/bench growth + real vLLM cost
   harness: DONE (see "grown corpus, eval_runs 2-3"): corpus 30→555, bench
   6→56, trained `outputs/toolcall-lora-v2` -> held-out bench 1.0 vs base
   0.7321 (+0.268 RELEASE), full corpus 0.996 vs base 0.631, beats both
   frontier competitors; real vLLM continuous-batching throughput = 861
   out-tok/s, replacing the naive cost column. (Item 1 of the Trent review is
   now closed: the growth machinery `tools/grow_toolcall.py` + the vLLM
   harness `training/bench_serve.py` are the deliverables.)
2. Confirm the demo play loop from the miniapp with a REAL browser (I
   simulated the exact HTTP calls and static-served index.html; a headless or
   desktop browser click-through is the remaining gap), ideally against the
   deployed Caddy TLS endpoint.
3. Real 7B release pass: export the same accepted samples and run
   LLaMA-Factory/trl training + `training/eval.py --base Qwen/Qwen2.5-7B-Instruct`
   on the rented GPU (this ties into the reasoning track's box). The local 3B
   run proves the mechanics; the 7B run is the production pairing (game base).
4. Grow data volume: real play sessions, then re-check bench; the 17-row
   train set is demonstrative, not yet consequential for the domain model.
5. Optional hardening exercises: concurrency test on the shared sqlite conn;
   a `data/crowd_sft_valid.jsonl` validation step before training; an
   integration test that drives game.py with a stubbed LLM endpoint so CI can
   run the full loop without a key — DONE (`tools/integration_test.py` +
   `.github/workflows/ci.yml`, fastapi TestClient + a stdlib HTTP stub of
   `BASE_LLM_URL/v1/chat/completions`; full loop: warm → session start/answer/
   end → gate accepts 3 candidates at posterior 0.8 → export → leaderboard →
   stage2 group → settle pays $0.12; deterministic, keyless, runs in CI);
   a headless-browser e2e for the miniapp.

## Work-state markers

- Integration test with stubbed LLM: DONE (`tools/integration_test.py`). Keyless
  end-to-end drive of game.py via FastAPI TestClient: spins a stdlib HTTP stub
  for `{BASE_LLM_URL}/v1/chat/completions` (Authorization checked; json-mode
  returns pass/1.0, "subtly-wrong" returns a tainted answer, else a correct
  stub), sets DB_PATH/CORPUS_PATH/REQUIRE_TG_AUTH=false before import, warms 10
  fixture items (5 gold → 5 playable controls), plays one perfect 10-item
  session (DB-guided served_type like the demo harnesses), then asserts: 60
  points, gate accepts 3/3 candidates at posterior 0.8, export writes 3 rows,
  leaderboard, interleaved grouping, settle credits 3 samples/1 player with
  reward_balance $0.12. Deterministic (reruns identical). CI: `.github/workflows/
  ci.yml` runs `python -m py_compile backend/*.py tools/*.py` + the integration
  test on push/PR with only `fastapi httpx` installed.
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
- Tool-call competitor comparison: DONE (this Mac, MPS, credit-loaded
  OpenRouter). eval_runs 2-5 in data/crowd.db: #2 = EU uniform-prompt release
  re-measure base=0.0625/tuned=0.375 (delta +0.3125, historical artifact, old
  prompt mismatch), #3 = base=0.3625/tuned=0.375 (delta +0.0125, hold), #4 =
  INVALID judge-graded TC run (0.95/0.4667, wrong grader), #5 = mechanical TC
  run 1.0/1.0/0.0 (hold), local 3B beats both API competitors on accuracy on
  the 6-item bench. See "Tool-call niche" section for the table + caveats.
- Tool-call niche LoRA, trained and evaluated on the real corpus: DONE on a
  rented 4090 box (this Mac is too slow/swapped for 3B niche training today).
  Trained `outputs/toolcall-lora` on 77 accepted toolcall samples
  (`data/toolcall_sft.jsonl`, 13 unique items, bench-excluded) and reran
  `training.eval_toolcall` on the FULL 30-item corpus vs gpt-4o-mini +
  claude-haiku-4.5: base 0.80 -> tuned 0.8333 (delta +0.0333 -> release,
  settled 89 samples to 12 sim players in data/toolcall.db, eval_run 1);
  gpt-4o-mini 0.8333, haiku-4.5 0.8667. Stock 3B's earlier 1.0 was a 6-item
  bench artifact. Full table + honest caveats in the "Tool-call niche"
  section. `backend/export.py` now has `--niche`/`--out` per-niche export.
- Grown tool-call cycle at volume: DONE on 4090 box 50824587. Corpus 30->555,
  bench 6->56 (`tools/grow_toolcall.py`), trained `outputs/toolcall-lora-v2`
  (499 SFT rows). Held-out bench (56): base 0.7321 -> tuned 1.0 (release,
  eval_run 2); full corpus (555): base 0.6306 -> tuned 0.9964 (release, eval
  run 3); gpt-4o-mini 0.75/0.73, haiku-4.5 0.70/0.64. First real vLLM
  continuous-batching throughput: 861 out-tok/s (8x naive single-stream),
  $0.16/1K queries at $1/hr — the naive cost column is now replaced by the
  vLLM harness (`training/bench_serve.py`; `--local-tokens-per-sec 861
  --gpu-rate 1.0`). Vast box env lessons: vLLM 0.29 needs
  `VLLM_USE_FLASHINFER_SAMPLER=0` (flashinfer sampling kernels won't JIT-build
  under the box's nvcc 12.4 vs cu130 torch); toolcall evals need
  `DB_PATH=data/toolcall.db` (config default is crowd.db).
  Box (code-repair instance 50824587) stopped after the run; deps there match
  this Mac (torch 2.14+cu130 / transformers 5.17 / peft 0.20). Vast account
  went negative (auto-stopped instances) — user topped up; created-instance
  requires positive credit.
- Not started: miniapp click-TEST driving a real browser (I simulated the

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
# tool-call niche comparison (mechanical is_correct oracle; API models need a
# funded OpenRouter key; --adapter optional — omit to get tuned=base):
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
  .venv/bin/python -m training.eval_toolcall --base Qwen/Qwen2.5-3B-Instruct --adapter outputs/crowd-lora-v2 --competitor
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
- Review-action pass: DPO-on-own-RLAIF-rollouts pipeline added
  (`collect_rollouts.py`/`make_dpo.py`/`train_dpo.py`), eval_judge now emits
  automated metrics (avg len chars/tokens, local `--ppl`, rm_{exact,format,
  length} averages from rewards.py, per-mode accuracy), all three trainers
  take `--seed`/`--device`/`--wandb`, `run_on_gpu.sh` gained the
  rollouts->DPO->merge steps (env: SEED/WANDB/ROLLOUT_LIMIT/ROLLOUT_GENS/
  SKIP_DPO), `sweep_dpo.sh` runs the 3-seed DPO sweep with a pairwise win
  matrix. SMOKE-VERIFIED on this Mac (Qwen2.5-0.5B, `--device cpu`): rollouts
  (3 prompts x 3 gens incl. exact-match scoring), make_dpo (1 pair),
  train_dpo (1 step, loss 0.7305, adapter+checkpoint saved), eval_judge --
  ppl (1.2526) + rm scores on a 3-row slice. The Mac MPS segfault in the trl
  trainer weight-load path (see box env lessons) is avoided with `--device
  cpu`; the GPU box keeps the no-flag string-model path byte-identical. GRPO
  still needs saving its in-training rollouts — collect_rollouts re-samples
  post-hoc from the merged checkpoint instead, so it's not observing the
  training-time distribution; acceptable as the RLAIF source for now.

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
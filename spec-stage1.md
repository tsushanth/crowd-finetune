# Stage 1 spec — crowdsourced adversarial QA + LoRA fine-tune

Goal: prove the loop from GAP (gamified adversarial data) + Crowd-SFT (grouped
competitive fine-tuning) on ONE niche, at minimum cost, in ~3 weeks.

## The loop

candidate corpus → base model answers → judge verifies gold set → tainted
controls generated → Telegram miniapp session → player scores + Bayesian
acceptance filter → accepted samples → LoRA fine-tune → held-out eval gate →
release if delta positive.

Players never see which items are controls. Controls are the reCAPTCHA-style
calibration: reward comes from catching a deliberately-wrong answer on a
control item, so player quality is measured per-session, for free.

## Niche requirement (chosen before running anything)

1. Correctness is decidable and cheap to grade (judge-model + golden answers).
2. Frontier/base model is visibly wrong on a meaningful fraction of items.
3. A crowd exists with intrinsic + extrinsic reason to contribute.
4. Someone will pay for the resulting niche model.

This scaffold ships a demo corpus (fictional building-code QA). Replace it via
`data/corpus.jsonl` before running for real.

## Mechanics (tuned from GAP)

| Param | Value | Notes |
|---|---|---|
| Items per session | 10 (5 controls + 5 candidates) | players blind to split |
| Seconds per item | 120 | urgency, session churn |
| Points per correct catch | 20 | only on tainted controls |
| Accepted threshold θ | 0.8 | P(model wrong \| player marked wrong) |
| Judge pass | score ≥ 0.9 | verdict from judge rubric |
| Laplace smoothing | 1 count | avoids div-by-zero in posterior |
| Cache budget | weekly top-3 cash + 3 random top-10 | extrinsic; XP+leaderboard intrinsic |

## Acceptance filter (from GAP Eq. 8)

Estimate P(M=0 | H=0) on control items each session:

    p = (hits + 1) / (hits + false_alarms + 2)

Accept a candidate as an adversarial sample when the player marked it wrong
AND p >= θ. Control stats are updated only on tainted items. The candidate
side is unevaluable by design — the posterior substitutes for direct
verification. On a large enough funnel you second-stage check accepted
samples against golden answers.

## Sampling / export

Accepted samples → JSONL in `alpaca` format (instruction/output) for
LLaMA-Factory. Fine-tune LoRA on Qwen2.5-7B-Instruct (or smaller niche base).
Every release is gated: eval a held-out bench (items excluded from training),
judge-model-graded, release only if tuned_bench - base_bench >= MIN_EVAL_DELTA.

## The funnel reality

GAP: 50,000 players → ~3,700 accepted QAs (~93% discarded). Expect the same.
Your system's value is the filter + the eval gate, not the raw crowd volume.
Quality concentrates in a small top fraction; that is where Stage 2's
"incentivize only high-quality users" (interleaved grouping + epsilon-greedy
weighting, reward only eval-proven contributions) becomes operational.

## Cost envelope

| Item | ~Cost |
|---|---|
| Judge + base inference (API) | $50-150/mo at small scale |
| LoRA train, rented GPU (spiky) | $20-60 per run |
| Miniapp hosting | ~$0 (static + tiny backend) |
| Crowd rewards | $150/wk (GAP numbers) or $0 pre-PMF |

## Success criteria for Stage 1

1. ≥ 300 accepted samples from a real (non-synthetic) crowd.
2. Held-out bench delta ≥ +0.03 from a LoRA run.
3. Sustained weekly active players without paid marketing (TG virality),
   OR a paid community you can seed.

## Explicit non-goals (Stage 1)

- No token/web3 wiring (keep XP + optional cash).
- No DPO/RLHF (SFT only; DPO is Stage 2+).
- No on-chain anything.
- No multi-niche support (one niche, one model, one eval).

## Stage 2 — reputation, grouping, eval-proven payouts

Mechanisms from Crowd-SFT, wired into the schema and API:

- Per-player `reliability = (hits+1)/(hits+fa+2)` smoothed control-catch accuracy,
  recomputed in `grouping.sync_reliability`.
- Three grouping methods (Crowd-SFT's: interleaved balances strong + new
  contributors on each team; ε-greedy gives a chance of random placement so
  newcomers can prove themselves; random is the control). Groups are recorded
  as `rounds` + `team_assignments` with per-player weights.
- `export_weighted` upsamples high-reliability contributors' samples into the
  training set (weight-capped at 3 copies) instead of flat sampling.
- `reward.settle` credits an eval run with the exact accepted samples that
  shipped in the released dataset (`release_samples`), then pays contributors
  `n_samples × rate × reliability` into `players.reward_balance` and logs
  `payouts`. Whoever's samples measurably improved the model gets paid; nobody
  else does.

API: `POST /stage2/group` `{method, n_groups}`, `POST /stage2/settle`
`{eval_run_id, sample_ids, rate}`.

Crowd-SFT's scaling caveat applies: reward-attribution accuracy drops as the
player pool grows past ~100; the tainted-control gate (not the reputation
engine) remains the primary quality defence.
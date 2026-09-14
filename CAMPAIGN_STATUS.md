# Campaign status — snapshot

As of **2026-09-13**. Read-only audit of all branches and the remote. Working
tree is clean at audit time; all local branches are **in sync with `origin`
(0 behind / 0 ahead)**. Single author across the whole history
(`tsushanth`). Mainline history is `main` = 8 commits; each instance branched
from it.

Instance numbering in commit messages exists for Instance 2 (`niche/toolcall`)
and Instance 4 (`reasoning/hybrid-toggle`); the other tracks are unlabeled.

## Branch map

| Branch | Track | Head | Status | Key outcomes (recorded in commits) |
|---|---|---|---|---|
| `main` | Game/bench/release platform + month-1 reasoning | `1088f6a` | synced | GAP/Crowd-SFT game, control gate (posterior ≥ 0.8), eval-proven payouts, LoRA release gate (3B deltas +0.2875/+0.3125), reasoning SFT (+5 GSM8K), corpus tripled to 38 items |
| `eval/baseline-comparison` | **Instance 1** — EU-AI-Act baseline comparison | `c66f368` | synced | `--competitor` mode in `training/eval.py`: scores GPT-4o-mini + Claude Haiku via `gold.chat()`, grades base/tuned/competitors with the identical `bench_grade` judge + `judge.md`, same system prompt for every model, `$/1K-queries` column. Smoke-verified (0.5B, isolated DB). |
| `niche/toolcall` | **Instance 2** — tool-call correctness | `0d038e6` | synced | Schema-checked tool routing + decline judge; 30-item corpus; mechanical (non-LLM) grading. Recorded result: **stock Qwen2.5-3B 6/6 (1.0) beats GPT-4o-mini and Haiku-4.5 (5/6 each)** on the 6-item bench — but n=6, EU LoRA does not transfer (delta 0.0, correctly held), and gpt-4o-mini is the cost leader. Added DPO-on-own-rollouts pipeline. |
| `niche/code-repair` | Code-repair niche (probable Instance 3) | `af6a6f4` | synced | buggy-fn + failing-tests → patch; AST mutators sandbox-verified; 142 train / 27 eval variants; DPO pipeline; `system`-field + multi-variant bugs fixed. Recorded GPU result: pass@1 **flat 0.52** across base/SFT/GRPO vs gpt-4o-mini 0.68 / haiku 0.56. |
| `reasoning/hybrid-toggle` | **Instance 4** — hybrid think toggle | `f6ce24e` | synced | `/think` vs `/no_think` control flag: `prep_data` emits terse + reasoning rows, `train_sft` renders the flag, `eval_judge` scores both modes with token/cost metrics. |
| `reasoning/ida-loop` | Reasoning iterative loop | `1ad415f` | synced | IDA loop (`train_ida.py`, iterative distill→SFT→GRPO with self-teacher). Month-2 GPU run (~$20.5, both boxes): **3x SFT data + shaped rewards bought zero GSM8K gain** (0.77→0.82→0.79, identical to month-1); shaped rewards did fix the month-1 GRPO format-drift regression; strict MATH-500 now a meaningful 0.39/0.39/0.38; 7B LoRA SFT does not fit a 24GB card (documented non-result). |

## Detail per branch

### `main` — platform + month-1 track
- GAP/Crowd-SFT crowdsourced adversarial-QA loop; acceptance gate
  (`posterior = (hits+1)/(hits+fa+2)`, theta 0.8); release gate
  (`tuned − base >= 0.03`) auto-settles payer credits; sim added 147 accepted
  samples across 14 questions.
- Month-1 reasoning: SFT on 235 distilled GSM8K traces → +5; GRPO on top
  regressed 82→79 (format drift, later fixed in month-2 run).
- Judge-rubric audit (`8e91519`): grade only decisive facts, stop penalizing
  self-volunteered citations; `gold.py` system prompt simplified.

### `eval/baseline-comparison` — Instance 1
- `85c4cfb` (this session): `--competitor` / `--competitors`; one system
  prompt (`gold._BASE_SYSTEM`) for all models; identical `bench_grade` judge
  for all; `$/1K-q` cost column (API = `COMPETITOR_PRICES` × tokens,
  local = `--gpu-rate` ÷ measured tok/s). Default Haiku is
  `anthropic/claude-3-haiku` — `claude-haiku-4.5` returned OpenRouter **402**
  on this account at the time.
- `c66f368` (follow-up): a transient failure on one competitor no longer
  aborts the run / discards already-scored models.
- Verified end-to-end on Qwen2.5-0.5B + 2 bench rows: all three models
  scored, table rendered, `eval_runs` written, hold verdict.

### `niche/toolcall` — Instance 2
- `40a7446`: decline parsing fix (`{"tool": null}` = correct decline),
  30-item corpus, niche dispatch in `gold.warm()`, panoramic bench harness.
- `9722b40` records the comparison result (above) and two important caveats:
  the cost win favors gpt-4o-mini, and cross-niche transfer is a negative
  result. Also capped `gold.chat()` at `max_tokens=512` — the haiku-4.5 402
  was OpenRouter's 64K output ceiling reserving.
- `0d038e6`: DPO pipeline (collect_rollouts → make_dpo → train_dpo → sweep)
  shared with the reasoning track.

### `niche/code-repair`
- Phase 1 (`60dc8e4`): repair variant + sandbox verification (46/51 traces),
  `REPAIR_SYSTEM`, MBPP eval.
- GPU result (`ecb6e92`): SFT loss 0.2118, GRPO reward 0.79→0.83, pass@1 flat
  0.52 vs gpt-4o-mini 0.68 / haiku 0.56.
- Phase 2 (`af6a6f4`): fixed `system`-field bug, multi-variant (142 train / 27
  eval), DPO pipeline. Box `50824587` **STOPPED/unreachable** — restart needed
  before next run, and `training/coderepair/data/` is untracked (rsync, not
  git).

### `reasoning/hybrid-toggle` — Instance 4
- Single commit on top of `main`: think/no-think affordance end-to-end with
  stance-agnostic scoring + token/cost metrics. Branch is a clean 2-commit
  delta from `main` (Instance 4 commit + `1088f6a`).

### `reasoning/ida-loop`
- `4cc3d55` + `ed114b6`: IDA loop plus GPU runbook stability work
  (expandable CUDA alloc, batch2/seq1024, `loss_type=nll`, chunked-CE patch
  disabled).
- `1ad415f`: month-2 GPU run results (above). Takeaway: SFT is the only real
  lever at this scale; more distilled data bought zero additional gain.

## Cross-cutting notes / open risks
- **haiku-4.5 402 had two causes**: (a) OpenRouter output-ceiling reservation —
  fixed by the `max_tokens=512` cap that landed on `niche/toolcall`; that fix
  is not yet on `eval/baseline-comparison` (Instance 1 defaults to
  `claude-3-haiku`). Porting the cap to `gold.py` on `main` would settle it
  for every branch.
- **Untracked runtime data**: `data/*.jsonl` outputs and the
  `training/coderepair/data/` tree are gitignored/untracked; GPU boxes need
  rsync, not git pull.
- **GPU boxes**: several referenced by commits are stopped (code-repair
  `50824587` stopped/unreachable; cogito-3b `50765026` stopped awaiting
  reasoning_sft_more2). Restart before reprising those runs.
- **Nothing merged to `main` except the platform track itself.** All instance
  branches are intact feature branches; cleanup/merge/PR decisions remain open.
- Stale commit note: an early `1541f57` "GPU runbook tuning" commit seen on
  `niche/toolcall` mid-campaign was rebased away/re-organized into
  `reasoning/ida-loop` history — the current graph is internally consistent.

## Plugin: verified this session
- Instance 1 smoke run reproduced: `--competitor` scores base + gpt-4o-mini +
  claude-3-haiku row-by-row, `$/1K-q` column and `eval_runs` insert correct.
- All branches ahead/behind = 0 vs origin at audit time.
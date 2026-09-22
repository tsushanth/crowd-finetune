# Data behind the figures and tables

Reports 1–5's figure CSVs are written by `python build.py` from constants at the top of that script, alongside the matching SVGs, so the two cannot drift apart. The `per_chain_*`, `prm_results_by_model.csv` and `prm_bootstrap_ranges.csv` files come from the scripts in `tools/`. The `grpo_*.csv` files come from `training/reasoning/paired_compare.py` against the pulled eval/diagnostic JSON (not from `build.py`); they were first written for report 6 (4 models), then extended in place for report 7's 3 arms, report 8's IDA round 1, report 9's bigger-PRM arm, report 10's three MATH-domain models, and report 11's three MATH-domain-retry models (15 models total, 30 eval rows, 39 comparison rows, 18 gaming-diagnostic rows), so every one of reports 6–11 reads the same three files rather than each having its own. `build.py`'s figures then read those CSVs directly, rather than re-typing the numbers as constants — see `make_figures_grpo()` (report 6), `make_figures_grpo2()` (report 7), `make_figures_thread_a()` (report 8), `make_figures_thread_b()` (report 9), `make_figures_thread_c()` (report 10) and `make_figures_thread_c2()` (report 11).

| File | What it holds | Used in |
|---|---|---|
| `fig_auroc_arms.csv` | AUROC per PRM variant with 95% ranges, plus the length-only baseline | Report 3 chart |
| `fig_validation.csv` | AUROC of the PRM and the length-only rule on base and SFT model samples | Report 2 chart |
| `fig_label_hist.csv` | Where the first bad step was labelled (counts and percentages), 4 vs 8 rollouts | Report 3 chart |
| `fig_baselines.csv` | SFT accuracy with 95% ranges at n=100 and n=300, average output tokens | Report 4 chart |
| `fig_cost.csv` | Account credit before and after each phase; approximate cost | Report 5 chart |
| `table_earlier_results_n100.csv` | Earlier base / SFT / GRPO accuracy (n=100), checked against the saved result files | Report 4 |
| `table_machines.csv` | Rented machines: location, price, driver, use | Report 5 |
| `per_chain_sft.csv` | One row per SFT-model sample (600): correctness, tokens, steps, and the min and mean step score from each PRM (v0, v1, and v2 with three seeds each) | Reports 2 and 3 |
| `per_chain_base_fallback.csv` | Same for the base model's samples, scored with the fallback splitter | Report 2 |
| `prm_results_by_model.csv` | AUROC (weakest step, mean step, equal step count) for every model, recomputed from the per-chain files | Reports 2 and 3 |
| `prm_bootstrap_ranges.csv` | Bootstrap ranges (resampling questions) for the headline comparisons | Report 3 |
| `grpo_eval_results.csv` | Accuracy and average output tokens (all questions and correct-only), 15 models × 2 benchmarks, full test sets | Reports 6–11 |
| `grpo_paired_comparisons.csv` | McNemar p-values and bootstrap ranges for accuracy and token differences, 39 rows across model pairs × benchmarks (report 10's 3 pairs use GSM8K only, labelled "GSM8K (cross-domain transfer)", since its MATH-500 comparisons are trivial at 0.0 accuracy each; report 11's pairs cover both benchmarks) | Reports 6–11 |
| `grpo_gaming_diagnostics.csv` | Step counts and PRM scores on each model's own eval outputs (reward-gaming check), 18 rows (report 10's 3 MATH-domain models have one MATH-only row each; report 11's 3 models each have a GSM8K row and a MATH-500 row, distinguished in the `model` column since this file has no separate dataset column) | Reports 6–11 |
| `prm_validation_pinned_sft.csv` | PRM validation (report 2's method) re-run on report 6's pinned SFT model | Report 6 |

Report 9's bigger-PRM validation (`training/reasoning/data/val_prm_big.json`, report 9's rescoring of report 6/7's saved validation samples via `--reuse`) is **not** in this tracked directory — it stays with the rest of the untracked raw data under `training/reasoning/data/` and in the Hugging Face raw-data backup, and its summary numbers are quoted directly in report 9's markdown instead of being folded into `prm_validation_pinned_sft.csv`'s schema.

## Notes for reuse

- `chain_id` is the first 10 hex characters of the SHA-1 of the question text, and `sample_idx` numbers the four samples per question. Question text is not included.
- The 596 scorable chains are those with a score. Four of the 600 SFT samples had no parseable steps and have blank scores.
- Base-model scores were recomputed on CPU in 32-bit precision. The original GPU run in bf16 differed in the third decimal (0.719 and 0.704 against 0.720 and 0.703).
- Ranges for the length-only baseline come from `tools/analysis.py` (400 resamples, seed 1). The v1 comparison uses 300 resamples (seed 0).
- Cost figures come from account-credit readings between phases, not billing lines, so they include idle time.
- The accuracy ranges in `fig_baselines.csv` use the normal approximation `p ± 1.96 * sqrt(p(1-p)/n)`.
- Rerunning `tools/extract_per_chain.py` needs the untracked result files in `training/reasoning/data/`; the CSVs here are the exported copies.

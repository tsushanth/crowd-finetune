# PRM track reports

Ten PDF reports, one per thread of work on the process-reward-model (PRM) track, with their markdown sources.

| Report | Thread |
|---|---|
| `01_process_reward_model.pdf` | Building the step-level PRM and wiring it into GRPO |
| `02_validating_on_real_errors.pdf` | Testing the PRM on the model's real mistakes |
| `03_training_on_real_negatives.pdf` | Real-mistake training data and the improved PRM |
| `04_sft_rebaseline.pdf` | SFT re-baseline at n=300 with answer lengths |
| `05_gpu_operations_and_cost.pdf` | Rented GPUs, cost, problems and fixes, repo setup |
| `06_grpo_shorter_chains_experiment.pdf` | The GRPO experiment: does the PRM reward give shorter chains at equal accuracy? (No, at this scale.) |
| `07_length_reward_ablation.pdf` | Follow-up: isolating the pipeline's pre-existing length-shaping reward from the PRM's own effect, plus a seed-replication check |
| `08_ida_round_one.pdf` | One round of self-distillation (IDA): does teaching the model with its own best checkpoint help? (No.) |
| `09_bigger_prm.pdf` | A materially stronger PRM (bigger backbone, more/better-labeled data): does a sharper reward change the GRPO outcome? (No.) |
| `10_math_domain.pdf` | Same pipeline on competition MATH instead of GSM8K: a 12% SFT-distillation yield left every model unable to produce the tagged answer format, so all three score 0.0 on MATH-500 — a data-sufficiency finding, not a test of process rewards on a harder domain |

Reports 8, 9 and 10 ran concurrently on three separate rented GPUs, each reusing the pinned SFT model and/or PRM from reports 6/7 (pulled from Hugging Face) rather than retraining them, to keep the combined session's cost down.

## Rebuilding

Needs `pandoc` and WeasyPrint (`pip install weasyprint`). The stylesheet asks for the Hiragino Sans and Monaco fonts (macOS); other systems fall back to a generic sans-serif.

```
python build.py            # regenerates figs/, data/fig_*.csv and all PDFs
python build.py 09_bigger_prm.md   # one report
```

The data behind every chart and table is in `data/` (see `data/README.md`); `tools/analysis.py` recomputes the AUROCs and bootstrap ranges from the per-chain scores. Reports 6 through 9 share the same `data/grpo_*.csv` files — each new report added rows for its new arms rather than creating separate files — so every report's figures are drawn directly from those CSVs and no numbers are re-typed into `build.py`.

Numbers come from local, untracked result files under `training/reasoning/data/` and from the run logs; see each report's Code Structure section. Report 6's trained weights (SFT adapter, both PRMs, all three GRPO adapters) are on Hugging Face as private model repos under `sushanth9/prm-track-*`. Reports 7, 8 and 9's new adapters were not uploaded before their rental machines were deleted — only the eval outputs and diagnostics survive for those, not the checkpoints. All raw per-question data behind every session (evaluation completions, diagnostics, PRM validation) is backed up as a private Hugging Face dataset, `sushanth9/prm-track-raw-data`.

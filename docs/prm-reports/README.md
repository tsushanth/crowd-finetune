# PRM track reports

Five PDF reports, one per thread of work on the process-reward-model (PRM) track, with their markdown sources.

| Report | Thread |
|---|---|
| `01_process_reward_model.pdf` | Building the step-level PRM and wiring it into GRPO |
| `02_validating_on_real_errors.pdf` | Testing the PRM on the model's real mistakes |
| `03_training_on_real_negatives.pdf` | Real-mistake training data and the improved PRM |
| `04_sft_rebaseline.pdf` | SFT re-baseline at n=300 with answer lengths |
| `05_gpu_operations_and_cost.pdf` | Rented GPUs, cost, problems and fixes, repo setup |

## Rebuilding

Needs `pandoc` and WeasyPrint (`pip install weasyprint`). The stylesheet asks for the Hiragino Sans and Monaco fonts (macOS); other systems fall back to a generic sans-serif.

```
python build.py            # regenerates figs/, data/fig_*.csv and all five PDFs
python build.py 03_training_on_real_negatives.md   # one report
```

The data behind every chart and table is in `data/` (see `data/README.md`); `tools/analysis.py` recomputes the AUROCs and bootstrap ranges from the per-chain scores.

Numbers come from local, untracked result files under `training/reasoning/data/` and from the run logs; see each report's Code Structure section.

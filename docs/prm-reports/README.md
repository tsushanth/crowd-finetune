# PRM track reports

Six PDF reports, one per thread of work on the process-reward-model (PRM) track, with their markdown sources.

| Report | Thread |
|---|---|
| `01_process_reward_model.pdf` | Building the step-level PRM and wiring it into GRPO |
| `02_validating_on_real_errors.pdf` | Testing the PRM on the model's real mistakes |
| `03_training_on_real_negatives.pdf` | Real-mistake training data and the improved PRM |
| `04_sft_rebaseline.pdf` | SFT re-baseline at n=300 with answer lengths |
| `05_gpu_operations_and_cost.pdf` | Rented GPUs, cost, problems and fixes, repo setup |
| `06_grpo_shorter_chains_experiment.pdf` | The GRPO experiment: does the PRM reward give shorter chains at equal accuracy? (No, at this scale.) |

## Rebuilding

Needs `pandoc` and WeasyPrint (`pip install weasyprint`). The stylesheet asks for the Hiragino Sans and Monaco fonts (macOS); other systems fall back to a generic sans-serif.

```
python build.py            # regenerates figs/, data/fig_*.csv and all six PDFs
python build.py 06_grpo_shorter_chains_experiment.md   # one report
```

The data behind every chart and table is in `data/` (see `data/README.md`); `tools/analysis.py` recomputes the AUROCs and bootstrap ranges from the per-chain scores, and report 6's figures are drawn directly from `data/grpo_*.csv` (no numbers are re-typed into `build.py`).

Numbers come from local, untracked result files under `training/reasoning/data/` and from the run logs; see each report's Code Structure section. Report 6's trained weights (SFT adapter, both PRMs, all three GRPO adapters) are on Hugging Face as private model repos under `sushanth9/prm-track-*`.

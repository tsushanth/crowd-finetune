"""Gaming check: step counts and PRM scores of saved eval completions (needs eval_judge --save-text output).

  python -m training.reasoning.score_eval_chains data/full_x_gsm8k.json --prm outputs/prm_real --out data/diag_x.json
"""
import argparse
import json
import statistics as st
from pathlib import Path

import torch

from . import formats, prm as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_json")
    ap.add_argument("--prm", default="outputs/prm")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    rows = json.loads((root / args.eval_json).read_text())["rows"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = P.PRM.load(str(root / args.prm) if (root / args.prm).exists() else args.prm, dev,
                   torch.bfloat16 if dev == "cuda" else torch.float32)
    steps = [P.split_steps(formats.parse_completion(r["text"])["reasoning"]) for r in rows]
    idx = [i for i, s in enumerate(steps) if s]
    probs = m.score_steps([rows[i]["question"] for i in idx], [steps[i] for i in idx], 32)
    score = dict(zip(idx, probs))
    n_steps = [len(s) for s in steps]
    mean = lambda xs: st.mean(xs) if xs else None
    res = {
        "n": len(rows), "n_parsed": len(idx),
        "accuracy": sum(r["ok"] for r in rows) / len(rows),
        "mean_steps": mean(n_steps), "mean_steps_parsed": mean([n_steps[i] for i in idx]),
        "share_le_1_step": sum(1 for i in idx if n_steps[i] <= 1) / max(len(idx), 1),
        "mean_prm_min_parsed": mean([min(score[i]) for i in idx]),
        "mean_prm_min_correct": mean([min(score[i]) for i in idx if rows[i]["ok"]]),
        "mean_prm_min_wrong": mean([min(score[i]) for i in idx if not rows[i]["ok"]]),
        "mean_tokens": mean([r["output_tokens"] for r in rows]),
    }
    (root / args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

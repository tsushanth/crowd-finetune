"""Export per-chain PRM scores behind the AUROC figures to CSV.

Run from the repository root (needs the untracked result files in training/reasoning/data/).
  python docs/prm-reports/tools/extract_per_chain.py          # SFT-model chains (no model needed)
  python docs/prm-reports/tools/extract_per_chain.py --base   # also re-score the base-model chains (needs the PRM in training/reasoning/outputs/prm)
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "training/reasoning/data"
OUT = ROOT / "docs/prm-reports/data"

# column name -> saved rescoring file (all score the same 600 SFT-model samples)
MODELS = {
    "v0_synthetic_cpu": "prm_validation_sft.json",
    "v1_synthetic_top8": "val_prm_syn8.json",
    "v1_real4_top8": "val_prm_real8.json",
    **{f"v2_synthetic_seed{s}": f"val_syn_s{s}.json" for s in range(3)},
    **{f"v2_real8_seed{s}": f"val_real_s{s}.json" for s in range(3)},
}


def chain_ids(rows):
    seen, out = {}, []
    for r in rows:
        h = hashlib.sha1(r["question"].encode()).hexdigest()[:10]
        out.append((h, seen.setdefault(h, 0)))
        seen[h] += 1
    return out


def extract_sft():
    files = {m: json.loads((DATA / f).read_text())["rows"] for m, f in MODELS.items()}
    base = files["v0_synthetic_cpu"]
    for m, rows in files.items():
        assert [(r["question"], r["text"]) for r in rows] == [(r["question"], r["text"]) for r in base], m
    ids = chain_ids(base)
    cols = ["chain_id", "sample_idx", "correct", "output_tokens", "n_steps"]
    for m in MODELS:
        cols += [f"{m}_min", f"{m}_mean"]
    with (OUT / "per_chain_sft.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, r in enumerate(base):
            row = [ids[i][0], ids[i][1], int(r["ok"]), r["tokens"], r.get("n_steps", "")]
            for m in MODELS:
                x = files[m][i]
                row += [round(x["min"], 5) if "min" in x else "", round(x["mean"], 5) if "mean" in x else ""]
            w.writerow(row)
    print("wrote", OUT / "per_chain_sft.csv", len(base), "rows")


def clean(t):
    """Fallback splitter for the base model's markdown/LaTeX output (as used in report 2)."""
    t = re.sub(r"\\[\(\)\[\]]", "", t)
    t = t.replace("**", "").replace("\\text", "").replace("\\times", "*").replace("\\frac", "frac")
    out = []
    for line in t.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", line).strip()
        if len(line) >= 3 and re.search(r"\d", line):
            out.append(line)
    return out


def extract_base():
    sys.path.insert(0, str(ROOT))
    import torch
    from training.reasoning import prm as P

    rows = json.loads((DATA / "prm_validation.json").read_text())["rows"]
    m = P.PRM.load(str(ROOT / "training/reasoning/outputs/prm"), "cpu", torch.float32)
    steps = [clean(r["text"]) for r in rows]
    idx = [i for i, s in enumerate(steps) if s]
    probs = m.score_steps([rows[i]["question"] for i in idx], [steps[i] for i in idx], 16)
    score = dict(zip(idx, probs))
    ids = chain_ids(rows)
    with (OUT / "per_chain_base_fallback.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["chain_id", "sample_idx", "correct", "output_tokens", "n_steps", "v0_synthetic_cpu_min", "v0_synthetic_cpu_mean"])
        for i, r in enumerate(rows):
            p = score.get(i)
            w.writerow([ids[i][0], ids[i][1], int(r["ok"]), r["tokens"], len(steps[i]) or "",
                        round(min(p), 5) if p else "", round(sum(p) / len(p), 5) if p else ""])
    print("wrote", OUT / "per_chain_base_fallback.csv", len(rows), "rows")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    extract_sft()
    if "--base" in sys.argv:
        extract_base()

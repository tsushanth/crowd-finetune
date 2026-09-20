"""Recompute the AUROCs and bootstrap ranges reported in reports 2 and 3 from the per-chain CSVs.

  python docs/prm-reports/tools/analysis.py
Reads  docs/prm-reports/data/per_chain_sft.csv, per_chain_base_fallback.csv
Writes docs/prm-reports/data/prm_results_by_model.csv, prm_bootstrap_ranges.csv
"""
import csv
import random
import statistics as st
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def load(name):
    rows = list(csv.DictReader((DATA / name).open()))
    return rows


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def score_auroc(rows, col, sel=None):
    sel = range(len(rows)) if sel is None else sel
    pos = [float(rows[i][col]) for i in sel if rows[i]["correct"] == "1"]
    neg = [float(rows[i][col]) for i in sel if rows[i]["correct"] == "0"]
    return auroc(pos, neg)


def length_auroc(rows, sel=None):
    sel = range(len(rows)) if sel is None else sel
    pos = [-int(rows[i]["output_tokens"]) for i in sel if rows[i]["correct"] == "1"]
    neg = [-int(rows[i]["output_tokens"]) for i in sel if rows[i]["correct"] == "0"]
    return auroc(pos, neg)


def within_nsteps(rows, col):
    num = den = 0.0
    for n in {r["n_steps"] for r in rows}:
        g = [i for i, r in enumerate(rows) if r["n_steps"] == n]
        pos = [float(rows[i][col]) for i in g if rows[i]["correct"] == "1"]
        neg = [float(rows[i][col]) for i in g if rows[i]["correct"] == "0"]
        if pos and neg:
            num += auroc(pos, neg) * len(pos) * len(neg)
            den += len(pos) * len(neg)
    return num / den


def rng_ci(vals):
    v = sorted(vals)
    return round(v[int(0.025 * len(v))], 3), round(v[int(0.975 * len(v))], 3)


def bootstrap(rows, fn, seed, n_boot):
    byq = {}
    for i, r in enumerate(rows):
        byq.setdefault(r["chain_id"], []).append(i)
    qs = list(byq)
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        sel = [i for q in (rng.choice(qs) for _ in qs) for i in byq[q]]
        vals = fn(sel)
        if vals is not None:
            out.append(vals)
    return out


def main():
    sft_all = load("per_chain_sft.csv")
    rows = [r for r in sft_all if r["v0_synthetic_cpu_min"] != ""]  # the 596 scorable chains
    models = [c[:-4] for c in rows[0] if c.endswith("_min")]
    out = []
    for m in models:
        out.append({"policy": "SFT", "model": m, "n_chains": len(rows),
                    "auroc_min": round(score_auroc(rows, m + "_min"), 4),
                    "auroc_mean": round(score_auroc(rows, m + "_mean"), 4),
                    "auroc_min_equal_step_count": round(within_nsteps(rows, m + "_min"), 4)})
    out.append({"policy": "SFT", "model": "length_only", "n_chains": len(rows),
                "auroc_min": round(length_auroc(rows), 4), "auroc_mean": "", "auroc_min_equal_step_count": ""})
    base = [r for r in load("per_chain_base_fallback.csv") if r["v0_synthetic_cpu_min"] != ""]
    out.append({"policy": "base (fallback splitter)", "model": "v0_synthetic_cpu", "n_chains": len(base),
                "auroc_min": round(score_auroc(base, "v0_synthetic_cpu_min"), 4),
                "auroc_mean": round(score_auroc(base, "v0_synthetic_cpu_mean"), 4),
                "auroc_min_equal_step_count": ""})
    out.append({"policy": "base (fallback splitter)", "model": "length_only", "n_chains": len(base),
                "auroc_min": round(length_auroc(base), 4), "auroc_mean": "", "auroc_min_equal_step_count": ""})
    with (DATA / "prm_results_by_model.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r)

    real = [f"v2_real8_seed{s}_min" for s in range(3)]
    syn = [f"v2_synthetic_seed{s}_min" for s in range(3)]
    mean3 = lambda cols, sel: st.mean(score_auroc(rows, c, sel) for c in cols)

    def v2(sel):
        r, s, l = mean3(real, sel), mean3(syn, sel), length_auroc(rows, sel)
        return r, s, l

    b = bootstrap(rows, v2, seed=1, n_boot=400)
    ranges = [
        ("v2: real negatives (8 rollouts), 3-seed mean", round(mean3(real, None), 3), *rng_ci([x[0] for x in b])),
        ("v2: synthetic only, 3-seed mean", round(mean3(syn, None), 3), *rng_ci([x[1] for x in b])),
        ("length-only baseline", round(length_auroc(rows), 3), *rng_ci([x[2] for x in b])),
        ("difference: real minus synthetic", round(mean3(real, None) - mean3(syn, None), 3), *rng_ci([x[0] - x[1] for x in b])),
        ("difference: real minus length-only", round(mean3(real, None) - length_auroc(rows), 3), *rng_ci([x[0] - x[2] for x in b])),
    ]
    b1 = bootstrap(rows, lambda sel: (score_auroc(rows, "v1_real4_top8_min", sel), score_auroc(rows, "v1_synthetic_top8_min", sel)), seed=0, n_boot=300)
    ranges += [
        ("v1: real negatives (4 rollouts), 1 seed", round(score_auroc(rows, "v1_real4_top8_min"), 3), *rng_ci([x[0] for x in b1])),
        ("v1: synthetic only, 1 seed", round(score_auroc(rows, "v1_synthetic_top8_min"), 3), *rng_ci([x[1] for x in b1])),
        ("v1 difference: real minus synthetic", round(score_auroc(rows, "v1_real4_top8_min") - score_auroc(rows, "v1_synthetic_top8_min"), 3), *rng_ci([x[0] - x[1] for x in b1])),
    ]
    with (DATA / "prm_bootstrap_ranges.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "estimate", "range_low_2.5pct", "range_high_97.5pct"])
        w.writerows(ranges)
    for r in ranges:
        print(r)


if __name__ == "__main__":
    main()

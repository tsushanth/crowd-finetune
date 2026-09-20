"""Build the PRM-track PDFs: figures -> markdown (+glossary) -> pandoc HTML -> WeasyPrint PDF."""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEAL, INK, MUTED, GRID, BAND, PALE = "#0f766e", "#2c3e50", "#7f8c8d", "#e3e8eb", "#eef2f4", "#f0fdf4"
FONT = "Hiragino Sans, Helvetica Neue, Arial, sans-serif"


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<rect width="{w}" height="{h}" fill="white"/>{body}</svg>')


def t(x, y, s, size=11, fill=INK, anchor="start", weight="400"):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>')


def line(x1, y1, x2, y2, stroke=GRID, sw=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}/>'


def dotplot(path, rows, lo, hi, ticks, band=None, xlabel="", width=760, label_w=280, row_h=56, top=48, right_pad=60):
    """rows: (label, sublabel, value, ci_lo, ci_hi, highlight). Axis lo..hi maps to the plot width."""
    x0, x1 = label_w, width - right_pad
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    h = top + row_h * len(rows) + 46
    b = []
    if band:
        b.append(f'<rect x="{sx(band[1]):.1f}" y="{top-10}" width="{sx(band[2])-sx(band[1]):.1f}" '
                 f'height="{row_h*len(rows)+10}" fill="{BAND}"/>')
        b.append(line(sx(band[0]), top - 10, sx(band[0]), top + row_h * len(rows), MUTED, 1.4, "4 3"))
        b.append(t(sx(band[0]), top - 16, band[3], 12, MUTED, "middle"))
    for tk in ticks:
        b.append(line(sx(tk), top - 10, sx(tk), top + row_h * len(rows), GRID))
        b.append(t(sx(tk), top + row_h * len(rows) + 18, f"{tk:.2f}", 12, MUTED, "middle"))
    b.append(t((x0 + x1) / 2, h - 8, xlabel, 11.5, MUTED, "middle"))
    for i, (lab, sub, v, cl, ch, hl) in enumerate(rows):
        y = top + row_h * i + row_h / 2 - 6
        col = TEAL if hl else INK
        b.append(t(label_w - 14, y + 1, lab, 13, INK, "end", "600" if hl else "400"))
        b.append(t(label_w - 14, y + 17, sub, 11.5, MUTED, "end"))
        if cl is not None:
            b.append(line(sx(cl), y, sx(ch), y, col, 2.2))
            b.append(line(sx(cl), y - 6, sx(cl), y + 6, col, 2.2))
            b.append(line(sx(ch), y - 6, sx(ch), y + 6, col, 2.2))
        b.append(f'<circle cx="{sx(v):.1f}" cy="{y}" r="{6 if hl else 5}" fill="{col}"/>')
        txt = f"{v:.3f}" if cl is None else f"{v:.3f}  [{cl:.3f}, {ch:.3f}]"
        b.append(t(sx(ch if ch is not None else v) + 10, y + 4, txt, 11.5, MUTED))
    Path(path).write_text(svg(width, h, "".join(b)))


def hbars(path, groups, lo, hi, ticks, xlabel="", width=760, label_w=250, bar_h=24, gap=9, gap_g=26, top=14, ref=None, fmt=None):
    """groups: [(group title, [(label, value, highlight)])]; bars grow from `lo`."""
    x0, x1 = label_w, width - 70
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    n = sum(len(g[1]) for g in groups)
    h = int(top + n * (bar_h + gap) + len(groups) * (gap_g + 14) + 46)
    b, y = [], top
    plot_top = top
    for title, bars in groups:
        y += 14
        b.append(t(0, y, title, 13, TEAL, "start", "700"))
        y += gap_g - 8
        for lab, v, hl in bars:
            b.append(t(label_w - 12, y + bar_h / 2 + 5, lab, 12.5, INK, "end"))
            b.append(f'<rect x="{x0}" y="{y}" width="{sx(v)-x0:.1f}" height="{bar_h}" fill="{TEAL if hl else "#9bb7b3"}"/>')
            b.append(t(sx(v) + 8, y + bar_h / 2 + 5, fmt(v) if fmt else f"{v:.3f}", 12.5, INK))
            y += bar_h + gap
        y += 4
    for tk in ticks:
        b.insert(0, line(sx(tk), plot_top, sx(tk), y, GRID))
        b.append(t(sx(tk), y + 16, f"{tk:.2f}", 12, MUTED, "middle"))
    if ref:
        b.append(line(sx(ref[0]), plot_top, sx(ref[0]), y, MUTED, 1.4, "4 3"))
        b.append(t(sx(ref[0]) + 4, plot_top + 8, ref[1], 9, MUTED))
    b.append(t((x0 + x1) / 2, y + 36, xlabel, 11.5, MUTED, "middle"))
    Path(path).write_text(svg(width, int(y + 50), "".join(b)))


def vbars2(path, cats, s1, s2, names, ymax, width=760, height=270):
    """Two-series grouped vertical bars of percentages."""
    left, bottom, top = 56, height - 52, 34
    plot_h = bottom - top
    sy = lambda v: bottom - v / ymax * plot_h
    bw, gw = 30, (width - left - 30) / len(cats)
    b = []
    for tk in range(0, int(ymax) + 1, 10):
        b.append(line(left, sy(tk), width - 20, sy(tk), GRID))
        b.append(t(left - 8, sy(tk) + 4, f"{tk}%", 12, MUTED, "end"))
    for i, c in enumerate(cats):
        cx = left + gw * i + gw / 2
        for j, (series, col) in enumerate(((s1, "#9bb7b3"), (s2, TEAL))):
            x = cx - bw - 2 if j == 0 else cx + 2
            b.append(f'<rect x="{x:.1f}" y="{sy(series[i]):.1f}" width="{bw}" height="{bottom - sy(series[i]):.1f}" fill="{col}"/>')
            b.append(t(x + bw / 2, sy(series[i]) - 5, f"{series[i]:.0f}%", 11.5, INK, "middle"))
        b.append(t(cx, bottom + 18, c, 12.5, INK, "middle"))
    b.append(t((left + width - 20) / 2, height - 8, "Index of the first step labelled bad (0 = the very first step)", 11.5, MUTED, "middle"))
    b.append(f'<rect x="{left+10}" y="8" width="12" height="12" fill="#9bb7b3"/>' + t(left + 28, 19, names[0], 11.5, INK))
    b.append(f'<rect x="{left+300}" y="8" width="12" height="12" fill="{TEAL}"/>' + t(left + 318, 19, names[1], 11.5, INK))
    Path(path).write_text(svg(width, height, "".join(b)))


# ---------------------------------------------------------------------------------------------
# Figure data. Each block feeds both the SVG figure and a CSV in data/ (single source of truth).
# ---------------------------------------------------------------------------------------------
# AUROC per PRM variant on the 596 scorable SFT-model chains. Ranges: 95% from resampling questions
# (see tools/analysis.py; v2 rows use 400 resamples seed 1, v1 rows 300 resamples seed 0).
AUROC_ARMS = [  # arm, detail, auroc, ci_low, ci_high, headline
    ("Synthetic negatives only", "3-seed mean", 0.693, 0.641, 0.748, False),
    ("+ real negatives, 4 rollouts", "1 seed, 600 wrong chains", 0.764, 0.707, 0.816, False),
    ("+ real negatives, 8 rollouts", "3-seed mean, 900 wrong chains", 0.800, 0.751, 0.843, True),
]
LENGTH_ONLY = (0.733, 0.676, 0.789)  # auroc, ci_low, ci_high

VALIDATION = [  # policy, score, auroc, highlight (base rows: CPU fp32 rescoring; see data/README.md)
    ("Base model samples (markdown output, fallback step splitter)", "PRM, weakest-step score", 0.720, False),
    ("Base model samples (markdown output, fallback step splitter)", "PRM, mean step score", 0.703, False),
    ("Base model samples (markdown output, fallback step splitter)", "Length-only baseline", 0.766, False),
    ("SFT model samples (matches the PRM's training format)", "PRM, weakest-step score", 0.699, True),
    ("SFT model samples (matches the PRM's training format)", "PRM, mean step score", 0.680, True),
    ("SFT model samples (matches the PRM's training format)", "PRM, weakest-step, equal step count", 0.686, True),
    ("SFT model samples (matches the PRM's training format)", "Length-only baseline", 0.733, False),
]

LABEL_HIST_CATS = ["0", "1", "2", "3", "4", "5+"]
LABEL_COUNTS_4 = [248, 153, 93, 48, 31, 27]   # 600 labelled wrong chains, 4 rollouts per prefix
LABEL_COUNTS_8 = [251, 248, 153, 118, 64, 66]  # 900 labelled wrong chains, 8 rollouts per prefix

BASELINES = [  # test, run, n, accuracy, ci_low, ci_high, avg_tokens, avg_tokens_correct, highlight
    ("GSM8K", "Earlier SFT run", 100, 0.82, 0.745, 0.895, None, None, False),
    ("GSM8K", "Retrained SFT", 100, 0.76, 0.676, 0.844, 104.9, 96.7, False),
    ("GSM8K", "Retrained SFT", 300, 0.797, 0.751, 0.842, 104.9, 100.3, True),
    ("MATH-500 strict", "Retrained SFT", 100, 0.38, 0.285, 0.475, 241.1, 171.9, False),
    ("MATH-500 strict", "Retrained SFT", 300, 0.350, 0.296, 0.404, 259.8, 179.0, True),
]

EARLIER_RESULTS = [  # verified against training/reasoning/data/eval_*_r2.json (n=100 each)
    ("Base", 0.77, 0.39), ("SFT", 0.82, 0.39), ("GRPO, exact-match reward", 0.79, 0.38),
]

COST_PHASES = [  # phase, credit_before, credit_after
    ("PRM check (base + SFT samples)", 25.95, 25.76),
    ("SFT + first baseline evals", 25.76, 25.49),
    ("Real-negative PRM, v1", 25.49, 25.05),
    ("Real-negative PRM, v2 (8 rollouts)", 25.05, 23.86),
    ("SFT re-baseline at n=300", 23.86, 23.53),
]

MACHINES = [  # id, location, price_per_hour_usd, driver, used_for
    (51468149, "Oregon, US", 0.469, "580.159", "PRM check on base samples; SFT run 1; PRM check on SFT samples; first n=100 evals"),
    (51482907, "Illinois, US", 0.367, "580.95", "SFT run 2; real-negative data v1; first PRM arms incl. failed full-training arms"),
    (51496010, "Utah, US", 0.456, "595.71", "v2 pipeline: SFT, 30,296 rollouts, six PRMs and scoring"),
    (51524225, "Pennsylvania, US", 0.481, "615.71", "SFT re-baseline at n=300"),
]


def write_csv(path, header, rows):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def make_figures():
    f = HERE / "figs"
    d = HERE / "data"
    f.mkdir(exist_ok=True)
    d.mkdir(exist_ok=True)
    # Figure: AUROC by PRM variant
    dotplot(f / "auroc_arms.svg", AUROC_ARMS, 0.55, 0.90, [0.6, 0.7, 0.8, 0.9],
            band=(LENGTH_ONLY[0], LENGTH_ONLY[1], LENGTH_ONLY[2], f"length-only baseline {LENGTH_ONLY[0]:.3f}"),
            xlabel="AUROC on 596 SFT-model chains (0.5 = chance); bars are 95% ranges from resampling questions")
    write_csv(d / "fig_auroc_arms.csv", ["variant", "detail", "auroc", "range_low", "range_high", "headline"],
              [(a, b, v, lo, hi, int(h)) for a, b, v, lo, hi, h in AUROC_ARMS]
              + [("Length-only baseline", "shorter answer = right", *LENGTH_ONLY, 0)])
    # Figure: validation on real errors
    groups = {}
    for pol, score, v, hl in VALIDATION:
        groups.setdefault(pol, []).append((score, v, hl))
    hbars(f / "validation.svg", list(groups.items()), 0.5, 0.8, [0.5, 0.6, 0.7, 0.8],
          "AUROC (0.5 = chance, 1.0 = perfect); bars start at chance")
    write_csv(d / "fig_validation.csv", ["policy_samples", "score", "auroc"], [(p, s, v) for p, s, v, _ in VALIDATION])
    # Figure: first-bad-step histogram
    pct = lambda counts: [100.0 * c / sum(counts) for c in counts]
    vbars2(f / "label_hist.svg", LABEL_HIST_CATS, pct(LABEL_COUNTS_4), pct(LABEL_COUNTS_8),
           ["4 rollouts per prefix (600 wrong chains)", "8 rollouts per prefix (900 wrong chains)"], 45)
    write_csv(d / "fig_label_hist.csv",
              ["first_bad_step", "count_4_rollouts", "pct_4_rollouts", "count_8_rollouts", "pct_8_rollouts"],
              [(c, a, round(pa, 2), b, round(pb, 2)) for c, a, pa, b, pb in
               zip(LABEL_HIST_CATS, LABEL_COUNTS_4, pct(LABEL_COUNTS_4), LABEL_COUNTS_8, pct(LABEL_COUNTS_8))])
    # Figure: SFT baselines
    dotplot(f / "baselines.svg",
            [(f"{t}, {r[0].lower() + r[1:]}", f"n={n}", a, lo, hi, hl) for t, r, n, a, lo, hi, _, _, hl in BASELINES],
            0.25, 0.95, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            xlabel="Accuracy with 95% range (normal approximation on the number of questions)", row_h=46, right_pad=190)
    write_csv(d / "fig_baselines.csv",
              ["test", "run", "n", "accuracy", "range_low", "range_high", "range_method", "avg_output_tokens", "avg_output_tokens_correct"],
              [(t, r, n, a, lo, hi, "normal approximation", "" if tok is None else tok, "" if tc is None else tc)
               for t, r, n, a, lo, hi, tok, tc, _ in BASELINES])
    write_csv(d / "table_earlier_results_n100.csv", ["model", "gsm8k_accuracy", "math500_strict_accuracy"], EARLIER_RESULTS)
    # Figure: cost by phase
    costs = [(p, round(b - a, 2), i == 3) for i, (p, b, a) in enumerate(COST_PHASES)]
    hbars(f / "cost.svg", [("Approximate credit spent per phase (USD)", costs)], 0.0, 1.4,
          [0.0, 0.25, 0.5, 0.75, 1.0, 1.25], "USD (from account-credit readings, so includes idle time)",
          label_w=290, fmt=lambda v: f"${v:.2f}")
    write_csv(d / "fig_cost.csv", ["phase", "credit_before_usd", "credit_after_usd", "approx_cost_usd"],
              [(p, b, a, round(b - a, 2)) for p, b, a in COST_PHASES])
    write_csv(d / "table_machines.csv", ["instance_id", "location", "price_usd_per_hour", "driver", "used_for"], MACHINES)


GLOSSARY = {
    "AUROC": "How well a scorer separates right from wrong solutions: the chance it gives a randomly chosen right solution a higher score than a randomly chosen wrong one. 0.5 is a coin flip, 1.0 is perfect.",
    "Baseline": "The score of the model before the new idea is applied. Any improvement has to be measured against it.",
    "Bootstrap": "A way to size the uncertainty of a score: re-draw the test questions at random many times and see how much the score moves. We re-draw questions, not single answers, because several answers share a question.",
    "Confidence interval (95% range)": "The range in which the true value plausibly lies, given the noise in the test. A wide range means we are unsure; heavily overlapping ranges mean we cannot claim one result is better.",
    "Corruption (synthetic negative)": "A fake mistake planted in a correct solution (a changed result, a swapped number, a flipped operator) to teach the PRM what bad working looks like.",
    "Exact-match reward": "1 point if the final number equals the correct answer, otherwise 0. It only looks at the end, so it is an outcome reward.",
    "Fine-tuning": "Further training of an existing model on a specific job instead of training from scratch.",
    "GRPO": "The reinforcement-learning recipe used in this pipeline. For each question the model writes several attempts, they are scored, and it is nudged toward the better-scoring ones.",
    "GSM8K": "A standard test set of grade-school maths word problems.",
    "IDA": "Iterated distillation and amplification: a loop of teaching a model, letting it practise, and using the improved model to write better teaching material.",
    "LoRA / adapter": "A small add-on file that records only the changes made by fine-tuning, instead of a full copy of the model.",
    "MATH-500": "500 harder competition-style maths problems. In strict mode the answer must appear in the required tag format or it counts as wrong.",
    "Monte-Carlo rollout": "Letting the model finish a solution from a given point, several times, and checking how often it reaches the right answer. Used to find which step of a wrong solution went wrong.",
    "n": "The number of test questions. Small tests are noisy, like judging a coin from a few flips.",
    "Outcome vs process reward": "An outcome reward marks only the final answer. A process reward marks every line of working.",
    "PRM (process reward model)": "A small model that reads a question and the working so far and outputs, after each step, the probability that the step is correct.",
    "Reward": "The score given to an attempt during reinforcement learning. The model drifts toward whatever the reward favours, including loopholes.",
    "Seed": "The random starting point of a training run. Several seeds giving similar scores shows a result is not a fluke.",
    "SFT": "Supervised fine-tuning: showing the model worked examples (question, then written solution) so it learns to imitate them.",
    "Step": "One line or sentence of working. Solutions are split into steps automatically at line and sentence breaks.",
    "Token": "The unit a model reads and writes, roughly three-quarters of a word. Average output tokens is our measure of answer length.",
    "Trace": "A written-out step-by-step solution used as a training example.",
    "Length-only baseline": "A deliberately lazy scorer that says the shorter the answer, the likelier it is right. A real scorer has to beat it or it is just measuring length.",
    "Vast": "The GPU rental marketplace used for the runs. Machines are rented by the hour and deleted after each job.",
}


def glossary_table(names):
    rows = "\n".join(f"| {n} | {GLOSSARY[n]} |" for n in names)
    return "| Term | Plain-language meaning |\n|---|---|\n" + rows + "\n"


def build(md_path):
    src = md_path.read_text()
    src = re.sub(r"<!--GLOSSARY:(.*?)-->", lambda m: glossary_table([s.strip() for s in m.group(1).split(";")]), src)
    tmp = md_path.with_suffix(".expanded.md")
    tmp.write_text(src)
    html = md_path.with_suffix(".html")
    subprocess.run(["pandoc", str(tmp), "--standalone", f"--template={HERE/'template.html'}", "--toc", "--toc-depth=2",
                    "--number-sections", "--resource-path", str(HERE), "-o", str(html)], check=True)
    pdf = HERE / (md_path.stem + ".pdf")
    subprocess.run([sys.executable, "-m", "weasyprint", str(html), str(pdf)], check=True, cwd=str(HERE),
                   stderr=subprocess.DEVNULL)
    tmp.unlink()
    html.unlink()
    return pdf


if __name__ == "__main__":
    make_figures()
    targets = [HERE / a for a in sys.argv[1:]] or sorted(HERE.glob("0[1-5]_*.md"))
    for md in targets:
        print("built", build(md))

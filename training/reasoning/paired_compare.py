"""Paired comparison of two eval_judge outputs on the same questions.

  python -m training.reasoning.paired_compare A.json B.json [--label-a SFT --label-b GRPO]
Accuracy: exact McNemar test on discordant pairs, plus a question-level bootstrap range for the difference.
Length: paired difference in output tokens (all questions, and questions both models got right), bootstrap range.
Files must come from the same test set and --limit (row i = question i); pass --save-text runs so rows are checked.
"""
import argparse
import json
import math
import random
import statistics as st


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def ci(vals):
    v = sorted(vals)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


def compare(a_rows, b_rows, n_boot=2000, seed=0):
    assert len(a_rows) == len(b_rows), "different number of questions"
    if "question" in a_rows[0] and "question" in b_rows[0]:
        assert all(x["question"] == y["question"] for x, y in zip(a_rows, b_rows)), "question order differs"
    n = len(a_rows)
    ao = [int(r["ok"]) for r in a_rows]
    bo = [int(r["ok"]) for r in b_rows]
    at = [r["output_tokens"] for r in a_rows]
    bt = [r["output_tokens"] for r in b_rows]
    b_only = sum(1 for x, y in zip(ao, bo) if y and not x)   # B right, A wrong
    a_only = sum(1 for x, y in zip(ao, bo) if x and not y)
    both = [i for i in range(n) if ao[i] and bo[i]]
    rng = random.Random(seed)
    d_acc, d_tok, d_tok_both = [], [], []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        d_acc.append(sum(bo[i] - ao[i] for i in idx) / n)
        d_tok.append(sum(bt[i] - at[i] for i in idx) / n)
        both_idx = [i for i in idx if ao[i] and bo[i]]
        if both_idx:
            d_tok_both.append(sum(bt[i] - at[i] for i in both_idx) / len(both_idx))
    return {
        "n": n,
        "acc_a": sum(ao) / n, "acc_b": sum(bo) / n,
        "acc_diff_b_minus_a": sum(bo) / n - sum(ao) / n, "acc_diff_range": ci(d_acc),
        "mcnemar_b_only": b_only, "mcnemar_a_only": a_only, "mcnemar_p": binom_two_sided(min(a_only, b_only), a_only + b_only),
        "tok_a": st.mean(at), "tok_b": st.mean(bt),
        "tok_diff_b_minus_a": st.mean(bt) - st.mean(at), "tok_diff_range": ci(d_tok),
        "n_both_right": len(both),
        "tok_both_a": st.mean(at[i] for i in both) if both else None,
        "tok_both_b": st.mean(bt[i] for i in both) if both else None,
        "tok_both_diff_b_minus_a": (st.mean(bt[i] - at[i] for i in both)) if both else None,
        "tok_both_diff_range": ci(d_tok_both) if d_tok_both else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()
    a = json.load(open(args.a))["rows"]
    b = json.load(open(args.b))["rows"]
    r = compare(a, b)
    f = lambda x, d=3: "n/a" if x is None else f"{x:.{d}f}"
    print(f"{args.label_b} minus {args.label_a}  (n={r['n']})")
    print(f"  accuracy   {f(r['acc_a'])} -> {f(r['acc_b'])}  diff {r['acc_diff_b_minus_a']:+.3f}  95% range [{r['acc_diff_range'][0]:+.3f}, {r['acc_diff_range'][1]:+.3f}]"
          f"  McNemar p={r['mcnemar_p']:.3f} ({r['mcnemar_b_only']} only-B vs {r['mcnemar_a_only']} only-A)")
    print(f"  tokens     {f(r['tok_a'], 1)} -> {f(r['tok_b'], 1)}  diff {r['tok_diff_b_minus_a']:+.1f}  95% range [{r['tok_diff_range'][0]:+.1f}, {r['tok_diff_range'][1]:+.1f}]")
    if r["tok_both_diff_range"]:
        print(f"  tokens, both right (n={r['n_both_right']})  {f(r['tok_both_a'], 1)} -> {f(r['tok_both_b'], 1)}  diff {r['tok_both_diff_b_minus_a']:+.1f}  95% range [{r['tok_both_diff_range'][0]:+.1f}, {r['tok_both_diff_range'][1]:+.1f}]")


if __name__ == "__main__":
    main()

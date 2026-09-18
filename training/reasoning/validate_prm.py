"""Does the PRM separate REAL correct from REAL wrong model solutions?

Samples n solutions per GSM8K-test question from a policy, scores each chain with the
PRM, labels by final-answer exact match, and reports AUROC of the chain score for
predicting correctness (plus a length-only baseline, so a gain over 'shorter = right'
is visible).
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import formats, prm as P
from .train_prm import auroc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--prm", default="outputs/prm")
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--count", type=int, default=150)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-system", action="store_true", help="omit the tag-format system prompt (use for SFT models)")
    ap.add_argument("--output", default="data/prm_validation.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.start, args.start + args.count))

    tok = AutoTokenizer.from_pretrained(args.policy, padding_side="left")
    pol = AutoModelForCausalLM.from_pretrained(args.policy, torch_dtype=torch.bfloat16).to(dev).eval()
    jobs = []
    for ex in ds:
        prompt = tok.apply_chat_template(
            ([] if args.no_system else [{"role": "system", "content": formats.TEACHER_SYSTEM}])
            + [{"role": "user", "content": ex["question"]}],
            tokenize=False, add_generation_prompt=True)
        ref = formats.extract_last_number(ex["answer"].split("####")[-1])
        jobs += [(ex["question"], prompt, ref)] * args.samples

    rows = []
    for i in range(0, len(jobs), args.batch):
        chunk = jobs[i:i + args.batch]
        enc = tok([j[1] for j in chunk], return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            out = pol.generate(**enc, max_new_tokens=args.max_new, do_sample=True,
                               temperature=args.temperature, top_p=0.95)
        gen = out[:, enc["input_ids"].shape[1]:]
        for (q, _, ref), g in zip(chunk, gen):
            text = tok.decode(g, skip_special_tokens=True)
            rows.append({"question": q, "text": text, "ok": formats.exact_match(text, ref),
                         "tokens": int((g != tok.pad_token_id).sum())})
        print(f"generated {len(rows)}/{len(jobs)}", flush=True)
    del pol
    torch.cuda.empty_cache()

    m = P.PRM.load(str(root / args.prm) if (root / args.prm).exists() else args.prm, dev, torch.bfloat16)
    steps = [P.split_steps(formats.parse_completion(r["text"])["reasoning"]) for r in rows]
    scored = [(r, s) for r, s in zip(rows, steps) if s]  # untagged/empty chains are excluded from AUROC
    probs = m.score_steps([r["question"] for r, _ in scored], [s for _, s in scored])
    for (r, s), p in zip(scored, probs):
        r["n_steps"], r["min"], r["mean"] = len(s), min(p), sum(p) / len(p)

    sc = [r for r, _ in scored]
    right = [r for r in sc if r["ok"]]
    wrong = [r for r in sc if not r["ok"]]
    # length-controlled: AUROC within groups of equal step count, pair-weighted, so the PRM
    # cannot win just by scoring long chains lower
    num = den = 0.0
    for n in sorted({r["n_steps"] for r in sc}):
        g_r = [r["min"] for r in right if r["n_steps"] == n]
        g_w = [r["min"] for r in wrong if r["n_steps"] == n]
        if g_r and g_w:
            num += auroc(g_r, g_w) * len(g_r) * len(g_w)
            den += len(g_r) * len(g_w)
    res = {
        "auroc_min_within_nsteps": num / den if den else float("nan"),
        "pairs_within_nsteps": den,
        "policy": args.policy, "n_generated": len(rows), "n_parsed": len(sc),
        "acc_parsed": len(right) / max(len(sc), 1), "n_right": len(right), "n_wrong": len(wrong),
        "auroc_min": auroc([r["min"] for r in right], [r["min"] for r in wrong]),
        "auroc_mean": auroc([r["mean"] for r in right], [r["mean"] for r in wrong]),
        "auroc_shorter_is_right": auroc([-r["tokens"] for r in right], [-r["tokens"] for r in wrong]),
        "mean_min_right": sum(r["min"] for r in right) / max(len(right), 1),
        "mean_min_wrong": sum(r["min"] for r in wrong) / max(len(wrong), 1),
        "avg_tokens": sum(r["tokens"] for r in rows) / len(rows),
    }
    print(json.dumps(res, indent=2))
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": res, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()

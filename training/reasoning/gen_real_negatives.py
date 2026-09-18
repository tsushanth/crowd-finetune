"""Real PRM training data from the policy's own samples on GSM8K *train*.

Right chains -> all steps 1. Wrong chains -> Math-Shepherd-style step labels: roll out
`--rollouts` completions from each prefix; the first prefix from which NO rollout reaches
the reference answer marks the first bad step (label 0), later steps are masked (-100).
Questions already in the SFT/PRM trace files are excluded so the policy has not seen them.
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import formats, prm as P


def gen(pol, tok, prompts, n_new, temp, batch, dev):
    outs = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            o = pol.generate(**enc, max_new_tokens=n_new, do_sample=True, temperature=temp, top_p=0.95)
        outs += tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  gen {len(outs)}/{len(prompts)}", flush=True)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--exclude", nargs="+", default=[
        "data/reasoning_sft.jsonl", "data/reasoning_sft_more.jsonl",
        "data/reasoning_sft_more2.jsonl", "data/reasoning_sft_more3.jsonl"])
    ap.add_argument("--questions", type=int, default=800)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--max-wrong", type=int, default=600)
    ap.add_argument("--max-right-per-q", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--output", default="data/prm_real.jsonl")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seen = {r["question"] for r in P.dedupe_rows([root / p for p in args.exclude if (root / p).exists()])}
    ds = load_dataset("openai/gsm8k", "main", split="train")
    pool = [ex for ex in ds if ex["question"] not in seen][: args.questions]
    print(f"{len(seen)} excluded questions; using {len(pool)} fresh train questions", flush=True)

    tok = AutoTokenizer.from_pretrained(args.policy, padding_side="left")
    pol = AutoModelForCausalLM.from_pretrained(args.policy, torch_dtype=torch.bfloat16).to(dev).eval()
    chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)

    jobs = []
    for ex in pool:
        ref = formats.extract_last_number(ex["answer"].split("####")[-1])
        jobs += [(ex["question"], ref)] * args.samples
    texts = gen(pol, tok, [chat(q) for q, _ in jobs], 384, args.temperature, args.batch, dev)

    out, wrong, right_count = [], [], {}
    for (q, ref), t in zip(jobs, texts):
        steps = P.split_steps(formats.parse_completion(t)["reasoning"])
        if not steps:
            continue
        if formats.exact_match(t, ref):
            if right_count.get(q, 0) < args.max_right_per_q:
                right_count[q] = right_count.get(q, 0) + 1
                out.append({"question": q, "steps": steps, "labels": [1] * len(steps), "kind": "real_pos"})
        else:
            wrong.append((q, ref, steps))
    print(f"{len(out)} right chains kept, {len(wrong)} wrong chains", flush=True)
    wrong = wrong[: args.max_wrong]

    # MC rollouts from prefixes 1..n-1 of each wrong chain
    ro = []
    for wi, (q, ref, steps) in enumerate(wrong):
        for i in range(1, len(steps)):
            ro.append((wi, i, chat(q) + formats.REASONING_OPEN + "\n" + " ".join(steps[:i]) + " "))
    print(f"{len(ro)} prefixes x {args.rollouts} rollouts", flush=True)
    prompts = [p for _, _, p in ro for _ in range(args.rollouts)]
    comps = gen(pol, tok, prompts, 256, args.temperature, args.batch, dev)
    good = {}
    for k, (wi, i, p) in enumerate(ro):
        ref = wrong[wi][1]
        prefix = formats.REASONING_OPEN + "\n" + " ".join(wrong[wi][2][:i]) + " "
        hit = any(formats.exact_match(prefix + c, ref) for c in comps[k * args.rollouts:(k + 1) * args.rollouts])
        good[(wi, i)] = hit
    n_first = {}
    for wi, (q, ref, steps) in enumerate(wrong):
        e = len(steps) - 1  # default: last step is the first bad one
        for i in range(1, len(steps)):
            if not good[(wi, i)]:
                e = i - 1
                break
        n_first[e] = n_first.get(e, 0) + 1
        out.append({"question": q, "steps": steps, "labels": [1] * e + [0] + [-100] * (len(steps) - e - 1),
                    "kind": "real_neg"})
    print("first-bad-step index histogram:", dict(sorted(n_first.items())), flush=True)
    dest = root / args.output
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(json.dumps(x) for x in out) + "\n")
    print(f"wrote {len(out)} chains -> {dest}")


if __name__ == "__main__":
    main()

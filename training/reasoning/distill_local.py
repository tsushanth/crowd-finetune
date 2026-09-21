"""Batched local self-distillation: sample a teacher (any local HF/merged model) on a
question pool, verify against the reference answer, write only the verified traces.

Unlike distill.py's ThreadPoolExecutor (built for API teachers), this batches generation
on one GPU so a *local* teacher (e.g. last round's own champion, for an IDA round) doesn't
serialize N threads onto one CUDA context for no benefit.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import formats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="local model dir (merged) to sample from")
    ap.add_argument("--pool", required=True, help="jsonl with question/answer (GSM8K raw '#### n' format ok)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    pool = [json.loads(l) for l in (root / args.pool).read_text().splitlines() if l.strip()]

    tok = AutoTokenizer.from_pretrained(args.teacher, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": formats.TEACHER_SYSTEM}, {"role": "user", "content": e["question"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for e in pool
    ]

    kept = []
    for i in range(0, len(prompts), args.batch):
        chunk_ex, chunk_p = pool[i:i + args.batch], prompts[i:i + args.batch]
        enc = tok(chunk_p, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True, temperature=args.temperature)
        gen = out[:, enc["input_ids"].shape[1]:]
        for ex, g in zip(chunk_ex, gen):
            text = tok.decode(g, skip_special_tokens=True)
            parsed = formats.parse_completion(text)
            if not parsed["ok"] or not parsed["reasoning"]:
                continue
            ref = formats.extract_last_number(ex["answer"])
            if ref and formats.exact_match(text, ref):
                kept.append({"question": ex["question"], "reasoning": parsed["reasoning"], "answer": parsed["answer"]})
        print(f"distilled {min(i + args.batch, len(pool))}/{len(pool)}, kept {len(kept)}", flush=True)

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in kept) + "\n")
    print(f"kept {len(kept)}/{len(pool)} verified traces -> {out}")


if __name__ == "__main__":
    main()

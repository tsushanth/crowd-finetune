import argparse
import json
from pathlib import Path

import torch

from . import formats, rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--config-name", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--gens", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/grpo_rollouts.jsonl")
    args = parser.parse_args()

    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    root = Path(__file__).resolve().parent
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if args.adapter:
        candidate = root / args.adapter
        adapter = str(candidate) if candidate.exists() else args.adapter
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def prepare(example):
        question = example["question"]
        reference = formats.extract_last_number(example["answer"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "question": question, "reference": reference}

    ds = load_dataset(args.dataset, args.config_name, split=args.split)
    ds = ds.select(range(min(args.limit, len(ds)))).map(prepare)
    ds = ds.filter(lambda ex: bool(ex["reference"]))
    print(f"{len(ds)} rollout prompts")

    gen_kwargs = dict(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        num_return_sequences=args.gens,
        max_new_tokens=args.max_new,
        pad_token_id=tokenizer.eos_token_id,
    )

    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for i, ex in enumerate(ds):
            prompt = ex["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            completions = [
                tokenizer.decode(o[inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                for o in out
            ]
            prompts = [prompt] * len(completions)
            answers = [ex["reference"]] * len(completions)
            exact = rewards.exact_match_reward(prompts, completions, answer=answers)
            format_scores = rewards.format_reward(prompts, completions)
            lengths = rewards.length_reward(prompts, completions)
            scores = [e + f + l for e, f, l in zip(exact, format_scores, lengths)]
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            std = variance**0.5 or 1.0
            item = {
                "question": ex["question"],
                "prompt": prompt,
                "reference": ex["reference"],
                "completions": [
                    {
                        "text": c,
                        "scores": {"exact": e, "format": f, "length": l},
                        "score": s,
                        "advantage": (s - mean) / std,
                    }
                    for c, e, f, l, s in zip(
                        completions, exact, format_scores, lengths, scores
                    )
                ],
            }
            fh.write(json.dumps(item) + "\n")
            n_ok = sum(1 for c in item["completions"] if c["scores"]["exact"])
            print(
                f"[{i + 1}/{len(ds)}] {n_ok}/{args.gens} exact on {out_path}"
            )
    print(f"rollouts -> {out_path}")


if __name__ == "__main__":
    main()
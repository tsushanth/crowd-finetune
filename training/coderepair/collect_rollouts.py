import argparse
import json
from pathlib import Path

import torch

from . import formats, rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--data", default="data/code_repair_eval.jsonl")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--gens", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/repair_rollouts.jsonl")
    args = parser.parse_args()

    from datasets import Dataset
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

    data_path = root / args.data if (root / args.data).exists() else args.data
    rows = [
        json.loads(line)
        for line in Path(data_path).read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    def prepare(row):
        system = row.get("system") or formats.REPAIR_SYSTEM
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": row["question"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt": prompt,
            "question": row["question"],
            "tests": row["tests"],
            "entry_point": row.get("entry_point"),
            "imports": row.get("imports") or [],
        }

    ds = Dataset.from_list([prepare(r) for r in rows])
    print(f"{len(ds)} rollout prompts")

    gen_kwargs = dict(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
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
                kwargs = dict(num_return_sequences=args.gens, **gen_kwargs)
                out = model.generate(**inputs, **kwargs)
            completions = [
                tokenizer.decode(
                    o[inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                for o in out
            ]
            prompts = [prompt] * len(completions)
            tests = [ex["tests"]] * len(completions)
            entry_points = [ex["entry_point"]] * len(completions)
            imports = [ex["imports"]] * len(completions)

            test_scores = rewards.test_pass_reward(
                prompts,
                completions,
                tests=tests,
                entry_point=entry_points,
                imports=imports,
                hard=True,
            )
            format_scores = rewards.code_format_reward(prompts, completions)
            scores = [t + f for t, f in zip(test_scores, format_scores)]
            mean = sum(scores) / len(scores)
            std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5 or 1.0
            n_ok = sum(1 for s in test_scores if s >= 1.0)
            print(f"[{i + 1}/{len(ds)}] {n_ok}/{len(completions)} exact on {out_path}")

            item = {
                "question": ex["question"],
                "prompt": prompt,
                "completions": [
                    {
                        "text": c,
                        "scores": {"test": t, "format": f},
                        "score": s,
                        "advantage": (s - mean) / std,
                    }
                    for c, t, f, s in zip(
                        completions, test_scores, format_scores, scores
                    )
                ],
            }
            fh.write(json.dumps(item) + "\n")
            fh.flush()
    print(f"rollouts -> {out_path}")


if __name__ == "__main__":
    main()

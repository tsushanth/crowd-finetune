import argparse
import functools
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from . import formats, rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", default="data/code_sft.jsonl")
    parser.add_argument("--output", default="outputs/code-grpo")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--gens", type=int, default=8)
    parser.add_argument("--max-completion", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=4.0)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument(
        "--reward",
        choices=["test", "test+format"],
        default="test+format",
        help="test = unit-test fraction only; test+format = add a small "
        "formatting bonus for <reasoning>/<answer> structure",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu"], default="cpu",
        help="cpu avoids transformers 5.17's MPS default (MPS + pin_memory "
        "segfaults on this Mac); auto keeps the default device",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output_dir = root / args.output

    def resolve(p):
        candidate = root / p
        return str(candidate) if candidate.exists() else p

    base_model = resolve(args.base)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def prepare(example):
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": formats.CODE_SYSTEM},
             {"role": "user", "content": example["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt": prompt,
            "tests": example["tests"],
            "entry_point": example.get("entry_point"),
            "imports": example.get("imports") or [],
        }

    rows = [
        json.loads(line)
        for line in (root / args.data).read_text().splitlines()
        if line.strip()
    ][: args.limit]
    if not rows:
        raise SystemExit("no RL rows; run datasets.py + distill.py first")
    ds = Dataset.from_list(rows).map(prepare)
    ds = ds.filter(lambda ex: bool(ex["entry_point"]) and len(ex["tests"]) > 0)
    print(f"{len(ds)} RL rows")

    kwargs = {"hard": args.hard, "timeout": args.timeout}
    reward_funcs = [
        functools.partial(
            rewards._reward_call, reward=rewards.test_pass_reward, **kwargs
        ),
    ]
    if args.reward == "test+format":
        reward_funcs.append(rewards.code_format_reward)

    grpo_config = GRPOConfig(
        use_cpu=args.device == "cpu",
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.gens,
        generation_batch_size=args.batch * args.gens,
        max_completion_length=args.max_completion,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        beta=0.04,
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
    )
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
    )

    trainer = GRPOTrainer(
        model=base_model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    print(f"checkpoint under -> {output_dir}")


if __name__ == "__main__":
    main()
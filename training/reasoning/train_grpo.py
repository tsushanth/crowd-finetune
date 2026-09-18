import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from . import formats, rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--config-name", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--output", default="outputs/reasoning-grpo")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--gens", type=int, default=8)
    parser.add_argument("--max-completion", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--reward", choices=["outcome", "prm", "prm+outcome"], default="outcome",
                        help="outcome = exact-match (original); prm = step-level PRM replaces exact-match; "
                        "prm+outcome = both")
    parser.add_argument("--prm", default="outputs/prm", help="PRM dir (relative to this package or absolute)")
    parser.add_argument("--prm-aggregate", choices=["min", "mean"], default="min")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output_dir = root / args.output

    def resolve(p):
        candidate = root / p
        return str(candidate) if candidate.exists() else p

    base_model = resolve(args.base)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def prepare(example):
        question = example["question"]
        reference = formats.extract_last_number(example["answer"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "answer": reference}

    ds = load_dataset(args.dataset, args.config_name, split=args.split)
    ds = ds.select(range(min(args.limit, len(ds)))).map(prepare)
    ds = ds.filter(lambda ex: bool(ex["answer"]))
    print(f"{len(ds)} RL rows")

    grpo_config = GRPOConfig(
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

    reward_funcs = [rewards.format_reward, rewards.length_reward]
    if args.reward in ("outcome", "prm+outcome"):
        reward_funcs.insert(0, rewards.exact_match_reward)
    if args.reward in ("prm", "prm+outcome"):
        reward_funcs.insert(0, rewards.make_prm_reward(resolve(args.prm), args.prm_aggregate))
    print(f"reward mode: {args.reward} -> {[f.__name__ for f in reward_funcs]}")

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
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from . import formats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", default="data/reasoning_sft.jsonl")
    parser.add_argument("--output", default="outputs/reasoning-sft")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument(
        "--control-style",
        choices=["system", "token"],
        default="system",
        help="how the /think vs /no_think control flag reaches the model; must "
        "match eval_judge.py and how the checkpoint is served",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_path = root / args.data
    output_dir = root / args.output

    rows = [
        json.loads(line)
        for line in data_path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("no training rows; run distill.py first")
    hybrid = any("control" in r for r in rows)
    counts = Counter(
        r.get("control", formats.CONTROL_THINK) for r in rows
    ) if hybrid else None
    print(f"{len(rows)} training rows from {data_path} {counts or '(single-mode)'}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)

    def build_text(example):
        control = example.get("control", formats.CONTROL_THINK)
        messages = []
        system = formats.render_system(control, args.control_style)
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": formats.render_control(
                    control, example["question"], args.control_style
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": formats.build_control_completion(
                    control, example.get("reasoning", ""), example["answer"]
                ),
            }
        )
        return {
            "text": tokenizer.apply_chat_template(messages, tokenize=False)
        }

    ds = Dataset.from_list(rows).map(build_text)
    tokenizer.pad_token = tokenizer.eos_token

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        max_length=args.seq_length,
        truncation_mode="keep_start",
        dataset_text_field="text",
        packing=False,
        logging_steps=10,
        save_steps=200,
        max_grad_norm=1.0,
    )
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
    )

    trainer = SFTTrainer(
        model=args.base,
        args=sft_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"saved adapter -> {output_dir}")


if __name__ == "__main__":
    main()